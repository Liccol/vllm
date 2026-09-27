# Prefix Execution Artifact Store：PR Roadmap ZH

## 最终目标



只保留两种 Artifact Connector backend：



```Plain Text
简单模式
GPU worker capture
  -> ArtifactConnector
  -> 本地 SHM backend
  -> 组装实际 R3 / prompt logprobs value
  -> ModelRunnerOutput
  -> EngineCore / HTTP

生产模式
GPU worker capture
  -> ArtifactConnector（唯一 producer）
  -> Mooncake storage backend
```



两种模式共享相同的 Artifact Connector 上层接口、block identity、profile、

full\-block/tail 组织和正确性约束。区别仅是 backend 的存储、可见性和生命周期。



最终决定：



- 简单模式也是 Artifact Connector，不是绕过 connector 的临时 capture 路径。

- 简单模式基于 SHM，但 HTTP 保持兼容并返回实际值；用户不需要拼接 block，

VIME 也不需要实现 SHM handle reader。

- 生产模式中 Artifact Connector 只执行一次 publish：

`ArtifactConnector -> Mooncake`。

## Roadmap Checklist



* [ ] **阶段 0：Finalize现有pr \& 迁移MRV1的到MRV2**

    * [ ] R3

        * [ ] 升级到MRV2 [https://github\.com/vllm\-project/vllm/pull/50721](https://github.com/vllm-project/vllm/pull/50721)

        * [ ]  [vllm\-project/vllm\#45635](https://github.com/vllm-project/vllm/pull/45635)  — Owner：；@申奥

            * [ ] 测试矩阵

                ### 正确性

                - SHM：基本 R3 返回正确。

                - Prefix cache：KV 命中时正确复用 R3。

                - MTP：rejected tokens 不进入 R3。

                - Reset/LRU：旧或被淘汰的 R3 不会被错误读取。

                - vllm开启Simple Offload：KV block 在 GPU/CPU 间迁移后，R3 仍按 KV hash 正确读取。

                ### 性能

                - 基线代码，Artifact 关闭。

                - Candidate，Artifact 关闭——验证改动本身零开销。

                - Candidate，Artifact 开启——测量 SHM 或 Mooncake 的实际开销。

        

    * [ ] DSA @刘荣@申奥

        * [ ] https://github\.com/vllm\-project/vllm/pull/47279

        * [ ] 迁移到shm

        * [ ] 迁移到MRV2

    * [ ] Top p/k token ids @招行@申奥

        * [ ] https://github\.com/vllm\-project/vllm/pull/49577

            * [ ] Waiting for review



* [ ] 阶段 1：建立Ar Connector，并且把shm作为Artifact Connector的第一@刘荣

    * [ ] Token\-wise

        * [ ] R3  [https://github\.com/aoshen02/vllm/pull/11](https://github.com/aoshen02/vllm/pull/11)@申奥

        * [ ] Logprobs @刘荣

        * [ ] DSA @刘荣

        * [ ] Top p/k token ids\. @招行

    * [ ] Request wise

        * [ ] 多模态数据



* [ ] S 2：Mooncake 生产 backend @刘荣

    * [ ] **Mooncake Artifact Store**

        * [ ] 唯一生产写入路径为

            `ArtifactConnector -> Mooncake`，不重复 put。

        * [ ] 保持与 SHM 相同的 identity、finalize/discard、manifest 和

            materialize contract。

    * [ ] **迁移三个 Token\-wise fields**

        * [ ] R3。https://github\.com/aoshen02/vllm/pull/7 

        * [ ] Logprobs。

        * [ ] DSA。

        * [ ] Top p/k token ids\.

    * [ ] **迁移两个 Request\-wise fields** 

        * [ ] 多模态数据。

            

* [ ] S 3：Integrate into veRL to do e2e test @刘荣

    * [ ] 基于kv put get接口

        * [ ] 在put的时候构建sample/request id（key）\-\> artifact key list \(value\) \-\> artifact的映射关系

        * [ ] 在get的时候基于sample id 去获取value

    * [ ] 基于dataloader接口, 单点瓶颈。relax

        * [ ] 可能需要改tq

        

* [ ] **在不同并行策略下扩展 writer topology **

    * [ ] **实现 multi\-rank writer** — Owner：\_\_\_\_\_\_\_\_

        * [ ] 在 single\-writer 正确性完成后，再按 token/layer logical ranges

            并行写入。

        * [ ] 验证 single\-writer 与 multi\-writer 结果完全一致。

    * [ ] **逐项解除并行/功能 guard** — Owner：\_\_\_\_\_\_\_\_

        * [ ] PP。

        * [ ] DCP/PCP。

        * [ ] DP/EP。

        * [ ] DBO/microbatching、EC transfer 和其它 KV connectors。

        * [ ] 每解除一项，先完成 global ordering、writer placement、

            finalize/discard 和故障 E2E；未验证组合继续启动拒绝。

            

* [ ] **跟踪mooncake/tq相关改动**

    * [ ] Mooncake

        * [ ] 大部分key put到本地就行，不走rdma只走cudamemcpy, DMA Copy的put https://github\.com/kvcache\-ai/Mooncake/pull/1946 （减少网卡的流量竞争）

    





Possible Optimizations:

先保持 `clone()` 是对的。除它之外，R3 路径上比较明显的优化点，按优先级看：

1. 每 step 清空整个 capture buffer

当前 `clear_buffer()` 对 `[max_num_batched_tokens, num_layers, top_k]` 整块 `zero_()`，即使本 step 只有少量 token，也清空最大容量。

可优化为：

- 只清空本 step 会使用的 token rows；

- 如果所有 MoE layer 都会完整覆盖活跃 rows，则只处理不会被覆盖的 layer；

- 预先维护 MoE layer 映射，不存没有 router 的层。

这是非常明确的无效显存写流量。

2. 每层 router IDs 复制到 capture buffer

每个 MoE layer 都执行：

```Plain Text
capture_buffer[:, layer_id, :] = topk_ids
```

这是额外的 GPU D2D 写入。长期可以让 router/kernel 直接把结果写到 capture 目标位置，或者直接保留 router 已产生的 top\-k tensor，减少一次逐层复制。但它涉及 kernel、CUDA Graph 和 tensor 生命周期，侵入性较大。

3. 所有 rank 都分配并写 capture buffer

最终只有 writer rank 做 D2H 和写 SHM，但当前 TP rank 都创建 capturer 并 capture。

可以区分：

- 普通 TP：router IDs 一致，只让 writer rank 保存。

- sequence parallel：所有 rank 仍需参与 gather，但只有 writer rank 保存最终结果。

这能减少非 writer rank 的显存占用和 capture 写流量。

4. `int32` 数据宽度

capture buffer 固定使用 `int32`，但 expert 数量通常可以用 `uint8` 或 `uint16` 表示。

如果按 `num_experts` 选择 dtype，可以同时减少：

- capture buffer 显存；

- D2D snapshot 流量；

- D2H 流量；

- SHM 大小和写入流量。

需要确认 router 输出转换成本低于节省的传输成本。

5. 每 step 新建 CPU tensor

`to("cpu", non_blocking=True)` 会为每个 output 创建 CPU tensor。可以使用预分配的 pinned CPU double/ring buffer：

```Plain Text
GPU snapshot A → pinned CPU A
GPU snapshot B → pinned CPU B
```

减少 CPU allocation 和 pinned\-memory 管理开销。

6. slot mapping 也在 clone 和 D2H

R3 payload 之外，当前还要复制：

```Plain Text
slot_mapping.clone()
slot_mapping.to("cpu")
```

实际上 scheduler 本来就知道 request 的 KV block IDs 和 token offset。可以研究让 scheduler 自己重建 R3 slots，从 worker 输出中移除 slot mapping 的 GPU snapshot 与 D2H。

这是很值得做的控制面简化。

7. SHM 的随机 scatter 写

当前类似：

```Plain Text
shm_array[slot_mapping] = routing_data
```

NumPy fancy indexing 会进行 scatter。可以把连续 slots 合并成连续区间，或者按 KV block 批量写入，减少索引和随机内存访问开销。

建议后续优化顺序是：

```Plain Text
只清理有效 rows
→ 压缩 dtype
→ 去掉 slot-mapping D2H
→ pinned CPU ring
→ 非 writer rank 减负
→ kernel/direct capture
```

PR \#3 这次只做 MRV2 正确集成并保留 clone，上述性能优化单独拆 PR、分别测量。

