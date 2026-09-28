# Prefix Artifact 功能与并行兼容性

# 



日期：2026\-07\-25



本文集中维护 R3/Artifact Connector 对 runner、请求功能、并行方式、KV connector

和模型 layout 的兼容性结论。Writer 如何分片、聚合和发布见

\[Multi\-rank Writer 拓扑\]\(\./prefix\-artifact\-writer\-topology\.md\)。



相关文档：



- \[V3 总体设计\]\(\./prefix\-execution\-artifact\-store\-design\-v3\.md\)

- \[实施 Roadmap\]\(\./prefix\-execution\-artifact\-store\-roadmap\.md\)

    

## 1\. 当前基线



```Plain Text
xhx/r3_offload @ 8c01cea0d5
= PR #45635 + 已合入该分支的 PR3 fixes/guards
```



当前 review worktree：



```Plain Text
review/07-mrv2-r3
```



MRV1/MRV2 共用 `RoutedExpertsCaptureState`。当前生产声明仍以

`#45635 + PR3` 的 MRV1 single\-writer 路径为基线；MRV2 改动仍需最终 E2E。



## 2\. 全局正确性条件



任何被标记为支持的组合都必须满足：



1. terminal R3 形状为

`[accepted_tokens, all_moe_layers, experts_per_token]`；

2. rejected speculative tokens 不进入 committed artifact；

3. KV 与 mandatory artifacts 共同 ready 后才能承认 prefix hit；

4. 请求 prompt logprobs 时，原子单元是 `KV + R3 + prompt-logprobs`；

5. artifact key 使用 logical KV hash、model、profile 和 policy epoch；

6. parallel shard 在 publish 前恢复 global logical order；

7. request finalize/discard 与 prefix block lifecycle 分离；

8. mandatory artifact 缺失、损坏或 profile mismatch 必须 fail closed。

    

## 3\. 状态定义



- **已验证**：已有定向测试或真实 GPU E2E。

- **实现中/待 E2E**：路径存在，但不能宣称生产支持。

- **阶段性拒绝**：当前启动 guard fail closed。

- **已知灰区**：代码未拒绝，但正确性尚未证明。

- **后续可实现**：需要明确的新聚合、transport 或 lifecycle。

    

## 4\. Runner 与 artifact 类型



|范围|Artifact|MRV1|MRV2|
|---|---|---|---|
|Token\-wise、可跨 request 复用|R3|维护现有 baseline|支持目标，待最终 E2E|
|Token\-wise、可跨 request 复用|Logprobs|不新增|仅在 MRV2 实现|
|Token\-wise、可跨 request 复用|DSA|不新增|仅在 MRV2 实现|
|Request\-wise、不跨 request 复用|多模态数据|不新增|仅在 MRV2 实现|
|Request\-wise、不跨 request 复用|Top\-p token IDs|不新增|仅在 MRV2 实现|



MRV1 只做 R3 regression；所有新增 artifact fields 只实现 MRV2。



## 5\. 请求功能



|Feature|设计结论|当前代码/验证|发布要求|
|---|---|---|---|
|Prefix cache|支持|R3 基线路径存在；joint lookup 属 connector PR|miss/full/partial hit equality|
|Prompt logprobs|支持|原型做过 prefix\-hit 数值验证|与 KV/R3 原子 readiness|
|Speculative|设计兼容，不 blanket 拒绝|overwrite 有定向设计/测试|各 method accepted/rejected E2E|
|Async scheduling|支持|snapshot 修复在基线|slot reuse/preemption 回归|
|DBO|原理兼容|无 R3 专用 guard；完整 E2E 不足|batch/range/finalize 隔离|
|Frontend stop|支持|有组合验证|response 等 internal finalize|
|Abort/cancel|支持|artifact 组合测试待补|discard tail，不发布部分 sample|



### 5\.1 Prefix cache 与 prompt logprobs



当请求需要 prompt logprobs 时，命中条件是：



```Plain Text
KV ready
and R3 ready
and prompt-logprobs ready
```



不存在“KV prefix cache 命中，但 mandatory prompt\-logprobs 不命中后继续运行”的合法

降级路径；这种状态必须视为 connector miss 或一致性错误。



### 5\.2 Speculative



正确路径：



```Plain Text
draft/verification capture
  -> scheduler decides accepted range
  -> commit accepted logical positions only
  -> recapture overwrites rejected positions
```



因此不应全局禁用 speculative。但 ngram、draft model、EAGLE、MTP 等方法必须逐项

验证，不能从一个 overwrite 单测推导全部支持。



### 5\.3 DBO



DBO 的风险是 mutable state 和 buffer ownership：



- overlapping batches 必须有独立 snapshot；

- logical range 和 request identity 不能串；

- ring slot 在所属 batch 完成前不能复用；

- finalize/ack 只能完成所属 request。

    

共享 capture state 是基础，不是完整 E2E 证明。



## 6\. 并行拓扑



|Feature|设计结论|当前策略|支持前还需完成|
|---|---|---|---|
|TP|支持|允许，TP2 已验证|TP16 scale 和跨节点 E2E|
|跨节点 TP|支持|collective 后 rank0 本地写|writer/EngineCore 共址及部署验证|
|DP|支持独立 replica|writer role 有灰区|每个 DP group 的 writer lifecycle E2E|
|固定 EP|原理兼容|topology E2E 不足|capture global top\-k 后验证|
|Elastic EP|服从 MRV2|MRV2 自身可能拒绝|stable membership/epoch|
|DCP|高度可实现|config/scheduler 拒绝|logical ordering、anchor/hash E2E|
|PCP|可实现|拒绝|token reorder/gather 或 owner routing|
|PP|可实现|拒绝|layer aggregation 与 writer group|
|PP\+PCP|可实现|拒绝|两级 restore 和固定 collective 顺序|



Guard 表示当前实现尚未保证正确，不表示设计上永久不兼容。Multi\-writer 的具体

rank ownership 见 \[Writer 拓扑\]\(\./prefix\-artifact\-writer\-topology\.md\)。



### 6\.1 TP



TP 不按 MoE layer 拆 routing。Sequence\-parallel shard 可以通过已有 TP collective

恢复当前 DP rank 的完整 token rows。



当前 single\-writer 路径：



```Plain Text
TP capture/all-gather -> rank0 writer
```



TP16 不需要新协议，只需要 scale 和跨节点验证。SHM 只要求实际 writer 与读取它的

EngineCore 共享 namespace；其他 TP rank 不需要写 SHM。



### 6\.2 DCP



DCP 主要改变 attention/KV physical slot。支持前必须证明：



- canonical/global logical token order；

- interleaved slot 到 KV hash block 的映射；

- selected anchor group 的 block size；

- cache miss/hit equality。

    

Artifact identity 必须基于 logical block，不能基于 DCP physical slot。



### 6\.3 PCP



PCP 初始数据是：



```Plain Text
[local_tokens, all_layers, topk]
```



支持前必须 restore global token index。Single\-writer 可 gather/reorder 到 canonical

rank；multi\-writer 可 all\-to\-all 到 block owner。



### 6\.4 PP



PP 初始数据是：



```Plain Text
[all_tokens, local_layers, topk]
```



首版应通过已有 PP transport 把 stage fragments 聚合到 last stage，再由 writer

写出完整 artifact。不同 stage 不能在没有 fragment schema 和 manifest barrier

时并发覆盖同一个 object。



### 6\.5 DP/EP



DP replica 的请求生命周期相互独立。每个 DP group 必须有自己的 writer 或 writer

group；当前固定 `output_rank == 0` 是否覆盖所有 in\-process DP 部署，需要验证。



EP 的 router top\-k 是 logical routing。Artifact 不保存 physical expert placement。

Elastic EP 的 membership 变化需要 epoch 或 drain。



## 7\. KV connector 与 offload



|Feature|当前结论|当前 guard|支持所需工作|
|---|---|---|---|
|`OffloadingConnector + CPUOffloadingSpec + kv_both`|必须兼容|唯一允许组合|MRV2\+artifact eviction/reload E2E|
|SimpleCPUOffload|未设计|拒绝|lifecycle adapter|
|TieringOffloadingSpec|可实现|拒绝|artifact tier/metadata 绑定|
|LMCache/NIXL/Mooncake KV connector|可实现|拒绝|joint readiness \+ artifact transport|
|同节点 PD|可实现|拒绝|shared namespace/joint lookup|
|跨节点 PD \+ local SHM|不可实现|拒绝|随 KV 传 artifact 或 TQ/Mooncake|



`#45635` 已把 R3 store/load 与 `OffloadingConnectorMetadata` job 对齐：



```Plain Text
GPU KV block <-> CPU KV block
GPU R3 block <-> CPU R3 block
```



Artifact Connector 不能 blanket 拒绝所有 KV connector，从而回退这项能力。实现必须

二选一：



1. 保留 `#45635` CPU R3 sidecar；或

2. 将 Artifact SHM retention 与 CPU KV block lifecycle 绑定。

    

无论哪一种，都不能无条件长期保存两份 R3。



纯 local SHM 不能跨节点传 payload。跨节点 PD 必须让 artifact 随 KV transport

传输，或者切换到 TQ/Mooncake 等跨节点 backend。



## 8\. 模型与 KV layout



|Feature|当前策略|结论|
|---|---|---|
|`FusedMoERouter`|capture hook|支持|
|Monolithic MoE kernel|启动拒绝|kernel 暴露可靠 hook 前不支持|
|其他 router|启动拒绝|需要 adapter|
|一个 full\-attention anchor|允许|支持|
|多 full\-attention groups|warning 后选第一个|**已知发布阻塞灰区**|
|无 full\-attention group|启动拒绝|pure sliding\-window/Mamba 需新 anchor|
|Hybrid \+ 唯一 full\-attention anchor|允许|需要 hash/offload E2E|



多 full\-attention group 发布前必须二选一：



1. 证明第一个 anchor 覆盖完整 logical prefix/R3 lifecycle；

2. `len(full_attn_group_ids) > 1` 时启动失败。

    

## 9\. 当前 guard



在 `--enable-return-routed-experts` 下：



- `PP > 1`：`ValueError`；

- `DCP > 1` 或 `PCP > 1`：`ValueError`，scheduler 还有 assertion；

- 无 full\-attention group：`ValueError`；

- monolithic/未知 router：`ValueError`；

- 非精确 CPU OffloadingConnector 组合：`ValueError`；

- PD 和其他 KV connector：`ValueError`；

- TP：允许；

- speculative：没有 blanket guard；

- DBO：没有 R3 专用 guard；

- 多 full\-attention groups：只有 warning，是必须关闭的灰区。

    

## 10\. 验证矩阵



发布前至少验证：



- MRV1/MRV2 TP1/TP2；

- prefix miss/full/partial hit；

- sync/async/preemption；

- speculative 各 method 的 accepted/rejected；

- DBO overlap；

- frontend stop/abort；

- CPU offload eviction/reload；

- prompt\-logprobs cache\-off/miss/hit equality；

- mandatory fields 缺失、损坏和 profile mismatch；

- unsupported topology/router/config fail closed。

    

新增并行方式必须额外验证 global logical order、single/multi\-writer 等价性，以及

collective failure 不会发布 partial artifact。

