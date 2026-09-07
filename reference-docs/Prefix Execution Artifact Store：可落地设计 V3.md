# Prefix Execution Artifact Store：可落地设计 V3

日期：2026\-07\-26



状态：按当前最终结论重写。当前阶段只实现 R3，以及 SHM/direct Mooncake 两个

Artifact Connector backend。SHM 重写、direct Mooncake fake\-client 验证和

H200 TP1/TCP 真实 master E2E 已完成；TP2、RDMA 和异常路径仍是显式验收门槛。



## 0\. 文档导航



本文维护总体架构、public contract、数据模型、模块边界和核心协议。专题内容：



- GPU/CPU ring、PCIe/RDMA 流量和推理冲突见

\[数据面与性能分析\]\(\./prefix\-artifact\-data\-plane\-performance\.md\)。

- 多 rank writer 的 ownership、聚合和 ready 协议见

\[Writer 拓扑\]\(\./prefix\-artifact\-writer\-topology\.md\)。

- 并行方式、功能支持状态和启动 guard 见

\[功能与并行兼容性\]\(\./prefix\-artifact\-compatibility\.md\)。

- Mooncake 的源码接口事实和复用边界见

\[存储接口调研\]\(\./mooncake\-interface\-survey\.md\)。

- 真实 master、独立 consumer、SHM parity 和 logprobs 数值结果见

\[Direct Mooncake R3 \+ logprobs E2E\]\(\./mooncake\-r3\-logprobs\-e2e\.md\)。

- 实施和 PR 顺序见

\[Roadmap\]\(\./prefix\-execution\-artifact\-store\-roadmap\.md\)。



## 1\. 当前范围



### 1\.1 基线



```Plain Text
#45635 + fork PR3 fixes
  8c01cea0d5
    ↓
MRV2 + R3
  7f94ae45b6
    ↓
unified ArtifactRequestCore + SHM
  7e4da344ed
    ↓
direct Mooncake
  556fe92125
```



当前 stacked PR：



- [aoshen02/vllm\#4](https://github.com/aoshen02/vllm/pull/4)：ordered\-key

Artifact Core \+ SHM；

- [aoshen02/vllm\#7](https://github.com/aoshen02/vllm/pull/7)：direct

Mooncake。



当前 PR4/PR7 已重写为精简 stack。PR7 保留真实 backend E2E 发现的关键修复：

Mooncake 0\.3\.10 的 registered\-buffer get 必须传入对象精确长度；store 内部按 key

调用 `get_size(key)`，再按实际长度打包 batch，public caller 仍只传 key。



关键源码锚点：



- [vllm\-project/vllm\#45635](https://github.com/vllm-project/vllm/pull/45635)：

physical\-slot R3 capture 与 CPU offload baseline；

- [vllm\-project/vllm\#39568](https://github.com/vllm-project/vllm/pull/39568)：

scheduler accepted\-range/speculative overwrite 语义参考；

- vLLM

`vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py`：

Mooncake client、registered buffer、batch I/O 和 lifecycle 参考。



### 1\.2 本阶段交付



只交付：



- Model Runner V2 的 R3 capture；

- R3 full\-block prefix reuse；

- SHM simple backend；

- direct Mooncake production backend；

- KV \+ R3 joint prefix readiness；

- ordered artifact\-key consumer contract。

    

不交付：



- logprobs、DSA、多模态、Top\-p token IDs；

- VIME dataloader；

- multi\-writer、PP、DCP/PCP 等后续 topology。

    

### 1\.3 部署前提与边界



- 同一 deployment 的模型、tokenizer、并行配置、artifact block size 和 field

profile 一致；不尝试让不兼容配置共享 key namespace。

- 一个 engine 不混跑普通 serving request 和 artifact rollout request。启用

Artifact Connector 后，所有可进入 prefix cache 的 request 都必须生产本部署要求的

mandatory fields。

- 一个 engine 一次只启用一个 backend：`shm` 或 `mooncake`，不双写。

- 当前版本不实现 policy epoch/fence 状态机。启用 R3 Artifact Connector 时，

in\-place weight update fail closed；切换权重应 drain 并重启 engine。若未来要支持

同一 engine 热更新，再单独设计 generation/epoch contract。

- Artifact Connector 关闭时，不创建 capture state、SHM store、Mooncake client 或

staging buffer，也不在普通 request/step 上构造 artifact metadata。



以下仍是 request/control\-plane metadata，不写入 ArtifactStore：



- reward、tool result、finish reason、group metadata；

- loss mask、consumer lease/ack；

- scheduler 的 request status、commit/finalize 命令和 worker ACK。

    

Value model、reward model 等其他 producer 将来可以复用 object/store 协议，但必须

使用自己的 producer/model namespace 和 retention policy，不能绑定到 policy KV 的

失效规则。



### 1\.4 后续字段模型仍保留



本阶段只实现 R3，但后续扩展不重新发明 lifecycle：



- Artifact Connector 只接入 MRV2；MRV1 保留上游 R3 行为，不接入本 store

lifecycle。

- `PREFIX_BLOCK`：R3、DSA 和一个统一的 `logprobs` field。开启

`prompt_logprobs` 只是给该 field 增加 prompt coverage，不创建第二种

prompt\-logprobs artifact；启用 artifact backend 后 logprobs 也从 backend 读取，

HTTP 不再重复返回一份完整 logprobs value。

- `REQUEST_ONLY`：多模态数据和 Top\-p token IDs，不参与跨 request prefix reuse。

- 所有字段共用 `ArtifactRequestCore`、backend object contract 和 finalize；

field adapter 只描述坐标、shape 和 codec。

- 多个 mandatory fields 使用独立 per\-field objects/key lists，但共享一个 completion

barrier：任一 required field 失败，request 不返回任何 production handle。



DSA 的上游实现跟踪：

[vllm\-project/vllm\#47279](https://github.com/vllm-project/vllm/pull/47279)。



### 1\.5 已有基线证据



这些结果证明 R3 capture/Core 的历史基线，不等于 direct Mooncake 已通过验收：



- H200 TP8：旧 \#45635 与修复后 HTTP R3 cold exact equality，shape

`[1016, 32, 8]`、`uint8`，40 个 expert 均出现；

- local prefix cache：992\-token R3 prefix exact；

- CPU KV offload：992\-token external R3 prefix exact；

- R3 \+ prompt\-logprobs 原型：shape `[372, 32, 8]`，352\-token local/external

prefix hit，cold/hot logprobs 与 cached R3 exact；

- 最新 stacked head：H200 TP1/TP2 MRV2 async cold/full\-hit，64\-token cached

R3 prefix exact。



完整历史命令和环境记录见 \[PR3 validation\]\(\./final\_validation\.md\)。



## 2\. 总体架构



### 2\.1 vLLM 进程边界



```Plain Text
┌──────────────────────────────────────────────────────────────┐
│ API Server Process                                           │
│                                                              │
│ Completion / Chat / AsyncLLM / OutputProcessor               │
│                                                              │
│ SHM:      routed_experts = actual R3                         │
│ Mooncake: artifact_keys = ordered list[str]                  │
└─────────────────────────────┬────────────────────────────────┘
                              │ ZMQ / IPC
                              │ ↓ request / ↑ terminal result
┌─────────────────────────────▼────────────────────────────────┐
│ EngineCore Process                                           │
│                                                              │
│ Scheduler / KVCacheManager / ArtifactSchedulerConnector      │
│ - executed boundary                                           │
│ - KV-compatible block hashes                                  │
│ - KV + R3 joint readiness                                     │
│ - commit/finalize ordering                                    │
└─────────────────────────────┬────────────────────────────────┘
                              │ executor RPC
                              │ ↓ ArtifactConnectorMetadata
                              │ ↑ commit ACK / result
┌─────────────────────────────▼────────────────────────────────┐
│ GPU Worker Process(es)                                       │
│                                                              │
│ MoE router capture                                           │
│   -> current-step async D2H                                  │
│   -> split by request + accepted/rejected boundary           │
│   -> worker-owned stable logical suffix buffer               │
│   -> ArtifactRequestCore                                     │
│   -> ArtifactStore                                           │
└───────────────┬──────────────────────────────┬───────────────┘
                │                              │
                ▼                              ▼
┌──────────────────────────┐     ┌─────────────────────────────┐
│ Logical SHM Block Store  │     │ MooncakeDistributedStore   │
│ simple/local             │     │ production/distributed      │
└──────────────────────────┘     └─────────────────────────────┘
```



### 2\.2 Artifact Connector 内部



```Plain Text
ArtifactSchedulerConnector
  ├─ accepted/executed token progress
  ├─ KV-compatible hashes
  ├─ backend reader existence lookup
  ├─ full-block commit plan
  └─ finalize + ACK validation
                  │
                  ▼
ArtifactWorkerConnector
  ├─ receives current-step accepted logical rows
  ├─ owns uncommitted per-request logical suffixes
  └─ delegates lifecycle to ArtifactRequestCore
                  │
                  ▼
ArtifactRequestCore
  ├─ field profile
  ├─ content-addressed full-block keys
  ├─ stateless commit/finalize operations
  ├─ request-scoped partial tail
  ├─ backend-independent materialization
  └─ object envelope validation
                  │
          ┌───────┴────────┐
          ▼                ▼
  ShmArtifactStore  MooncakeArtifactStore
```



`connector.py` 只编排 scheduler/worker metadata。key、coverage、ordered refs、

tail 和 materialization 只在 `request_core.py`。



### 2\.3 端到端 consumer 路径



```Plain Text
rollout request
      │
      ▼
vLLM Scheduler -> GPU Worker capture -> stable logical buffer -> ArtifactRequestCore
                                      │
                       ┌──────────────┴──────────────┐
                       ▼                             ▼
                SHM backend                  Mooncake backend
                       │                             │
              materialize actual R3          put immutable objects
                       │                             │
                       └──── terminal HTTP ──────────┘
                            actual R3 | ordered keys
                                              │
                                              ▼
                                  consumer materializer
                                              │
                                              ▼
                                      rollout/trainer
```



Consumer 不向 vLLM 回写 dataset/sample 状态。SHM 路径直接消费 actual R3；Mooncake

路径用 HTTP keys 读取 objects。两条路径的最终 tensor 必须逐元素一致。



## 3\. Capture 与 ArtifactStore 的区别



Artifact 模式复用 `#45635` 的 router capture hook 和 stable GPU snapshot，但不复用

其 physical\-slot mmap：



```Plain Text
GPU router IDs
  -> authoritative rank-0 worker
  -> current-step stable GPU snapshot
  -> async D2H
  -> per-request logical suffix buffer
  -> ArtifactRequestCore
```

\[block1 key, block2 key, \] request1 

\[block4 key, block3 key, \] request2 



两层含义不同：



|层|Identity|生命周期|用途|
|---|---|---|---|
|worker logical buffer|request \+ logical token range|commit 后释放 full block，terminal/abort 丢弃 tail|accepted\-row staging|
|ArtifactStore|logical KV hash \+ field profile|store retention|prefix reuse/consumer|



因此：



- Artifact Connector 没有第二次 GPU capture；

- physical slot 不能成为 artifact key；

- SHM 和 Mooncake 都发布 logical objects；

- Artifact write task 只允许一个 destination，不再同时写 physical\-slot mmap；

- SHM simple backend 因自身使用 `/dev/shm`，要求 writer 与 EngineCore 同机；

- Mooncake production backend 不依赖 EngineCore `/dev/shm`，writer/EngineCore

跨节点只剩部署验证，不再有 shared\-mmap 架构约束。



## 4\. Public contract



### 4\.1 SHM backend



SHM 面向简单、立即返回的使用方式：



```JSON
{
  "routed_experts": "<base64 npy>",
  "artifact_keys": null
}
```



- caller 不读取 `/dev/shm`；

- caller 不接收 path、mmap handle 或内部 object key；

- worker 通过统一 Core materialize actual R3；

- actual R3 继续经过 EngineCore/API IPC。

    

### 4\.2 Mooncake backend



Mooncake 面向生产 consumer：



```JSON
{
  "routed_experts": null,
  "artifact_keys": [
    "vllm-artifact/r3/block/...",
    "vllm-artifact/r3/block/...",
    "vllm-artifact/r3/tail/..."
  ]
}
```



`artifact_keys`：



- 是 ordered `list[str]`；

- 每个 key 对应一段连续 R3 logical tokens；

- full\-block key 可跨 request 复用；

- 最后一个不满块使用 request\-scoped tail key；

- 只在所有 object put 完成后返回；

- 不是 Mooncake segment/replica handle；

- 不包含 `artifact_sample_id`。

    

### 4\.3 Key list 长度



令：



```Plain Text
L = 实际执行 forward、应返回 R3 的 token 数
B = artifact hash block size
```



则：



```Plain Text
num_full_blocks = floor(L / B)
has_tail = (L % B) != 0

len(artifact_keys)
  = num_full_blocks + int(has_tail)
  = ceil(L / B)
```



因此它等于覆盖本 request R3 的 logical block 数，并且不超过 request KV block

table 的 block 数上界。



最后一个仅 sampled、尚未执行 forward 的 token 不计入 `L`。



### 4\.4 API delivery



- non\-streaming：每个 terminal choice 返回自己的 actual R3 或 ordered keys；

- streaming：只在 terminal chunk 返回一次，intermediate chunks 不带半成品；

- batch/`n > 1`：每个 child request 独立计算 `L`、tail 和 key list；可共享相同

full\-block keys；

- backend 失败时不返回部分 key list，也不把原本的 terminal success 伪装成可消费的

artifact success。



## 5\. R3 数据模型



### 5\.1 Logical coordinate



R3 使用：



```Plain Text
reuse_policy      = PREFIX_BLOCK
logical_coordinate = EXECUTED_TOKEN
shape             = [executed_tokens, num_moe_layers, topk]
```



### 5\.2 Forward 对齐



一次 `forward(token_i)` 的字段对齐必须显式固定：



```Plain Text
KV / R3 / DSA of token_i  <- forward(token_i)
logprob(token_i)          <- forward(token_{i-1})
```



因此：



- R3/DSA 使用 `EXECUTED_TOKEN` 坐标；

- logprobs 使用 `PREDICTED_TOKEN` 坐标；

- prompt 的第一个 token 没有前驱 row；

- 最后一个仅 sampled、尚未作为输入执行 forward 的 token 没有 R3/DSA；

- speculative rejected rows、stop\-trim 后 rows 和 async in\-flight rows 都不能进入

committed range；

- R3 保存 global expert IDs，DSA 保存 logical top\-k positions；logprobs profile

必须区分 raw/processed mode。



### 5\.3 Full\-block key



```Plain Text
block_key = H(
  schema_version,
  model_namespace,
  field="routed_experts",
  dtype,
  shape_per_token,
  artifact_block_size,
  kv_block_hash,
)
```



它不包含 request ID。相同 model/profile/prefix 得到同一个 key。当前权重切换通过

drain/restart 形成新的部署 namespace，不在 scheduler 内维护 policy epoch。



### 5\.4 Partial tail key



```Plain Text
tail_key = H(
  schema_version,
  model_namespace,
  field_profile_id,
  request_id,
  request_attempt_id,
  terminal_executed_boundary,
)
```



Tail：



- 只属于当前 request；

- 不进入 prefix lookup；

- abort 时不返回；

- 由 backend 自然淘汰。

    

### 5\.5 Object envelope



Mooncake/SHM object 使用同一逻辑 header：



```Plain Text
uint32 header_length
canonical JSON header
raw contiguous tensor bytes
```



Header 至少包含：



```JSON
{
  "schema_version": 1,
  "kind": "block | tail",
  "field": "routed_experts",
  "object_id": "...",
  "dtype": "|u1",
  "shape": [16, 32, 8],
  "valid_len": 16,
  "payload_sha256": "...",
  "header_sha256": "..."
}
```



Object 自描述 dtype、shape、valid length 和 checksum，因此不需要 request

manifest。



Mooncake get 需要 caller 提供 registered receive buffer。Mooncake 0\.3\.10 的

`batch_get_into_multi_buffers` 不接受“最大 capacity 代替实际对象长度”：真实 E2E

会返回 `TRANSFER_FAIL (-800)`。`MooncakeArtifactStore.get(keys)` 必须在内部调用

`get_size(key)`，校验长度不超过 field profile 上限，再按精确长度把多个对象紧凑

排入 registered staging buffer。Public HTTP 和

`materialize_routed_experts(store, keys)` 仍只传 ordered keys；full block 和 tail

都不需要 request manifest 或 public length list。



## 6\. Request Core



`ArtifactRequestCore` 只持有不可变依赖：



```Plain Text
store
writer
namespace / field profile
materialize mode
```



负责：



- 将 scheduler 已确认的 executed range 编码为 full blocks；

- key/profile；

- finalize 时从全部 KV hashes 重新派生 ordered key list；

- 校验 full blocks 均已存在并发布 request\-scoped tail；

- backend put 和 simple\-mode materialization。

    

不负责：



- GPU capture hook；

- scheduler admission；

- Mooncake master lifecycle；

- HTTP serialization；

- 外部 dataset/sample 状态机。

    

不再需要：



- public `artifact_sample_id`；

- request manifest；

- public `artifact_keys` 之外的 handle；

- final\-notify control row；

- worker 侧 request assembly、segment tracking、discard RPC；

- policy epoch/update fence 状态机。

    

## 7\. Publication protocol



### 7\.1 Full blocks



每次 accepted/executed boundary 新完成完整 block：



```Plain Text
1. scheduler 只为新完成的 full range 下发 commit
2. worker 从当前 step 的稳定逻辑 buffer 读取并生成 content-addressed objects
3. backend 对已存在的 immutable key 跳过 put
4. 缺失 objects 执行 batch put
5. SHM 同步 publish 后 ACK；Mooncake 在不可变 bytes 进入有界 publisher 后 ACK
6. prefix admission 仍以 backend existence lookup 为准，不把 Mooncake enqueue ACK
   当作 ready
```



只发布新完成的 full blocks。同一步多个 request 的新 blocks 合并成 batch，但每个

object 的结果独立记账：一个 item 失败不能把同 batch 的其他 item 误标 ready，也不能

返回引用失败 item 的 request key list。



### 7\.2 Finalize



```Plain Text
1. scheduler 根据 stop/abort/spec acceptance 得到 authoritative executed end
2. 通过正常 SchedulerOutput metadata 下发 finalize
3. full-block key 从 KV hashes 确定性派生
4. 若存在 partial tail，SHM 同步 put；Mooncake 将稳定 tail bytes enqueue
5. worker 返回 request-id-bound finalize ACK
6. scheduler 校验 ACK 后生成 ordered list[artifact_key]
7. SHM：materialize actual R3
8. Mooncake：返回 key list
9. 释放被暂存的 terminal HTTP output
```



Mooncake key list 本身就是完整 handle。HTTP 在全部对象都已 PutStart/enqueue 后

即可返回，不等待 PutEnd/ready；consumer 对 key 执行 `batch_is_exist` 轮询。因此不需要

request manifest 或额外 final notify。



Async scheduling 下不能直接使用包含下一帧 in\-flight tokens 的计数；speculative

路径必须使用 stop\-trim/acceptance 之后的 boundary。Finalize 可以占用下一次

zero\-token control step，但不能由 worker 自己猜 EOS/stop，因为它看不到 frontend

stop string、abort 和全部 scheduler 状态。



### 7\.3 Abort / preemption



- abort/cancel 不生成 tail 或 HTTP key list；

- scheduler 删除本地 request state，不下发 discard RPC；

- 已提交 full blocks 保留；

- full blocks 不绑定 request lifetime；

- backend retention/eviction 自然管理。

    

Preemption 保留已发布的 immutable full blocks；未发布的 accepted rows 已在 worker

当前 step D2H 后复制到 request\-scoped logical buffer。Scheduler 按标准 KV lifecycle

释放/复用 physical KV blocks，不维护 artifact 专用 snapshot 或 retention。若

recompute 覆盖同一 logical position，buffer 以新 accepted rows 覆盖；已提交前缀忽略。



### 7\.4 Put 失败



- SHM 只有同步 put 成功才返回 success；

- Mooncake enqueue/PutStart 可以返回 keys，但不能被 prefix admission 当作 ready；

- publisher 持有 immutable encoded bytes，原 KV slot 和 logical capture buffer 可立即

释放；registered staging buffer 仍由后台 put 持有到 Mooncake 调用返回；

- 仅对明确可重试的 transport failure 做有界重试和 backoff；

- PutStart 后后台失败会记录 fatal error，后续 publication fail closed；已返回 keys

将持续 not\-ready，由 consumer polling timeout/error 处理；

- 已成功发布的 immutable full blocks 不回滚，失败 tail 自然成为不可引用对象；

- crash 遗留的未引用对象由 backend retention/GC 回收。

    

## 8\. Prefix cache \+ R3



### 8\.1 Joint readiness



命中条件：



```Plain Text
KV block ready
AND corresponding R3 block key ready
```



Mooncake 模式不能只相信 scheduler 本地 ACK catalog，因为：



- Mooncake object 会自然淘汰；

- 其他 vLLM instance 可能已发布；

- engine 重启后本地 catalog 为空。

    

当前实现不维护跨请求 ready catalog。EngineCore 中的

`MooncakeArtifactReader` 直接执行 `batch_is_exist`；reader 不注册 staging buffer，

只有 rank\-0 GPU worker 的 writer store 注册 staging（默认 64 MiB）。



正确 admission：



```Plain Text
KV candidate prefix
  -> derive ordered R3 block keys
  -> Mooncake batch_is_exist(keys)
  -> longest KV-ready AND R3-ready prefix
```



这里的 KV\-only candidate 还不是“prefix hit”。只有 joint lookup 完成后，

scheduler 才能承认 hit：



- 尚未 ready 的 candidate 可以等待，或作为 joint miss 重新执行；

- 不存在“scheduler 已承认 KV hit，但合法地缺少 mandatory R3”的状态；

- scheduler 已承认 hit 后再 missing/corrupt/profile\-mismatch，必须报一致性错误，

不能静默缩短命中或补算。



### 8\.2 Hit path



承认 joint hit 后：



- cached span 不重新 forward；

- 不重新 capture cached R3；

- 不调用 block put；

- request 绑定已有 keys；

- 只对新执行完成的 blocks 增量 put。

    

```Plain Text
request A -> [K0, K1, K2, TailA]
request B -> [K0, K1, K2, TailB]
```



### 8\.3 Eviction race



- joint admission 前 missing：candidate 未形成 hit，可等待或重新执行；

- admission 后、finalize 前 missing：fail closed；

- HTTP 返回 keys 后 object 被淘汰：consumer get 返回 not found，受 deployment

retention SLA 管理。



同一 artifact\-enabled engine 不允许先由普通 request 只加热 KV，再把该 KV\-only

状态称为 artifact request 的命中。若实现观察到这种已承认的 KV/artifact

readiness 撕裂，应报错并使相关 cache entry 失效。



## 9\. SHM backend



SHM store 提供：



- `/dev/shm` trusted namespace；

- content\-addressed full blocks；

- atomic temporary\-file \+ rename publication；

- checksum；

- capacity limit；

- TTL/lease cleanup；

- ordered\-key materialization。

    

SHM public contract 不返回 keys，但内部仍复用同一 ordered refs。



## 10\. Direct Mooncake backend



### 10\.1 Deployment ownership



复用 vLLM `MooncakeStoreConnector` 的部署约定：



```Bash
mooncake_master --port 50051
export MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json
```



Artifact Connector：



- 不启动或关闭 master；

- 不修改 Mooncake deployment；

- authoritative rank\-0 GPU worker 创建 `MooncakeDistributedStore`；

- 复用 config 中的 master、protocol、segment 和 local\-buffer 配置；

- close 时只释放本进程 store/TransferEngine。

    

### 10\.2 Store operations



使用 Mooncake 已有接口：



```Plain Text
batch_is_exist
batch_put_from_multi_buffers
batch_get_into_multi_buffers
register_buffer
close
```



Artifact Core 生成的 logical object key 直接成为 Mooncake key。



### 10\.3 Registered staging buffer



MooncakeArtifactStore 维护有界 registered CPU staging pool：



```Plain Text
encode object
  -> acquire slot
  -> copy bytes
  -> batch_put_from_multi_buffers
  -> transfer complete
  -> release slot
```



读取同理。`MooncakeArtifactPublisher` 在其前面维护单写线程和 1\-batch 有界队列：



```Plain Text
encode immutable bytes
  -> bounded enqueue (PutStart / HTTP may return)
  -> background MooncakeArtifactStore.put
  -> PutEnd / object becomes ready
```



registered staging buffer 在底层调用返回前不复用；请求的 KV physical slot 不参与这段

lifecycle。多 slot ring/GDR 属于后续性能优化。



### 10\.4 Duplicate writers



第一版：



```Plain Text
batch_is_exist == 1 -> skip
batch_is_exist == 0 -> put
```



content key 相同意味着合法 payload 应完全一致。并发 cold writers 可能同时 put

相同 bytes；consumer 仍校验 checksum。



暂不引入 lock/control service。若需要严格防御不同 payload 的同\-key race，再增加

Mooncake create\-if\-absent/CAS。



### 10\.5 Retention、GC 与 staleness



- full block 的 lifetime 与单个 request 解耦；request 结束不删除 shared block；

- tail 只为返回该 request，按 backend 的自然淘汰策略流失；

- 当前不引入 consumer ACK/lease control plane；deployment 必须配置从 HTTP 返回到

consumer get 完成所需的最小 retention SLA；

- scheduler 每次 admission 直接查询 backend，不相信本地 stale catalog；

- field profile/model namespace 变化通过新 key namespace 隔离旧对象；

- 当前不支持同一 engine 内 policy 热更新；通过 drain/restart 切换 namespace；

- storage pressure 先拒绝新的 artifact admission 或使请求明确失败，不删除仍在

in\-flight transfer 中的 source。



### 10\.6 Crash 语义



- worker 在 enqueue 前 crash：不返回 keys；

- enqueue/HTTP 返回后、PutEnd 前 crash：keys 已返回但保持 not\-ready，consumer polling

超时；这是当前 PutStart contract 的明确失败语义；

- put 成功但 ACK 前 crash：对象可能成为可复用 orphan，不能据此伪造本 request

completion；

- ACK 后 EngineCore/API crash：已发布 immutable block 保留，原 HTTP request 失败；

- master/client failure 通过明确 backend error 传播，不能降级成成功但少 key。

    

## 11\. Consumer



调用方式：



```Python
r3 = materialize_routed_experts(
    store=MooncakeArtifactStore(...),
    artifact_keys=response.artifact_keys,
)
```



Materializer：



1. batch\-get ordered keys；

2. 验证 object identity、checksum、dtype、shape 和 field profile；

3. 按顺序拼接 full blocks/tail；

4. 返回完整 R3。

    

caller 不手动拼 tensor。key list 是 transport handle，不是让用户实现存储协议。

任一 required key not found、读取失败或损坏时，整条 artifact 读取失败；不返回较短

R3，也不自动从推理端补算。



## 12\. 配置



```Plain Text
backend: "shm" | "mooncake"

SHM:
  shm_dir
  max_shm_bytes
  shm_ttl_seconds

Mooncake:
  mooncake_store_id
  mooncake_staging_buffer_bytes
```



Mooncake master address、protocol、device、segment size 和 local buffer 只从

`MOONCAKE_CONFIG_PATH` 读取，ArtifactConfig 不重复配置。



## 13\. 代码布局



```Plain Text
vllm/distributed/artifact_connector/
├── __init__.py
├── connector.py
├── protocol.py
├── request_core.py
├── fields.py
├── store.py
├── shm.py
└── mooncake.py
```



- `protocol.py`：scheduler/worker IPC dataclasses；

- `request_core.py`：唯一 request state machine，以及 key、coverage、envelope、

ordered refs、tail 和 materialize；

- `fields.py`：小型 immutable field descriptors/coordinate adapters；未来 R3、DSA、

logprobs 都是同一 Core 的配置，不建立 per\-field Manager/生命周期；

- `store.py`：opaque object I/O Protocol 和 factory，不理解 R3/logprobs；

- `shm.py`、`mooncake.py`：两个 backend，只实现同一 store contract。

    

不创建独立 `prompt_logprobs.py`、backend\-specific materializer、Codec 继承树或

manager hierarchy。Prompt\-logprobs 特有的 boundary\-hidden capture/restore 留在

sampler 路径；可持久化的 logprobs 仍走统一 field adapter/Core。



## 14\. 当前兼容性边界



当前允许：



- CUDA MoE generate；

- MRV2；

- TP；

- single authoritative writer；

- SHM/direct Mooncake；

- local prefix cache；

- 明确支持的 CPU `OffloadingConnector` 组合。

    

当前 fail closed：



- PP；

- DCP/PCP；

- DBO/executor microbatching；

- elastic EP；

- unsupported router；

- 不受支持的 KV connector/PD；

- EC transfer；

- multimodal/encoder\-decoder/diffusion。

    

详细理由以

\[兼容性文档\]\(\./prefix\-artifact\-compatibility\.md\) 为准。



### 14\.1 当前明确接受的妥协



- 当前只有 rank\-0 authoritative writer。SHM simple backend 要求它与 EngineCore

共享 `/dev/shm`；Mooncake production backend 没有该共址要求，但 TP16/cross\-node

仍须实测。

- Mooncake 第一版使用单后台 writer \+ 单 pending batch；HTTP 不等 PutEnd，但高负载时

bounded queue 会回压。不做 multi\-slot ring/GDR。

- 没有 create\-if\-absent/CAS；跨 instance cold race 依赖 same\-key deterministic

same\-bytes，严格 collision 防护留到 multi\-writer 阶段。

- 生命周期采用自然淘汰 \+ deployment retention SLA，不增加 consumer ACK/lease

control plane。

- 为保证正确性暂时 guard PP、DCP/PCP、DBO 和未验证 KV connector，而不是宣称这些

topology 永久不兼容。

- SHM simple path 为了立即返回 actual R3，接受 terminal payload 经过

ModelRunnerOutput/EngineCore/API IPC；production Mooncake path 不这样做。



## 15\. 正确性验收



必须覆盖：



- `len(keys) == ceil(executed_tokens / block_size)`；

- exact\-block request 没有 tail；

- partial request 最后一个 key 是 tail；

- key 顺序与 logical token span 一致；

- non\-streaming/streaming/batch/`n > 1` 的 terminal\-only delivery；

- SHM actual R3 与 Mooncake keys materialization exact equality；

- cold/full/partial prefix hit；

- hit path 不调用 block put；

- joint admission 前 missing 不会被承认为 hit；

- admission 后 missing/corrupt fail closed；

- batch 内单项 put 失败、retry exhausted 和 worker crash 不提前 ready；

- async in\-flight、spec stop\-trim 和 rejected rows 不越过 executed boundary；

- abort 不返回 tail/key list，preemption 不读取已释放 slot；

- in\-place weight update fail\-closed 和 disabled zero\-overhead；

- retention SLA、eviction/stale local catalog；

- TP1/TP2/TP scale；

- sync/async；

- preemption/abort；

- CPU KV offload；

- Mooncake TCP/RDMA；

- master/worker lifecycle isolation。

    

## 16\. 方案修正记录



本次相对旧 V3 的修正：



- production backend 使用 direct Mooncake；

- public handle 从 `artifact_sample_id` 改为 ordered `list[artifact_key]`；

- 删除 request manifest；

- list 长度明确为 `ceil(executed R3 tokens / block_size)`；

- full blocks 使用 KV\-compatible content keys；

- partial tail 使用 request\-scoped key；

- keys 全部 ready 后才返回，因此不需要 final notify；

- Mooncake object 自描述，consumer 通过统一 materializer 拼接；

    

以下不是遗漏，而是相对 V2 的明确决策变化：



- 删除 request manifest，ordered keys \+ self\-describing objects 已足够；

- 删除 public sample ID，HTTP 直接返回 ordered keys；

- 删除独立 final\-notify，沿用 finalize metadata \+ worker ACK；

- V2 的“API 不等 PutEnd”改为 keys 全部 put 成功后才返回，否则 key list 不具备

ready 语义；

- V2 的“推理只看 KV”改为 scheduler 只承认 KV \+ mandatory artifact joint hit。

- V2 的“所有 fields 合成一个 block value”改为 per\-field immutable objects \+

request\-level completion barrier，避免不同 field 的 codec/layout 耦合。

- V2 的 selected\-hidden placeholder 不在当前确认的字段矩阵中，暂不进入 Roadmap；

若重新提出需求，按 `PREFIX_BLOCK` 或 `REQUEST_ONLY` 明确归类后再接入统一 Core。



## 17\. 实施顺序



```Plain Text
#45635 + fixes
  -> PR3 MRV2 + R3
  -> PR4 unified Core + SHM
  -> PR7 direct Mooncake + ordered keys
  -> SHM/Mooncake parity
  -> remaining topology/performance matrix
```



## 18\. 旧设计覆盖审计



|旧内容|当前落点|
|---|---|
|目标、部署前提、disabled 零开销|§1\.2–1\.4|
|forward/token/field 对齐|§5\.1–5\.5|
|vLLM 进程边界、唯一 writer、simple/production path|§2–3、Writer 拓扑文档|
|full\-block batch put、tail finalize、Put/ACK|§7|
|ring、backpressure、PCIe/RDMA 冲突|数据面与性能文档|
|manifest/public handle|§4、§6；manifest 已明确删除|
|retention、GC、staleness、crash|§7\.3–7\.4、§10\.5–10\.6|
|speculative、abort、preemption、KV offload|兼容性文档及 §7|
|Scheduler/Worker/Core/backend 映射|§2、§6、§13|
|Mooncake 源码接口和 inflight/buffer ownership|存储接口调研文档|
|多 rank、PP/CP/DP/EP|Writer 拓扑与兼容性文档|
|实验数据与验证矩阵|§1\.5、§15、Roadmap|

## 10\. 配置与 fail\-closed



当前基线在 `--enable-return-routed-experts` 下：



- 拒绝 `PP > 1`；

- 拒绝 `DCP > 1` 或 `PCP > 1`；

- 只允许

`OffloadingConnector + CPUOffloadingSpec + kv_role=kv_both`；

- 拒绝 PD 和其他 KV connector；

- 没有 full\-attention KV group 时拒绝；

- monolithic MoE kernel 和未知 router 拒绝；

- speculative decoding 不做 blanket 拒绝；

- TP 不拒绝。

发布前还必须解决一个 guard 缺口：多个 full\-attention group 当前只 warning 并选择

第一个 anchor。若没有证明该选择覆盖全部需要复用/offload 的逻辑 token，应改为

启动失败。



启用策略应区分：



- 部署为本项目测试 R3 时，Artifact Connector \+ SHM 默认可工作，不要求用户再拼

隐含实验开关；

- 普通请求没有请求 R3/prompt\-logprobs artifact 时，不初始化 capture/store 热路径

或承担 D2H/序列化开销。

## 11\. 验证与验收



### 11\.1 MRV2 \+ R3



- MRV1 regression；

- MRV2 TP1/TP2；

- sync/async output\-copy；

- slot reuse、preemption；

- unsupported router/monolithic kernel fail closed；

- speculative accepted/rejected overwrite；

- frontend stop boundary；

- HTTP value 与直接 capture 逐元素一致。

### 11\.2 Artifact Connector \+ SHM



- 首次 miss、完整 hit、部分 hit；

- full\-block incremental commit；

- request\-local tail；

- terminal assembly；

- same\-key idempotency；

- missing/corrupt/profile mismatch fail closed；

- capacity/retention/cleanup；

- worker crash 和 stale SHM；

- TP2；

- CPU KV eviction/reload。

### 11\.3 Prompt logprobs



- cache disabled baseline；

- first miss；

- full/partial prefix hit；

- 数值和 token 顺序逐项一致；

- boundary hidden\-state restoration；

- mandatory miss/corruption fail closed；

- R3 \+ prompt logprobs \+ frontend stop。

### 11\.4 生产 backend



- SHM/TQ backend contract parity；

- single logical publish；

- full\-block dedup；

- sample ready 原子可见；

- not\-found/not\-ready/ready；

- producer/consumer crash；

- restart/checkpoint；

- capacity and backpressure；

- 单机、多进程、跨节点；

- VIME dataloader batch equality。

## 12\. 可观测性



至少暴露：



- capture tokens/bytes/time；

- full\-block commit/hit/miss；

- joint\-readiness miss reason；

- tail bytes；

- terminal assembly time；

- corrupt/profile mismatch/invariant violation；

- SHM used/free/evicted/stale；

- backend queue depth/backpressure；

- TQ not\-found/not\-ready/ready；

- Mooncake put/get latency；

- speculative overwrite count；

- CPU offload store/load count。

普通未启用 artifact 的 serving 性能应与基线比较 TTFT、TPOT 和 throughput。



## 13\. 已放弃或延后的方案



以下不再属于当前简单模式：



- HTTP 返回 request\-scoped mmap handle；

- VIME 使用 `openat` 读取 vLLM `/dev/shm`；

- VIME materialize 后调用 ack endpoint；

- 为简单模式增加 lease/TTL/ACKED 外部协议；

- 用户手工拼接 head/full\-block/tail；

- 把 `#45635` physical slot mmap 当成 immutable artifact store。

以下延后到生产 E2E 暴露真实需求后再决定：



- 修改 Mooncake key 生成；

- 新增 Mooncake deterministic bundle API；

- 新增 TransferQueue artifact control\-state machine；

- 独立 materialize\(handle\) 服务；

- VIME 向 TQ publish artifact handle。

## 14\. 调研结论的保留与修正



此前源码调研仍有三点有价值：



1. TransferQueue 的价值主要是 sample/dataloader/control abstraction，而不是让

vLLM 绕开它直接使用 Mooncake。

2. `#45635` 的 shared slot mmap 很适合 worker/scheduler staging，但 physical slot

生命周期决定了它不能直接承担跨请求 artifact retention。

3. Mooncake/TQ 当前接口是否缺少幂等 block dedup 或 sample atomic\-ready，只能由

真实 adapter E2E 证明，不能先假定必须修改上游。

旧 V3 的以下判断已被替换：



- “简单版 HTTP 返回 SHM handle”被替换为“内部 SHM backend，HTTP 返回实际值”；

- “TQ 只做 control plane、vLLM 直接写 Mooncake”被替换为

“ArtifactConnector \-\> TQ \-\> Mooncake 单写”；

- “首版一律拒绝 speculative”被替换为“不 blanket 拒绝，以 accepted\-token

correctness 和 E2E 决定支持范围”；

- “prompt logprobs 暂不进入 artifact”被替换为独立 PR C 的 mandatory profile；

- “需要预先修改 Mooncake/TQ API”被替换为先适配现有接口。

## 15\. 调研源码快照



旧调研使用的外部源码快照保留作背景，不代表开始生产 PR 时无需重新核对：



|项目|调研 revision|用途|
|---|---|---|
|Ascend/TransferQueue|`b75d570d88c50bbfcbe2171baa727fadd7216f76`|sample/schema/ready/Mooncake client/dataloader|
|kvcache\-ai/Mooncake|`ac010838926e3cff2659465bd9b7bf6c0e9656bf`|structured object 与 storage semantics|
|VIME|`5cabf1f3e459e3df276c5b1f21478c475f04fb9d`|rollout/Sample/train\_data consumer|



当前实现判断以本地 `#45635 + PR3 @ 8c01cea0d5` 和 review worktree 为准。

开始 PR D/E 前必须重新 pin 并核对 TransferQueue、Mooncake、VIME 的最新接口。



