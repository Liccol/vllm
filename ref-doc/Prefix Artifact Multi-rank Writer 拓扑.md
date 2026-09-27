# Prefix Artifact Multi\-rank Writer 拓扑

# Prefix Artifact Multi\-rank Writer 拓扑设计



日期：2026\-08\-03



状态：基于当前代码、设计文档和 8×H800 初步实验重新整理。本文区分：



\- **vLLM 拓扑可运行**：对应 artifact\-off runtime；

\- **public R3 可返回**：对应 \`enable\_return\_routed\_experts\`；

\- **Artifact writer 已兼容**：需要真实 rank\-private capture、写入和回读验证。



前两项通过不等于第三项通过。当前尚未完成 in\-tree multi\-rank writer，整体结论为

`PRELIMINARY-NOT-READY`。



资料来源：



- \[Prefix Artifact 功能与并行兼容性\]\(\./Prefix Artifact 功能与并行兼容性\.md\)

- \[Prefix Execution Artifact Store：可落地设计 V3\]\(\./Prefix Execution Artifact Store：可落地设计 V3\.md\)

- \[TP multi\-rank writer 初步实验\]\(\./20260802\_023850\_prefix\_artifact\_multi\_rank\_writer\_experiment\.md\)

- \[非 TP 并行拓扑初步实验\]\(\./20260803\_133629\_non\_tp\_parallel\_topology\_experiment\.md\)

- \[实验索引\]\(\./INDEX\.md\)

## 1\. Background and Motivation



### 1\.1 Artifact 与 R3



Prefix Execution Artifact 把推理过程中产生、并且能够按 prefix 复用的数据保存为

不可变对象。本阶段的首个字段是 R3（routed experts）：



```Plain Text
logical shape = [executed_tokens, num_moe_layers, topk]
coordinate    = EXECUTED_TOKEN
```



完整 block 使用与 KV prefix 对齐的 content key；未满 block 的 tail 只属于当前

request，不参与 prefix reuse：



```Plain Text
block_key = H(schema, model_namespace, field_profile, kv_block_hash)
tail_key  = H(schema, model_namespace, request_instance, executed_end)
```



`rank`、`DP rank`、PID、physical KV slot 和 writer ID 都不能进入

`block_key`。否则相同 prefix 会因为部署拓扑不同而失去复用能力。



### 1\.2 Single\-writer 基线



当前 public R3 的主路径接近 single\-output\-rank：



```Plain Text
MoE router hook
  -> worker GPU capture
  -> full scheduled-row D2H
  -> ModelRunnerOutput
  -> executor output rank
  -> scheduler physical-slot buffer
```



\[RoutedExpertsCapturer\]\(\.\./vllm/model\_executor/layers/fused\_moe/routed\_experts\_capturer\.py\)

维护 `[max_tokens, layers, topk]` GPU buffer；同步路径在

\[GPUModelRunner\]\(\.\./vllm/v1/worker/gpu\_model\_runner\.py\) 中把本 step 的完整

`routing_data` 和 `slot_mapping` D2H。Multiproc 默认只返回 last PP stage 的

第一个 TP rank，详见

\[multiproc\_executor\.py\]\(\.\./vllm/v1/executor/multiproc\_executor\.py\)。



Artifact 的 single\-writer 基线应改为：



```Plain Text
all required rank-local capture
  -> topology restore
  -> canonical logical blocks
  -> one writer
  -> ArtifactStore
```



它的价值是先建立唯一 ownership、全局逻辑顺序、finalize、失败处理和

W=1 correctness oracle。single\-writer 应一直保留为默认值和回归基线。



### 1\.3 为什么需要 Multi\-rank Writer



single\-writer 的 D2H、encode、CPU queue 和 backend client 都集中在一个 rank。

当 Artifact profile 很大、并发高，或节点有多 NIC/NUMA 路径时，该 rank 可能成为

吞吐和 TPOT 瓶颈。



multi\-rank writer 的含义是：



*不同 logical blocks/fragments 分配给不同 writer；不是多个 rank 无协调地覆盖同一*

*object。*



目标数据流为：



```Plain Text
rank-private capture
  -> topology restore / logical coordinates
  -> GPU owner filtering
  -> owner-only D2H
  -> per-writer bounded queue
  -> parallel backend put
  -> receipt aggregation
```



owner filtering 必须在 GPU D2H 之前发生。若每个 writer 先复制全部 payload 再在

CPU 过滤，总 D2H 会从 `1x` 放大为 `W x`，失去扩展意义。



### 1\.4 必须保持的不变量



1. 一个 logical row 只属于一个 writer；不能 missing、duplicate 或 wrong\-owner。

2. publish 前恢复 global request/token/layer/top\-k 顺序。

3. speculative rejected、stop\-trimmed 和 async in\-flight rows 不得 commit。

4. content object immutable；ready 只能在完整 payload 原子可见后成立。

5. EngineCore/executor 只传 plan、ACK、no\-work、error 和 credit，不传大 payload。

6. writer crash、queue full 或 put failure 必须 fail closed，不能提前返回 ready。

7. W=1 与 W\>1 必须产生相同 content keys、shape、dtype、checksum 和 logical bytes。

decode 时 open block 尚没有最终 `kv_block_hash`。第一版可使用稳定的临时 owner：



```Plain Text
open_owner = H(group_seed, request_instance_id, logical_block_index) % W
```



同一临时 owner 持续接收该 open block，并在 block seal 后以最终 content key

发布，避免 payload 在 writer 间二次迁移。writer 位置只是 storage metadata，

不能写入 content key。SHM 若按 writer 分 shard，需要额外的

`content_key -> shard` locator；Mooncake 的全局 key lookup 不需要暴露 writer。



这一方案允许两个 request/DP replica 对同一最终 content key 选到不同临时 owner，

因此 backend 必须提供以下至少一种机制：



- atomic create\-if\-absent/CAS，并向 loser 返回可见状态；

- per\-key lease 或 immutable object \+ atomic commit record；

- coordinator 按 content key 串行 publish。

`exists() + put()` 存在 TOCTOU race，不能替代 CAS。



## 2\. vLLM Parallel Topologies



### 2\.1 当前结论总览



|方式|vLLM runtime|当前 public R3 初筛|Artifact multi\-writer 结论|
|---|---|---|---|
|TP|支持|TP8 PASS|第一优先级，仍缺 rank\-private writer E2E|
|SP|内部执行优化，不是独立 world axis|未做专门 branch\-hit 验证|原理可行，不能标记已通过|
|DP<br>|支持|DP2、两个 TP4 engine PASS<br>|每个 replica 独立 writer group；shared backend 被 CAS 阻塞|
|EP|支持|EP off/on PASS 且 hash 相同|条件兼容；EPLB/private layout 未验证|
|DCP<br>|支持，复用 TP/PCP ranks|DCP2/4 被 R3 guard 拒绝|upstream blocked；需 logical slot adapter|
|PCP<br>|支持但组合有限制<br>|PCP2 在实验 wheel 中 crash|upstream blocked；需 padding/restore adapter|
|PP|支持<br>|PP2 被 R3 config guard 拒绝|upstream blocked；需跨 stage layer assembly|
|Multiproc / Ray|支持|普通 Ray TP8 与 Multiproc hash 相同|需 all\-writer receipt aggregator；Ray compiled DAG 未测|



这里没有独立的“CP\-only”结论。当前 vLLM 把 context parallelism 分为 PCP 和 DCP：

PCP 扩大 process world，DCP 在没有 PCP 时复用 TP ranks。即使配置

`TP=N, DCP=N`，也不能把结果解释成一个脱离 TP 轴的通用 CP 验证。



### 2\.2 所有拓扑共用的改造



当前代码按 physical slot 保存和返回 R3。multi\-writer 需要增加四个公共层：



1\. **Scheduler publish plan**：携带 request instance、accepted logical range、

logical block index、最终 KV hash（可用时）、writer group 和 topology epoch。

2\. **Topology adapter**：把 rank\-private physical layout 恢复为

`(request, logical_token, logical_layer, topk)`。

3\. **Worker data plane**：在 GPU 上执行 owner selection，只将 owner rows D2H，

encode 后放入每 writer 有界队列。

4\. **Receipt aggregator**：聚合 \`ACK/no\-work/error/credit\`，保留正常

`ModelRunnerOutput` 的 output\-rank 语义。



现有可复用入口：



- \[routed\_experts\_capturer\.py\]\(\.\./vllm/model\_executor/layers/fused\_moe/routed\_experts\_capturer\.py\)：

router hook、DP slice、SP all\-gather；

- \[gpu\_model\_runner\.py\]\(\.\./vllm/v1/worker/gpu\_model\_runner\.py\)：

slot snapshot、D2H、async snapshot；

- \[outputs\.py\]\(\.\./vllm/v1/outputs\.py\)：

`RoutedExpertsTensors`、`RoutedExpertsLists`、`ModelRunnerOutput`；

- \[scheduler\.py\]\(\.\./vllm/v1/core/sched/scheduler\.py\)：

current R3 guard、physical\-slot manager 和 authoritative request lifecycle；

- \[KVOutputAggregator\]\(\.\./vllm/distributed/kv\_transfer/kv\_connector/utils\.py\)：

all\-worker control receipt 聚合的参考实现。

Artifact payload 应停留在 worker data plane，不能继续经

`ModelRunnerOutput -> EngineCore` 搬运。



### 2\.3 TP



**当前布局**



TP 是第一版 multi\-writer 的最佳入口。当前 capturer 已能在支持的 MoE 路径上获得

完整或可恢复的 routing rows，public R3 在真实 TP8 上可以返回。



**需要修改**



```Plain Text
scheduler block plan
  -> every TP worker computes the same stable owner
  -> owner rank selects rows on GPU
  -> owner-only D2H / encode / put
  -> all writers return receipts
```



若某 kernel 路径中每个 rank 已有完整 logical R3，只需 owner filter；若只有 token

shard，则先 all\-gather 保证正确性，后续再优化为 owner\-targeted all\-to\-all。



**坑点**



- 当前 `GPUModelRunner` 仍按 step 执行 full D2H，必须绕开，否则只有 backend

并行而 D2H 没有扩展；

- Multiproc 默认只取一个 output rank，非 output rank 的 writer receipt 会丢失；

- 跨节点 TP 还要验证 writer/client 与 NIC/NUMA 亲和性；

- 小 payload 固定开销高，不能默认设置 `W=TP`。

### 2\.4 SP



SP 在当前 vLLM 中不是独立的 `sequence_parallel_size` 拓扑轴，至少包含两类路径：



- MoE SP：EP/DP/TP 和特定 all\-to\-all backend 组合下的 token shard；

- compilation SP：由 compilation pass 控制，捕获位置和 tensor layout 可能不同。

\[RoutedExpertsCapturer\.capture\]\(\.\./vllm/model\_executor/layers/fused\_moe/routed\_experts\_capturer\.py\)

已有 SP 分支：当 `topk_ids` 行数为

`ceil(token_num_per_dp / tp_size)` 时，通过 TP all\-gather 恢复该 DP rank 的 rows，

并裁掉尾部 padding。



**需要修改**



- correctness\-first：复用现有 all\-gather，恢复后按 block owner 在 GPU 过滤；

- performance path：确认每行 global token index 后，直接 all\-to\-all 到 owner；

- 对 MoE SP 和 compilation SP 分别增加 branch\-hit 和 rank\-private checksum probe。

**坑点**



现有分支存在不等于已验证。本轮没有证明该分支真实命中，也没有比较每个 SP rank

private capture 与 canonical output，因此 SP 只能标为“预期可支持”。



### 2\.5 DP 与 External DP



**当前布局**



每个 DP replica 有独立 scheduler、request lifecycle 和 TP/EP worker group。

capturer 会根据 `dp_metadata` 截取本 DP rank 的 token rows。



**需要修改**



- 每个 DP replica 建立独立 writer group 和 receipt aggregator；

- replica 内部执行 TP/SP/EP restore，不在 DP replica 间传 payload；

- 所有 replica 使用相同 content\-key 规则，DP rank 只能出现在 routing metadata；

- external DP 需要 backend CAS/lease，或由外部 coordinator 对 content key 串行。

**坑点**



不同 replica 处理相同 prefix 时会写入相同 `block_key`，这是正确的 dedup 行为，

不是 key 设计错误。给 key 加 DP rank 虽能避开竞争，但会破坏 prefix reuse，不能作为

修复。当前 Mooncake same\-key 语义没有 caller\-visible conflict，direct concurrent

put 暂不兼容。



### 2\.6 EP



**当前布局**



EP 改变 expert 的 physical placement 和通信方式，但 Artifact 保存的是 router 的

global logical expert IDs，不保存 physical expert rank。当前 public R3 的 EP off/on

结果一致。



**需要修改**



- 在 logical expert ID 尚未被 EPLB/physical mapping 改写的位置 capture；

- 按 logical token/block 做 owner routing，不能按 physical expert owner 写对象；

- writer group 仍以 DP replica 为边界；

- elastic EP membership 变化期间使用 epoch \+ drain，禁止执行中改变 writer group。

**坑点**



当前只验证了 public output。rank\-private pre\-mapping capture、EPLB placement、

redundant experts 和 elastic EP 尚未验证；不能由 EP off/on hash 相同直接推出

multi\-writer 已兼容。



### 2\.7 DCP



**当前布局**



\[ParallelConfig\]\(\.\./vllm/config/parallel\.py\) 中 DCP 主要分片 decode KV cache。

没有 PCP 时它复用 TP ranks，不扩展 process world。它首先改变的是 attention/KV

physical layout，不代表 R3 自然按 DCP rank 分片。



**需要修改**



- scheduler 以 logical executed\-token range 和 KV block hash 下发 plan；

- 增加 DCP physical slot/interleave 到 logical block 的 adapter；

- 选定 canonical attention KV group 和 block size；

- adapter 之后再执行 owner filter，DCP rank/slot 不进入 Artifact key。

**坑点**



当前 \[Scheduler\]\(\.\./vllm/v1/core/sched/scheduler\.py\) 在 R3 开启时要求

`dcp_world_size == 1 && pcp_world_size == 1`。DCP2/DCP4 artifact\-off 可运行，

但 public R3 被该 guard 拒绝。删除 guard 之前必须先证明 miss/hit、decode、

preemption 和 rank\-private logical ordering。



### 2\.8 PCP



**当前布局**



PCP 扩展 process world，并把 prefill token 切成包含 dual chunks、duplicate/padding

位置的 local layout。R3 初始形式可视为：



```Plain Text
[local_tokens_with_padding, all_local_layers, topk]
```



\[PCPManager\.restore\_hidden\_states\]\(\.\./vllm/v1/worker/gpu/pcp\_manager\.py\) 已有

PCP all\-gather \+ restore index，可作为 R3 restore 的实现参考。



**需要修改**



- 复用 PCP 的 global request/token mapping 和 restore index；

- 丢弃 padding、重复位置，只保留唯一 executed\-token rows；

- single\-writer：gather \+ restore 到 canonical rank；

- multi\-writer：按 logical block owner all\-to\-all，再由 owner reorder/assemble；

- 使用一致的 collective 顺序，所有 no\-work rank 也必须参与并返回 receipt。

**坑点**



当前 scheduler guard 拒绝 PCP\+R3。实验使用的 vLLM 0\.26\.0 wheel 在开启 R3 后切到

V1 runner，报错表现为目标 buffer 的 512 rows 与 PCP expanded tensor 的 1024 rows

发生 expansion mismatch。这证明当前 R3 buffer sizing 与 PCP expanded/padded layout

没有对齐；支持前需要先完成 logical restore。它不是 Artifact writer 自身的错误。



此外，当前 config 明确拒绝 `PCP > 1 && DP > 1`；PCP2\+DCP2 在本轮

artifact\-off MLA warmup 中还出现 `seq_lens=None` crash。组合支持必须分别解锁，

不能只移除 R3 guard。



### 2\.9 PP



**当前布局**



PP stage 只持有部分 model layers，因此 rank\-private R3 形式是：



```Plain Text
[all_local_tokens, local_layers, topk]
```



最后 stage 的 output rank 不天然拥有前面 stage 的 R3 layers。



**需要修改**



第一版优先使用完整对象方案：



```Plain Text
stage-local layer fragments
  -> existing PP transport / fixed-order gather
  -> last-stage TP writer group
  -> block ownership / owner-only D2H / put
```



若 stage gather 成为瓶颈，再扩展为每 stage 写

`(content_key, pp_stage_id)` fragment，并用 mandatory stage bitmap/manifest

原子 seal。没有 manifest barrier 时，各 stage 不能直接覆盖同一个 key。



**坑点**



- \[VllmConfig\]\(\.\./vllm/config/vllm\.py\) 当前明确拒绝

`enable_return_routed_experts && PP > 1`；

- pipeline batch queue 下必须绑定相同 request instance、step sequence 和 accepted

boundary；

- stage failure、retry、empty stage 和异步批次不能导致 manifest 提前 ready；

- last\-stage gather 简单但可能形成新的 layer\-assembly 热点。

### 2\.10 Multiproc 与 Ray



这两者不是 Artifact layout 轴，但决定 writer receipt 能否回到 scheduler。



- Multiproc 的 `execute_model` 默认设置 `unique_reply_rank=output_rank`；

- 普通 Ray 无 connector 时只取 output rank；有 connector 时才拉取所有 workers 并由

`KVOutputAggregator` 聚合；

- Artifact 应增加同类轻量 aggregator，只聚合 receipt，不聚合 payload；

- Ray compiled DAG 的 zero\-copy detach、错误传播和 all\-worker return 需要单独验证，

不能从普通 Ray 结果外推。

## 3\. Preliminary Experiments



### 3\.1 证据分级



|Gate|含义|当前覆盖|
|---|---|---|
|G0|artifact\-off 拓扑可启动和生成|已做大部分拓扑初筛|
|G1|physical layout 可恢复到 canonical logical rows|目前主要是 deterministic model|
|G2|W=1/W\>1 keys、bytes、readback、原子性一致|TP prototype 部分完成；same\-key 未通过|
|G3|D2H、TTFT/TPOT、吞吐、故障和 soak|仅 TP/Mooncake 初步性能点|



协议模型 PASS 只能证明设计自洽，不能替代真实 worker\-private capture 和 backend

readback。



### 3\.2 TP / Mooncake 原型



实验环境为单机 8×H800，完整结果见

\[TP multi\-rank writer 报告\]\(\./20260802\_023850\_prefix\_artifact\_multi\_rank\_writer\_experiment\.md\)。



主要结果：



- ownership/byte correctness 11/11 cases PASS；

- missing、duplicate、wrong\-owner rows 均为 0；

- owner\-filtered W=1/2/4/8 aggregate D2H amplification 均为 `1.0x`；

- 4096 blocks、W=8 的 owner load `max/mean=1.0605`；

- Mooncake 回读 byte\-exact，实验 key 不包含 writer/rank。

M2 pipeline 相对 true single writer：



|Payload/block||W=4 speedup|结论|
|---|---|---|---|
|16 KiB||0\.348x|明显退化|
|256 KiB||1\.254x|未达到 2x|
|1 MiB||2\.514x|大 payload 有收益|



当前 break\-even 位于 256 KiB 到 1 MiB 之间。Mooncake\-only 1 MiB fresh\-process

测试中 W=4/QD=1 为 `1.957x`，没有严格达到 2x；W=8 已接近饱和且尾延迟增大。



真实 TP8 public R3 probe 返回四个 arrays：

`[13,2,2]`、`[15,2,2]`、`[12,2,2]`、`[16,2,2]`，合计 224 B，

重复运行 byte\-exact。它证明 public return path 可运行，但没有导出全部 8 个 worker

的 private capture，也没有接入真实 multi\-writer。224 B 远低于本轮 break\-even，

R3\-only 不应默认启用 multi\-writer。



### 3\.3 非 TP 拓扑初筛



完整结果见

\[非 TP 拓扑报告\]\(\./20260803\_133629\_non\_tp\_parallel\_topology\_experiment\.md\)。



协议模型：



- 15/15 cases PASS；

- 128 requests、33,536 logical rows、209 canonical objects；

- W=1 与 W\>1 key/checksum diff 为 0，model D2H accounting 为 `1.0x`；

- 三个 PCP cases 各排除 340 个 padding slots；

- 该模型没有读取真实 private capture，也没有执行真实 GPU filter 或 Mooncake G2。

真实 runtime：



|拓扑|Artifact\-off|Public R3|当前解释|
|---|---|---|---|
|DP2×TP4|PASS|PASS|writer 未集成；shared backend 有 CAS blocker|
|两个 external TP4 engines|PASS|PASS|同上|
|TP8, EP off/on|PASS|PASS，hash 相同|fixed EP 条件兼容|
|TP8, DCP2/DCP4|PASS|expected rejection|scheduler context\-parallel guard|
|TP4×PCP2|PASS|FAIL|tested wheel 512/1024 expansion mismatch|
|TP4×PP2|PASS|expected rejection|PP config guard|
|Ray TP8|PASS|PASS|hash 与 Multiproc TP8 相同；compiled DAG 未测|
|TP4×PCP2×DCP2|FAIL|未跑|MLA warmup `seq_lens=None`|



总计 artifact\-off 15/16 PASS。真实 R3 尝试 10 个配置：6 PASS、3 个 expected

rejection、1 个 PCP upstream failure。PP/DCP/PCP 的结论是“当前被 upstream

guard/crash 阻塞”，不是“设计上永久不支持”。



### 3\.4 Mooncake Same\-key / CAS



Mooncake 0\.3\.12、两个独立 clients、同一个 256 KiB key：



|场景|两个 writer 返回值|Readback|业务结论|
|---|---|---|---|
|相同 key \+ 相同 bytes|`0, 0`|bytes 正确|无 caller\-visible loser|
|相同 key \+ 不同 bytes|`0, 0`|只保留其中一份|loser 没有 conflict/error|



因此该问题**不会稳定报错**：两个 writer 都会认为 put 成功。冲突 payload 的 readback

只保留一个值，另一方无法知道自己已丢失。业务要求

`cas_requirement_status=UNSUPPORTED`。



这对 DP/shared backend 是硬 blocker；对同一 engine 内“不同 key 分给不同 writer”

的吞吐并行不是 blocker。若 Mooncake 接口不修改，第一版必须在 producer/coordinator

内按 content key 串行 publication，同时保留不同 keys 的并行。



### 3\.5 当前可得出的结论



1. owner\-filtered multi\-writer 的算法和大 payload 收益已有初步证据。

2. TP、DP、EP、普通 Ray 的 public R3 有初筛，但真实 Artifact writer 尚未集成。

3. SP 没有专门验证；DCP、PCP、PP 被当前 R3 guard/crash 阻塞。

4. same\-key content publication 的 key 本来就应相同；问题是 backend 缺少原子冲突

语义，不应通过修改 content key 绕开。

5. 当前资料不足以声明任一非 TP 拓扑已通过 rank\-private multi\-writer E2E。

## 4\. Target and Validation



### 4\.1 最终目标



```Plain Text
Scheduler / EngineCore
  | logical publish plan, finalize, discard
  v
GPU workers
  capture
  -> topology-specific logical restore
  -> stable block ownership
  -> GPU owner filter
  -> owner-only D2H
  -> bounded encode/store queue
  -> SHM or Mooncake
  |
  +-> ACK / no-work / error / credit
        -> executor aggregator
        -> scheduler readiness/finalize
```



最终实现需要同时满足：



- single\-writer 和 multi\-writer 共用 key/schema/finalize contract；

- W 可按 profile/payload/backend 能力选择，默认 W=1；

- 每个 DP replica 有独立 writer group，但共享 content namespace；

- TP/SP/DCP/PCP/PP adapter 只处理 layout，不改变 Artifact identity；

- backend publication 对 same\-key equal/conflict 都有明确、可观测的原子结果；

- large payload 不经过 EngineCore。

### 4\.2 建议实施顺序



1\. **W=1 基线**：完成真实 logical block assembly、SHM/Mooncake readback、

finalize/abort/spec acceptance 和 disabled\-path regression。

2\. **TP multi\-writer**：加入 stable owner、GPU filter、per\-writer queue，以及

Multiproc/Ray receipt aggregator。

3\. **DP/EP**：建立 replica\-local writer group；先解决 CAS/serialization，再做

shared\-backend same\-prefix E2E 和 EPLB。

4\. **SP/DCP/PCP**：逐个增加 rank\-private layout dump、restore adapter 和

branch\-hit tests；满足 gate 后才移除对应 R3 guard。

5\. **PP**：先实现 stage gather 到 last\-stage writer group；有性能证据后再考虑

fragment manifest。

6\. **组合与生产化**：PCP\+DCP、PP\+DCP、DP\+PP、Ray compiled DAG、RDMA、多节点、

backpressure、故障和 soak。



### 4\.3 正确性验收



每个受支持拓扑至少验证：



- W=1 与 W=2/4/8 的 ordered keys、shape、dtype、checksum、materialized bytes

完全一致；

- 每个 private rank 的 input row coverage 可追踪，missing/duplicate/conflict 为 0；

- prefix cold/full\-hit/partial\-hit 与 cache\-off R3 exact equality；

- preemption、abort、frontend stop、spec rejection 和 async scheduling 不提交越界 row；

- no\-work rank 返回 receipt，任一 writer error 时 request 不 ready；

- PP 所有 mandatory stage 完成前 object/manifest 不可见；

- DP replicas 同 prefix 使用同 content key，equal race 可 dedup，conflict race

fail closed；

- writer restart、slow writer、queue full、timeout 和 retry 不产生 partial ready。

建议最小拓扑矩阵：



|组|配置|
|---|---|
|基线|TP1/2/4/8，Multiproc|
|Executor|Ray TP8，Ray compiled DAG|
|Replica/Expert|DP2×TP4、2×external TP4、EP off/on、EPLB|
|Context|SP branch、DCP2/4、PCP2|
|Pipeline|PP2、TP4×PP2|
|组合|DP2×TP2×PP2、TP4×PP2×DCP2、TP4×PCP2×DCP2|



### 4\.4 性能与稳定性验收



- owner\-filtered aggregate D2H amplification `<= 1.10x`；

- 足够多 blocks 时 owner bytes `max/mean <= 1.25`；

- disabled 和 W=1 路径无显著 TTFT/TPOT regression；

- 只有达到预设 payload 阈值且 W\>1 有稳定收益时才自动启用；

- 记录 D2H、encode、queue wait、put、ready latency 和 per\-writer imbalance；

- 对 16 KiB、256 KiB、1 MiB 以及真实 Artifact profiles 重测 break\-even；

- TCP 与 RDMA 分开结论，验证 connection reuse、batch put 和 registered buffers；

- 至少 30\-60 分钟 soak，无 queue/memory growth、stuck receipt 或残留对象失控。

### 4\.5 Go / No\-Go



以下任一条件存在时不得声明 topology compatible：



- 仍被 upstream guard/crash 阻塞；

- 只有 deterministic protocol model，没有真实 rank\-private capture；

- W=1/W\>1 没有 backend byte\-exact readback；

- shared backend 没有 CAS/lease/commit 或 producer\-side per\-key serialization；

- writer failure 可导致提前 ready；

- payload 仍通过 EngineCore 聚合；

- 性能收益只来自 synthetic unique keys，且真实 profile 低于 break\-even。

在这些 gate 全部完成前，生产默认保持 single writer。



