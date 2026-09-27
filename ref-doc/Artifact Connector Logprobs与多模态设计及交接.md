# Artifact Connector：Logprobs 与多模态 Artifact 设计及交接

更新日期：2026-09-27  
代码基线：`main`（调研时 HEAD `3b4566c5cf`）  
Routed-expert/AuxOutput 参考提交：`0b7f11a1eece3dc9c689f1b5ab90d60c03b10793`

## 1. 文档目的

本文整理当前已经完成的代码调研和设计讨论，供后续 agent 继续实现。

本文不表示 logprobs 或多模态 Artifact 已经实现。当前代码已经具备 routed-expert
AuxOutput 的捕获、block-keyed 本地存储和回读能力；本文在此基础上确定：

- 哪些 logprobs 和多模态数据值得保存；
- AuxOutput 与 Artifact Connector 的职责边界；
- 字段、key、object、完成屏障和失败语义；
- 推荐的实现顺序和验证范围。

相关总体设计：

- [Prefix Execution Artifact Store：可落地设计 V3](./Prefix%20Execution%20Artifact%20Store：可落地设计%20V3.md)
- [Prefix Artifact 功能与并行兼容性](./Prefix%20Artifact%20功能与并行兼容性.md)
- [Prefix Artifact Multi-rank Writer 拓扑](./Prefix%20Artifact%20Multi-rank%20Writer%20拓扑.md)
- [TransferQueue / Mooncake 接口调研](./TransferQueue%20_%20Mooncake%20接口调研.md)

## 2. 核心结论

不要把所有 logprobs 和所有多模态 tensor 都默认保存。

三类数据的语义不同：

|数据|语义|默认结论|
|---|---|---|
|Routed expert IDs|推理时的离散执行决策|作为 execution replay artifact，mandatory|
|Response behavior logprobs|生成时策略对 accepted token 的评分|按 RL 算法启用，通常值得保存|
|多模态输入|训练样本来源和预处理结果|默认保存 manifest；payload 仅在不能确定性重建时保存|

推荐的默认 rollout bundle：

```text
R3 keys
+ accepted output token IDs
+ selected-token behavior logprobs（算法需要时）
+ multimodal sample identity/hash/placeholder/processor metadata
+ multimodal payload key（仅 snapshot mode）
```

明确不建议默认保存：

- prompt top-k logprobs；
- response top-k 或 full-vocab logprobs；
- full-vocab logits；
- 可以从稳定 dataset 重取的图片、音频、视频副本；
- 多模态 encoder outputs；
- scheduler budget、cache occupancy、GPU slot 等临时运行时状态。

## 3. 当前代码基线

### 3.1 已实现的 AuxOutput/R3 能力

`0b7f11a1ee` 建立了 routed-expert AuxOutput 数据路径：

1. scheduler 增量发送 request cursor 和 logical KV block hashes；
2. worker 在 forward 后 snapshot routed-expert tensor；
3. output copy stream 异步 D2H；
4. worker 按 request 和 accepted range 整理 rows；
5. speculative rejected suffix 不提交；
6. incomplete block 保留在 request-local buffer；
7. 完整 block 按 KV block hash 生成 key；
8. 本地 mmap store 提供 immutable object、引用计数、LRU 和后台写；
9. prefix reuse 时按 ordered keys materialize。

主要代码：

```text
vllm/distributed/aux_output_connector/
├── connector.py       scheduler metadata 和输出切片
├── worker.py          worker capture/commit 生命周期
├── routed_experts.py  R3 buffer、key、publish、materialize
└── store.py           fixed-size mmap object store
```

### 3.2 当前实现不能直接承载新字段的地方

- `AuxRequestOutput` 只有 `token_start + rows`，是 R3 专用结构；
- key 只有 generation + block hash，没有 model/field/profile namespace；
- `BlockObjectStore` 要求所有 object 使用同一个固定 `object_nbytes`；
- payload 是裸 bytes，没有 schema、shape、valid range 和 checksum；
- worker connector 同时拥有 capture、key、store 和 materialize，职责耦合；
- 没有 per-field receipt 和 request-level completion barrier；
- 没有生产 backend 的 ready marker/bundle；
- 没有 Artifact 与 KV 的 joint readiness；
- 当前 prompt logprobs 默认跳过 prefix-cache read：
  `SamplingParams.skip_reading_prefix_cache = prompt_logprobs is not None`。

因此新字段接入前，应先把 R3 capture 和 Artifact storage contract 解耦。

## 4. AuxOutput 与 Artifact Connector 的边界

### 4.1 AuxOutput 负责

- 从 model runner、sampler 或 EngineCore 捕获运行时数据；
- 维护 request cursor；
- 根据 accepted/rejected token 范围切片；
- 复用已有 async D2H；
- 将数据转换为 field batch；
- 不理解 SHM、Mooncake、TQ 或 public handle。

建议统一输出：

```python
@dataclass
class ArtifactFieldBatch:
    request_id: str
    field: str
    logical_start: int
    values: object
    terminal: bool = False
```

### 4.2 Artifact Connector 负责

- field profile；
- model/profile namespace；
- key 生成；
- object envelope 和 checksum；
- full block、request chunk、tail、request item；
- backend put/get；
- per-field ordered refs；
- request completion barrier；
- bundle/ready marker；
- fail-closed 和 orphan cleanup。

ArtifactStore 只处理 opaque bytes，不理解 R3、logprobs、image 或 audio。

### 4.3 Producer ownership

|字段|唯一 producer|
|---|---|
|`routed_experts`|output TP rank 的 AuxOutput worker|
|`behavior_logprobs`|output TP rank 的 sampler/AuxOutput worker|
|`multimodal_manifest`|EngineCore input-processing 侧|
|`multimodal_payload`|EngineCore input-processing 侧|
|terminal bundle/ready marker|ArtifactRequestCore|

不同字段可以来自不同执行位置，但同一个字段不能重复 publish。

## 5. 通用字段模型

建议增加：

```python
class ArtifactScope(StrEnum):
    PREFIX_BLOCK = "prefix_block"
    REQUEST_ONLY = "request_only"


class ArtifactCoordinate(StrEnum):
    EXECUTED_TOKEN = "executed_token"
    PREDICTED_TOKEN = "predicted_token"
    ACCEPTED_OUTPUT_TOKEN = "accepted_output_token"
    REQUEST_ITEM = "request_item"
    REQUEST = "request"


@dataclass(frozen=True)
class ArtifactFieldProfile:
    name: str
    schema_version: int
    scope: ArtifactScope
    coordinate: ArtifactCoordinate
    codec: str
    profile_id: str
    mandatory: bool
```

建议字段矩阵：

|Field|Scope|Coordinate|默认状态|
|---|---|---|---|
|`routed_experts`|`PREFIX_BLOCK`|`EXECUTED_TOKEN`|mandatory|
|`behavior_logprobs`|`REQUEST_ONLY`|`ACCEPTED_OUTPUT_TOKEN`|算法决定|
|`multimodal_manifest`|`REQUEST_ONLY`|`REQUEST`|MM request mandatory|
|`multimodal_payload`|`REQUEST_ONLY`|`REQUEST_ITEM`|optional|
|`prompt_topk_logprobs`|`PREFIX_BLOCK`|`PREDICTED_TOKEN`|默认关闭|

RL behavior logprobs 与 serving prompt logprobs 不应合并为一个 mandatory field。

## 6. Behavior logprobs 设计

### 6.1 训练价值

Behavior logprobs 可用于：

- PPO/GRPO importance ratio；
- asynchronous rollout 的 behavior-policy 固化；
- 避免保留完整 old-policy engine 或重新 forward；
- 训推数值差异诊断；
- policy-lag 过滤和审计。

它不是 execution replay 数据。保存 logprobs 不会让训练 forward 自动与推理一致，
只能记录生成策略对 token 的评分。

### 6.2 默认存储内容

只保存 accepted response token 的 selected-token score：

```python
@dataclass
class BehaviorLogprobsChunk:
    token_start: int
    token_ids: np.ndarray   # int32 [N]
    logprobs: np.ndarray    # float32 [N]
```

默认不保存 top-k token IDs、top-k scores、rank 或 full-vocab logits。

Object header 至少包含：

```json
{
  "schema_version": 1,
  "field": "behavior_logprobs",
  "kind": "request_chunk",
  "request_id": "...",
  "request_attempt_id": 0,
  "token_start": 0,
  "valid_len": 128,
  "score_semantics": "raw_logprobs",
  "policy_version": "...",
  "sampling_profile_id": "...",
  "token_dtype": "<i4",
  "score_dtype": "<f4",
  "payload_sha256": "..."
}
```

`score_semantics` 必须明确区分 raw/processed，以及 temperature、top-k/top-p、
logits processor 等采样语义。不同语义不能共享 profile/key namespace。

### 6.3 Capture 数据流

当前 `AsyncOutput` 已经把 `SamplerOutput.logprobs_tensors` 异步复制到 CPU，不能
为 Artifact 再做一份 D2H：

```text
SamplerOutput.logprobs_tensors
    -> existing async D2H
    -> AsyncOutput.logprobs_tensors
       ├─ existing API output
       └─ BehaviorLogprobsAdapter
          -> ArtifactRequestCore
```

扩展 `PendingAuxOutput`，让它引用这份 CPU tensor、sampled token IDs 和每个 request
的 `num_sampled`。在 copy event synchronize 后提交。

### 6.4 Accepted-token 规则

- speculative rejected suffix 不提交；
- frontend stop trim 后不返回的 token 不提交；
- preemption/resume 不重复写已提交 token；
- async output 继续按 step 顺序消费；
- 每个 request 维护单调 `output_cursor`；
- selected score 必须与最终 sampled token ID 对齐；找不到对应 token 时 fail closed。

### 6.5 与 API logprobs 解耦

Artifact 模式不能要求调用者额外设置 HTTP `logprobs` 参数。建议增加：

```python
AuxOutputConfig:
    enable_behavior_logprobs: bool
```

启用后 sampler 至少产生 selected-token logprob。优先实现 selected-only 计算，避免
为了 Artifact 构造完整 top-k 表。必须单独测量其 sampler/log-softmax 开销。

### 6.6 Key 与分块

Behavior logprobs 不跨 request 去重：

```text
vllm-artifact/
  <model_namespace>/
  behavior-logprobs/<profile_id>/
  request/<request_id>/<attempt_id>/chunk/<chunk_index>
```

建议每 256 或 1024 个 accepted tokens 一个 request chunk：

- 与生成重叠写出；
- 避免长 response 全量驻留内存；
- abort 时不发布 terminal bundle；
- orphan chunks 由 TTL 清理。

## 7. 多模态设计

### 7.1 值得保存的内容

多模态默认保存小型 `multimodal_manifest`：

- sample/request identity；
- processor profile；
- prompt token IDs 或稳定引用；
- item 顺序；
- modality；
- `mm_hash`；
- `identifier`；
- `mm_position.offset/length/is_embed`；
- 外部 dataset/object-store reference；
- 可选 payload key。

只有输入无法确定性重建时，才保存实际 payload。

### 7.2 Payload mode

```python
MultimodalPayloadMode = Literal[
    "reference_only",
    "raw_media",
    "processed_tensors",
]
```

#### `reference_only`

只保存 manifest 和稳定外部引用。适用于已有 dataset/object store，且 processor 可
确定性复现。

#### `raw_media`

保存原始 encoded bytes 和 decode/processor profile。适用于临时 URL、用户上传媒体
或训练需要重新运行可训练 encoder 的场景。

#### `processed_tensors`

保存 `MultiModalFeatureSpec.data`。适用于随机抽帧、crop、augmentation 无法稳定复现，
或 API 直接传入 embedding 的场景。

不要默认同时保存 raw media 和 processed tensors。

### 7.3 Capture 时机

必须在 EngineCore：

```text
mm_receiver_cache.get_and_update_features()
    -> Artifact multimodal capture
    -> Request/Scheduler
    -> strip_covered_mm_data()
```

原因：

- receiver cache 可能让 IPC 输入中的 `data` 暂时为 `None`；
- 恢复后才拥有完整 payload；
- `strip_covered_mm_data()` 会在发送 worker 前再次删除 prefix 已覆盖 item 的 data；
- 因此不能等 worker capture MM payload。

建议 EngineCore 接口：

```python
artifact_connector.stage_multimodal_request(
    request_id=request.request_id,
    mm_features=request.mm_features,
    prompt_token_ids=request.prompt_token_ids,
    profile=...,
)
```

该调用异步编码和 put，不阻塞 scheduler；terminal barrier 再等待完成。

### 7.4 Manifest 示例

```json
{
  "schema_version": 1,
  "field": "multimodal_manifest",
  "request_id": "...",
  "sample_id": "...",
  "processor_profile_id": "...",
  "prompt_token_ids_ref": "...",
  "items": [
    {
      "item_index": 0,
      "modality": "image",
      "mm_hash": "...",
      "identifier": "...",
      "position": {
        "offset": 12,
        "length": 576,
        "is_embed_key": null
      },
      "source_ref": {
        "kind": "dataset",
        "value": "dataset/sample/image-0"
      },
      "payload_key": null
    }
  ]
}
```

`mm_hash` 是 processor/input identity；`identifier` 还可能包含 LoRA identity。两者都
需要保存。

### 7.5 Payload key 与去重

MM 字段逻辑上是 request-only，但 immutable payload 可按内容物理去重：

```text
vllm-artifact/
  <model_namespace>/
  multimodal/<processor_profile_id>/
  content/<mm_hash>
```

request manifest 引用 payload key。MM payload 存在不等于 KV prefix hit，也不参加
KV/R3 joint readiness。

Processed payload 必须把 processor profile 放入 namespace。若未来存 encoder output，
必须使用包含 model weights/LoRA/dtype 的独立 profile 和 identity；不要复用本字段。

### 7.6 Tensor-tree codec

禁止 pickle `MultiModalKwargsItem`、`MultiModalFieldElem` 或 field class。

统一 object 格式：

```text
uint32 header_length
canonical JSON header
concatenated tensor segments
```

codec 递归支持：

- Tensor；
- list/tuple；
- nested list；
- empty tensor；
-不同 dtype/shape。

每个 field 保存：

```text
field name
layout kind: batched | flat | shared
keep_on_cpu
concat dim（适用时）
tensor-tree descriptor
```

field name 不能做固定枚举；必须遍历实际 `MultiModalKwargsItem`，以兼容新模型。

### 7.7 不保存的 MM 运行时状态

- `scheduled_encoder_inputs`；
- encoder compute/cache budget；
- `free_encoder_mm_hashes`；
- worker GPU cache slot；
- `finished_sending/finished_recving`；
- device/pinned 状态；
- prefix stripping 后的 `data=None`；
- encoder cache 引用计数。

### 7.8 Encoder outputs

默认不存多模态 encoder outputs：

- 与权重、LoRA、dtype 强绑定；
- 权重更新后 stale；
- encoder 可训练时复用 detached output 会丢梯度；
- 现有 EncoderCache/EC Connector 已负责其缓存和传输。

只有 encoder 永久冻结且有明确复用收益时，才设计独立
`multimodal_encoder_outputs` profile。

## 8. Request completion barrier

建议 request state：

```python
@dataclass
class ArtifactRequestState:
    required_fields: frozenset[str]
    field_futures: dict[str, Future]
    field_refs: dict[str, list[str]]
    terminal_seen: bool
```

典型 rollout profile：

```text
required:
  routed_experts
  behavior_logprobs       # 算法要求时
  multimodal_manifest     # MM request
  multimodal_payload      # snapshot mode
```

finalize：

```text
scheduler observes terminal
    -> finish R3 tail
    -> finish behavior-logprobs chunk
    -> await MM admission writes
    -> validate all required refs
    -> publish bundle/ready marker
    -> return production handle
```

任一 required field 失败：

- 不发布 ready marker；
- 不返回部分 bundle；
- RL rollout mode fail closed；
- 已发布 immutable objects 作为 orphan 由 TTL 清理；
- 是否仍返回普通文本结果由 deployment policy 决定。

## 9. Bundle 与生产传递

建议 terminal bundle：

```json
{
  "schema_version": 1,
  "sample_id": "...",
  "request_id": "...",
  "request_attempt_id": 0,
  "policy_version": "...",
  "fields": {
    "routed_experts": [
      "vllm-artifact/.../block/...",
      "vllm-artifact/.../tail/..."
    ],
    "behavior_logprobs": [
      "vllm-artifact/.../chunk/0"
    ],
    "multimodal_manifest": [
      "vllm-artifact/.../request/..."
    ]
  }
}
```

写入顺序：

```text
put all payload objects
    -> verify all required puts
    -> put bundle/ready marker last
```

- SHM/simple mode 可以 materialize actual values；
- Mooncake consumer 先读 bundle，再 batch-get refs；
- TransferQueue request row 可以承担 bundle/ready marker；
- TQ 不重复复制 payload；
- Artifact Connector 是唯一 publish 路径。

V3 中“无需 request manifest”和 TQ 调研中的“sample manifest”可以这样统一：

- field materializer 只需要 ordered refs，不依赖 manifest；
- bundle 是跨字段 completion/ready marker 和 sample index；
- bundle 不重复保存 tensor payload。

## 10. Prefix readiness

参与 prefix hit：

```text
KV
+ mandatory PREFIX_BLOCK fields
  - routed_experts
  - prompt_topk_logprobs（仅显式启用时）
```

不参与 prefix hit：

```text
behavior_logprobs
multimodal_manifest
multimodal_payload
```

MM 物理 content dedup 不应改变 scheduler 的 prefix readiness。

## 11. 配置建议

```yaml
aux_output:
  enable_return_routed_experts: true
  enable_behavior_logprobs: true

artifact:
  backend: mooncake

  rollout_profile:
    routed_experts:
      required: true

    behavior_logprobs:
      required: true
      semantics: raw_logprobs
      payload: selected_only
      chunk_tokens: 1024

    multimodal:
      manifest_required: true
      payload_mode: reference_only
      payload_required: false
      max_item_bytes: 67108864
```

动态、不可重建输入：

```yaml
multimodal:
  payload_mode: processed_tensors
  payload_required: true
```

配置字段名称仍是设计建议，实施前应与当前 `AuxOutputConfig`、ArtifactConfig 的实际
代码状态对齐。

## 12. 推荐实施顺序

### PR 1：抽取 Artifact contract，不改变 R3 行为

- 将 key/store/envelope/receipt 从 R3 capture 中抽离；
- 建立 `ArtifactStore` opaque object protocol；
- 建立 field profile 和 namespace；
- 为当前 R3 增加 envelope/checksum；
- 保持现有 R3 API 和测试结果不变。

### PR 2：Behavior logprobs

- 增加 selected-only capture；
- 复用 `AsyncOutput` 已有 D2H；
- accepted-token slicing；
- request chunk/finalize；
- policy/sampling profile；
- 不依赖 HTTP `logprobs` 参数。

### PR 3：MM manifest

- EngineCore admission capture；
- item order、identity、position、processor profile；
- external source refs；
- terminal barrier；
- 暂不存大 payload。

### PR 4：可选 MM payload

- raw/processed mode；
- tensor-tree codec；
- content-hash dedup；
- object size limit 和 backpressure。

### PR 5：生产 bundle/backend

- Mooncake/TQ publish；
- ready marker；
- consumer batch materializer；
- crash/restart/orphan cleanup。

## 13. 测试与验收清单

### 13.1 Behavior logprobs

- normal decode accepted-token equality；
- speculative accepted/rejected；
- frontend stop trim；
- async scheduling；
- preemption/resume 无重复；
- batch 和 `n > 1` request 隔离；
- raw/processed profile 隔离；
- policy version mismatch；
- missing/corrupt chunk fail closed；
- cache-off 与 Artifact-on output equality；
- selected-only 性能开销。

### 13.2 Multimodal manifest

- image/audio/video/vision_chunk/prompt_embeds；
- 多 item 顺序按 prompt position；
- `offset/length/is_embed` roundtrip；
- `mm_hash` 与 `identifier` 保留；
- receiver-cache `data=None` 恢复后 capture；
- prefix stripping 不丢 Artifact payload；
- external reference-only mode；
- abort 不发布 ready bundle。

### 13.3 Multimodal payload

- Tensor/list/tuple/nested-list codec；
- batched/flat/shared layout；
- keep-on-CPU metadata；
- variable dtype/shape/empty tensor；
- checksum/profile mismatch；
- same `mm_hash` dedup；
- processor profile 隔离；
- max item size/backpressure；
- 禁止 pickle；
- payload orphan TTL cleanup。

### 13.4 组合原子性

- R3 + behavior logprobs；
- R3 + behavior logprobs + MM manifest；
- 任一 mandatory field 失败时不发布 bundle；
- worker crash/core writer crash；
- bundle 可见前所有 refs 已 ready；
- consumer 不会观察 partial sample。

## 14. 已知设计风险和待决问题

后续 agent 开工前需要确认：

1. **Artifact Connector 当前实际分支状态**  
   本文调研的 `main` 只有 `aux_output_connector`。如果 Artifact Connector 在其他
   commit/branch，需要先对比接口再落代码。

2. **行为 logprob 的数学语义**  
   RL consumer 需要 raw model logprob，还是经过 temperature/top-k/top-p 后的真实
   behavior distribution？必须由训练算法确认并进入 profile。

3. **selected-only 计算成本**  
   不返回 top-k 时，当前 sampler 是否能低成本产生 selected-token logprob，需要原型
   和 benchmark。

4. **sample ID 来源**  
   request ID 是否足够，还是 rollout controller 需要传入稳定 `artifact_sample_id`？建议
   sample ID 属 control plane，不进入 content block identity。

5. **MM source reference 的入口协议**  
   当前 `MultiModalFeatureSpec` 不保留 dataset URI/sample ID。reference-only mode 需要
   frontend/controller 提供稳定 source ref，不能只依赖 `mm_hash` 反查原始媒体。

6. **ArtifactStore variable-length 支持**  
   当前 mmap store 是 fixed-size slot。可以选择：
   - generic variable-size store；或
   - 每个 fixed-size token field 一个 store，加独立 variable-size request store。

7. **Bundle 是否持久化**  
   direct Mooncake 模式需要小型 ready marker 供独立 consumer 判断原子完成；这不是
   payload manifest，但需要明确 backend contract。

8. **MM processed payload 是否需要可重新注入 vLLM**  
   若只供 trainer 读取，保存 tensor tree 即可；若要求重新构造
   `MultiModalKwargsItem`，还需稳定编码 layout descriptor。

## 15. 下一位 agent 的起步清单

1. 阅读仓库根目录 `AGENTS.md` 和相关 domain guide。
2. 检查工作树和 Artifact Connector 所在 commit/branch。
3. 阅读：
   - `vllm/distributed/aux_output_connector/connector.py`
   - `vllm/distributed/aux_output_connector/worker.py`
   - `vllm/distributed/aux_output_connector/routed_experts.py`
   - `vllm/distributed/aux_output_connector/store.py`
   - `vllm/v1/worker/gpu/async_utils.py`
   - `vllm/v1/outputs.py`
   - `vllm/v1/engine/core.py`
   - `vllm/v1/engine/input_processor.py`
   - `vllm/multimodal/inputs.py`
4. 先画出当前 Artifact Connector 的实际接口，不要直接从本文假设类名。
5. 优先实现 PR 1 的职责解耦，保持 R3 行为不变。
6. 在提出 PR 前按仓库规则检查重复 issue/PR。
7. 新测试先回答模块目的、I/O contract、要防的 failure 和最低成本测试层级。

## 16. 非目标

当前设计不包含：

- full-vocab logits 持久化；
- 通用 Python object dump；
- pickle codec；
- 训练侧 optimizer/checkpoint 状态；
- MM encoder gradient replay；
- 用 Artifact Connector 替换现有 EncoderCache/EC Connector；
- 未验证 topology 的直接解禁；
- 多字段 payload 合并为一个大 object。

实现中应保持 per-field immutable objects 和 request-level completion barrier，避免
field codec、生命周期和失败语义互相耦合。
