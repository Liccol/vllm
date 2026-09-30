# vLLM 与 vLLM-Ascend 多模态数据及 Request-wise 语义

更新日期：2026-09-30  
vLLM 代码基线：`091b14f7b5c0968a6543c53992671812dbf263ad`  
vLLM-Ascend 代码基线：`fd815467c221ee600137f6bdd53fe354d5e7c999`

## 1. 文档目的

本文整理当前 vLLM 与 vLLM-Ascend 代码中的多模态数据模型，并解释 Prefix
Artifact 设计中“多模态数据是 request-wise”的含义。重点回答：

- 多模态请求从入口到模型执行会出现哪些数据；
- 哪些数据表达训练样本语义，哪些只是推理缓存或调度状态；
- vLLM-Ascend 是否定义了不同的多模态协议；
- RL/OPD 训推交接时应保存或传递什么；
- `REQUEST_ONLY` 与按 prefix block 保存的 artifact 有什么区别。

本文记录当前代码事实和 Artifact 侧建议，不表示多模态 Artifact 已经实现。

相关文档：

- [Artifact Connector：Logprobs 与多模态 Artifact 设计及交接](./Artifact%20Connector%20Logprobs与多模态设计及交接.md)
- [Prefix Execution Artifact Store：可落地设计 V3](./Prefix%20Execution%20Artifact%20Store：可落地设计%20V3.md)
- [Prefix Artifact 功能与并行兼容性](./Prefix%20Artifact%20功能与并行兼容性.md)

## 2. 核心结论

vLLM 中的多模态数据不是单个固定 tensor，而是一个 request 内的有序媒体 items、
处理后字段以及它们与 prompt token 的对应关系：

```text
request
├── final prompt token IDs
├── ordered multimodal items
│   ├── modality
│   ├── processed tensor fields
│   ├── mm_hash / identifier
│   └── placeholder position
└── optional encoder outputs/cache state
```

这里需要严格区分：

1. **样本数据**：媒体来源或 processed tensors、prompt token IDs、item 顺序、
   placeholder 和 processor identity。训练与推理需要对齐。
2. **推理派生值**：encoder outputs。只在明确的冻结 encoder 复用场景下考虑保存。
3. **运行时状态**：缓存 slot、调度预算、引用计数和搬运状态。不属于训练样本。

“多模态数据是 request-wise”表示它的**逻辑所有权、顺序和完成语义属于一次
request/sample**。它不表示实际媒体 payload 不能通过 `mm_hash` 跨 request 去重或缓存。

## 3. vLLM 多模态数据的四个层级

### 3.1 请求入口：原始媒体与处理配置

入口参数定义在 `vllm/inputs/llm.py` 的 `_PromptOptions`：

|参数|含义|Artifact 关注点|
|---|---|---|
|`multi_modal_data`|图片、音频、视频或预计算 embedding 等原始输入|保存稳定 source reference，或在必要时保存 raw payload|
|`mm_processor_kwargs`|请求级 processor 参数|应进入 processor profile 或可复现配置|
|`multi_modal_uuids`|调用方提供的媒体 identity|可作为缓存/关联线索，但不能自动证明内容一致|
|`prompt` / `prompt_token_ids`|文本或 token 输入|Artifact 最终关心 processor 展开后的 prompt token IDs|

常见外部数据形态包括：

- 图片：URL、encoded bytes、PIL image、image tensor、image embeddings；
- 视频：URL、视频 bytes、帧序列、video tensor、video embeddings；
- 音频：URL、waveform 与 sample rate、audio features、audio embeddings；
- 模型特有输入，例如 vision chunks、document pages 或 prompt embeddings。

这些类型是入口协议，不应直接作为通用 Artifact codec 的固定枚举。模型和 processor
会继续扩展可接受的数据形态。

### 3.2 Processor 输出：模型可消费的数据

Processor 输出在 `vllm/inputs/engine.py` 中表示为：

```python
MultiModalInput(
    mm_kwargs=...,
    mm_hashes=...,
    mm_placeholders=...,
)
```

各字段含义如下：

|字段|含义|
|---|---|
|`mm_kwargs`|按 modality 和 item 组织的 `MultiModalKwargsItem`|
|`mm_hashes`|每个 item 的 processor/input identity|
|`mm_placeholders`|每个 item 在最终 prompt token 序列中的位置|

`MultiModalKwargsItem` 是 `field name -> MultiModalFieldElem` 的映射。每个 field 除了
tensor 或 tensor tree，还带有布局信息，例如：

- `batched`：按 batch 第一维拆分 item；
- `flat`：按指定维度和 slice 拆分 item；
- `shared`：多个 item 共享同一个值；
- `keep_on_cpu`：该字段执行时仍需保留在 CPU；
- concat dimension：适用于需要拼接或拆分的字段。

因此不能将“多模态 payload”简化成 `pixel_values`。通用实现必须遍历实际
`MultiModalKwargsItem` 的全部字段。

### 3.3 EngineCore：按 prompt 位置排序的 `mm_features`

InputProcessor 将 `mm_kwargs`、`mm_hashes` 和 `mm_placeholders` 按 prompt position
排序，生成：

```python
MultiModalFeatureSpec(
    data=processed_item,
    modality=modality,
    identifier=encoder_cache_identity,
    mm_position=placeholder_range,
    mm_hash=processor_cache_identity,
)
```

这里的字段可以分为：

|字段|用途|训推交接结论|
|---|---|---|
|`data`|单个 item 的 processed model inputs|无法确定性重建时必须保存|
|`modality`|image/audio/video 等类型|必须保存|
|`mm_position.offset`|item 对应 token 区间起点|必须保存|
|`mm_position.length`|item 对应 token 区间长度|必须保存|
|`mm_position.is_embed`|区间内哪些位置实际注入 embedding|存在时必须保存或确定性重建|
|`mm_hash`|processor/input identity，不含 LoRA 前缀|建议保存，用于校验、缓存和内容去重|
|`identifier`|encoder cache identity，可能包含 LoRA identity|建议保存，用于追踪推理时的 encoder identity|

`mm_hash` 和 `identifier` 是 identity，不是训练 forward 的 tensor 参数；但二者对于
诊断错误复用、processor mismatch 和 LoRA/weight version mismatch 很重要。

### 3.4 模型执行：具体 field 随模型变化

以下是当前代码中的典型字段，不是完整或封闭枚举：

|模态/模型|processed fields|语义|
|---|---|---|
|Qwen2.5-VL image|`pixel_values`, `image_grid_thw`|视觉 patches 与时间/高/宽网格|
|Qwen2.5-VL image embeds|`image_embeds`, `image_grid_thw`|调用方直接提供的视觉 embedding 及布局|
|Qwen2.5-VL video|`pixel_values_videos`, `video_grid_thw`, `second_per_grid_ts`|视频 patches、网格和时间间隔|
|Qwen2.5-VL video embeds|`video_embeds`, `video_grid_thw`, `second_per_grid_ts`|预计算视频 embedding 及布局|
|Qwen2-Audio|`input_features`, `feature_attention_mask`, `audio_num_tokens`|音频特征、有效区间和 placeholder token 数|
|Qwen2-Audio embeds|`audio_embeds`, `audio_num_tokens`|预计算音频 embedding 与 placeholder 长度|

`image_grid_thw`、`video_grid_thw`、`second_per_grid_ts` 等字段会参与 MRoPE 或
媒体 token 布局计算。只保存主要 payload 而遗漏这些伴随字段，会使训练与 rollout
看到不同的位置语义。

## 4. 当前 scale-out wire format

当前 vLLM 的 token-in/token-out scale-out 接口已经定义了明确的多模态传输对象：

```python
class MultiModalFeatures(BaseModel):
    mm_hashes: dict[str, list[str]]
    mm_placeholders: dict[str, list[PlaceholderRangeInfo]]
    kwargs_data: dict[str, list[str | None]] | None
    mm_metadata: dict[str, list[str | None]] | None
```

四个字段按 modality 和 item index 平行排列：

- `mm_hashes`：item identity/cache key；
- `mm_placeholders`：item 在 token 序列中的 `offset/length`；
- `kwargs_data`：使用 MsgPack 并 base64 编码的完整 `MultiModalKwargsItem`；
- `mm_metadata`：只包含 embedding metadata 和 `keep_on_cpu` fields 的子集。

`kwargs_data[modality][i] is None` 表示该 item 可从接收侧 cache 恢复。整个
`kwargs_data` 也可在 metadata-only cache-hit 场景下为空。

当请求同时带有 `ec_transfer_params` 时，encoder embeddings 通过 EC Connector 传给
prefill/decoder；HTTP payload 只需携带 `mm_metadata`。没有 EC transfer 时，不能仅凭
metadata 完成模型执行，仍需要完整 `kwargs_data` 或本地 cache hit。

这套协议说明：

```text
processed payload      -> kwargs_data 或 processor cache
encoder embeddings     -> EC Connector 或 encoder cache
token/media alignment  -> mm_placeholders + mm_metadata
identity               -> mm_hashes
```

Artifact Connector 可以复用这些数据边界，但持久化 object 仍需要独立的 schema
version、profile namespace、checksum、完成屏障和 orphan cleanup。

## 5. vLLM-Ascend 的数据模型

vLLM-Ascend 是 vLLM 的硬件插件。当前代码没有定义一套独立的多模态样本协议，而是
复用上游的 `Request.mm_features`、模型接口、EncoderCache 和 EC Connector contract。

Ascend 侧主要补充以下执行能力：

1. **NPU 上的 encoder output cache**  
   `ScoreEncoderCacheManager` 用 `request.mm_features[input_id].identifier` 作为 key，
   维护 request 引用、CPU cache、NPU cache、promotion 和 eviction。

2. **CPU/NPU 分层缓存和异步搬运**  
   `model_runner_v1.py` 管理 `encoder_cache`、`cpu_encoder_cache`、临时 NPU cache
   以及 CPU 到 NPU 的 pending copies。

3. **MRoPE 位置准备**  
   310P 路径调用：

   ```python
   model.get_mrope_input_positions(prefill_token_ids, mm_features)
   ```

   这说明 Ascend 执行同样依赖 prompt token IDs、item position、grid/time metadata。

4. **分离式 Encoder/Prefill/Decode**  
   encoder 实例可以处理 image/audio items，encoder outputs 通过 EC 通路送往
   prefill/decoder，原请求仍提供媒体顺序和 prompt 上下文。

这些 CPU/NPU cache、promotion、eviction 和 pending copy 字段是运行时实现细节，
不应进入 RL 样本或通用多模态 Artifact。

## 6. Request-wise 的准确含义

### 6.1 它描述逻辑坐标和生命周期

`REQUEST_ONLY` 表示一个 artifact 的解释需要 request/sample 上下文：

- 哪些媒体 items 属于该 request；
- item 的顺序；
- item 对应 prompt 的哪段 token；
- 使用了哪个 processor/profile；
- 该 request 最终使用了哪些 payload refs；
- request abort 时是否允许发布最终 handle。

因此多模态 manifest 的自然坐标是 `REQUEST`，每个 payload ref 的自然坐标是
`REQUEST_ITEM`。

### 6.2 它不表示禁止跨 request 物理去重

同一图片可以在多个 request 中出现。推荐模型是：

```text
request A manifest ─┐
                    ├──> immutable payload object keyed by content/profile
request B manifest ─┘
```

两个 request 各自保存 item 顺序和 placeholder，但可以引用同一个按 `mm_hash` 与
processor profile 定位的 payload object。

### 6.3 它不参与 KV prefix readiness

多模态 payload 命中不等于 KV prefix 命中：

- 相同图片可以出现在不同 prompt 位置；
- 相同文本 prefix 可以绑定不同图片；
- 多 item 的顺序会改变模型条件；
- processor/model/LoRA version 会改变 processed inputs 或 encoder outputs。

因此 prefix readiness 应只包含 KV 与 mandatory `PREFIX_BLOCK` fields。MM manifest
和 MM payload 的完成状态由 request-level completion barrier 管理。

### 6.4 与 token-wise/block-wise 字段对比

|维度|`PREFIX_BLOCK`|`REQUEST_ONLY`|
|---|---|---|
|典型字段|R3、显式启用的 prompt logprobs|MM manifest、MM payload、behavior logprobs|
|逻辑坐标|executed/predicted token block|request、accepted output 或 request item|
|key 主体|model/profile + KV block identity|request/sample + item/chunk identity|
|跨 request 逻辑复用|允许，需满足 joint readiness|不允许直接复用 manifest；payload 可内容去重|
|完成条件|对应 prefix blocks ready|request 所有 required fields ready|
|失败语义|该 prefix 不可命中|不发布 terminal bundle/production handle|

## 7. RL/OPD 训推交接需要什么

对于多模态 rollout，teacher forward 和 student training forward 必须在同一个条件状态
上计算。两端 tensor 的 batch padding 或物理布局可以不同，但以下语义必须一致：

```text
final prompt token IDs
+ ordered media items
+ processed input semantics
+ item-to-token placeholder mapping
+ position/MRoPE semantics
+ processor/tokenizer/template identity
```

### 7.1 推荐的默认 manifest

```json
{
  "schema_version": 1,
  "field": "multimodal_manifest",
  "sample_id": "...",
  "request_id": "...",
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

### 7.2 Payload 策略

|模式|保存内容|适用场景|
|---|---|---|
|`reference_only`|manifest + 稳定 dataset/object-store ref|媒体稳定且 processor 可确定性复现|
|`raw_media`|encoded media bytes + decode/processor profile|临时 URL、用户上传媒体、训练需重新运行可训练 encoder|
|`processed_tensors`|`MultiModalFeatureSpec.data` 的完整 tensor tree|随机抽帧/crop/augmentation 或调用方直接传 embedding|

不要默认同时保存 raw media、processed tensors 和 encoder outputs。应选择满足训练重建
要求的最小层级。

### 7.3 默认不保存

- `scheduled_encoder_inputs`；
- encoder compute/cache budget；
- `free_encoder_mm_hashes`；
- CPU/NPU/GPU cache slots；
- cache promotion/eviction/refcount；
- pending copies 和 stream/event 状态；
- `finished_sending` / `finished_recving`；
- device、pinned memory 等物理放置状态；
- prefix stripping 后用于节省 IPC 的 `data=None`；
- 默认情况下的 multimodal encoder outputs。

encoder 可训练时，训练侧必须从 raw media 或 processed inputs 重新 forward，以建立梯度
图。复用 rollout 中 detached encoder output 会丢失 encoder 梯度。

## 8. Capture 时机与完成屏障

推荐在 EngineCore 恢复完整 `mm_features` 后捕获：

```text
mm_receiver_cache.get_and_update_features()
    -> capture manifest/payload
    -> Request/Scheduler
    -> prefix-covered data stripping
```

原因是 IPC/cache 优化可能让入站 `MultiModalFeatureSpec.data` 暂时为 `None`；恢复 cache
后才能得到完整 processed item。若等到 worker 或 prefix stripping 之后再捕获，可能只剩
metadata 或 `data=None`。

终止时的顺序应为：

```text
finish MM admission writes
    -> validate manifest and required payload refs
    -> wait for other required rollout fields
    -> publish terminal bundle/ready marker last
```

若 required MM field 失败，不应发布部分完成的 production handle。已写入的 immutable
payload object 可作为 orphan 由 TTL 清理。

## 9. Artifact 与当前 scale-out 协议的映射

|当前 vLLM 字段|Artifact 表达|备注|
|---|---|---|
|`prompt_token_ids`|manifest 中的值或稳定 ref|必须与 placeholder 坐标一致|
|`mm_hashes` / `mm_hash`|item identity、payload content key 的组成部分|必须加入 processor profile namespace|
|`mm_placeholders` / `mm_position`|manifest item position|`is_embed` 也需保留|
|`kwargs_data` / `MultiModalFeatureSpec.data`|processed payload object|完整 tensor tree，不固定 field name|
|`mm_metadata`|processed payload 中的 metadata fields|不能与 item position 脱离|
|`identifier`|manifest identity|可能包含 LoRA identity|
|`ec_transfer_params`|不直接持久化；可作为外部传输 receipt|属于执行传输协议|
|encoder outputs|默认不属于本 Artifact profile|若启用需独立 model/weight/LoRA/dtype profile|

当前 scale-out 的 MsgPack/base64 格式用于 HTTP 传递。Artifact backend 的持久化编码可以
借鉴其 tensor-tree 覆盖范围，但还需要 canonical header、schema version、checksum、
object size/backpressure 和 profile mismatch 校验。

## 10. 关键设计决策

### ADR-MM-1：MM manifest 使用 request scope

- **决定**：`multimodal_manifest` 使用 `REQUEST_ONLY / REQUEST`。
- **原因**：item 顺序和 placeholder mapping 只有在具体 prompt/request 中才完整。
- **代价**：每个 request 都需要一个小型 manifest。
- **收益**：避免将不同媒体条件错误地绑定到同一个文本/KV prefix。

### ADR-MM-2：Payload 使用 request item 逻辑引用和内容寻址物理对象

- **决定**：manifest 按 item 顺序引用 payload；payload 可由 content/profile key 去重。
- **原因**：兼顾 request 语义与大媒体对象的跨请求复用。
- **风险**：错误的 processor profile 会造成错误 dedup。
- **缓解**：key namespace 必须包含 processor schema/config identity 和 checksum。

### ADR-MM-3：不将 encoder outputs 混入 processed-input profile

- **决定**：默认不存 encoder outputs；若未来启用，建立独立 field/profile。
- **原因**：encoder outputs 与 model weights、LoRA、dtype 和实现版本强绑定。
- **风险**：权重更新后读到 stale embeddings。
- **缓解**：独立 profile 必须包含 model/weight generation，且更新时失效。

### ADR-MM-4：vLLM 与 vLLM-Ascend 共用逻辑 schema

- **决定**：Artifact schema 基于 vLLM 公共多模态对象，不引入 Ascend 专属样本格式。
- **原因**：Ascend 当前复用上游的 request、model 与 EC contracts。
- **收益**：trainer、CUDA rollout 和 Ascend rollout 使用同一个消费协议。
- **边界**：NPU cache/promotion/event 等状态留在 vLLM-Ascend 内部。

## 11. 风险和待决问题

1. **稳定 sample ID 来源**  
   `request_id` 是否能作为训练样本 identity，还是 rollout controller 必须显式传入
   `artifact_sample_id`。

2. **Source reference 入口**  
   当前 `MultiModalFeatureSpec` 不保留 dataset URI/sample ID。`reference_only` 需要
   frontend/controller 提供稳定 source ref，不能只依赖 `mm_hash` 反查原始媒体。

3. **Processor profile 定义**  
   需要明确包含哪些 tokenizer/template、HF processor、decode、resize、crop、sampling
   和模型特有 kwargs。仅保存用户传入的 `mm_processor_kwargs` 可能不足。

4. **可重新注入范围**  
   若 payload 只供 trainer 读取，保存 tensor tree 即可；若要求重新构造 vLLM 的
   `MultiModalKwargsItem`，还需要稳定保存 layout descriptor。

5. **动态多轮多模态输入**  
   agent/tool 在 rollout 中产生的新媒体必须进入该轮对应的 ordered item manifest，
   不能只保留初始 dataset 的媒体列表。

6. **Weight/LoRA cache coherence**  
   encoder output cache key 必须随权重或 adapter generation 变化，或在更新完成时明确
   invalidation。`mm_hash` 只描述媒体/processor identity，不能单独保证 embedding 新鲜。

## 12. 相关 upstream RFC 与实现

- [RFC #4194：Multi-modality Support on vLLM](https://github.com/vllm-project/vllm/issues/4194)
- [RFC #19702：Multimodal data IPC improvement](https://github.com/vllm-project/vllm/issues/19702)
- [RFC #21113：Reuse multimodal embeddings from encoder cache](https://github.com/vllm-project/vllm/issues/21113)
- [RFC #22044：Optimize Input Media Processing in vLLM](https://github.com/vllm-project/vllm/issues/22044)
- [RFC #12761：Cross-attention multimodal models in V1](https://github.com/vllm-project/vllm/issues/12761)
- `vllm/inputs/llm.py`
- `vllm/inputs/engine.py`
- `vllm/multimodal/inputs.py`
- `vllm/v1/engine/input_processor.py`
- `vllm/v1/engine/core.py`
- `vllm/entrypoints/scale_out/token_in_token_out/protocol.py`
- `vllm/entrypoints/scale_out/token_in_token_out/mm_features.py`
- `vllm/entrypoints/scale_out/token_in_token_out/mm_serde.py`
- `vllm-ascend/vllm_ascend/ec_manager/score_ec_manager.py`
- `vllm-ascend/vllm_ascend/worker/model_runner_v1.py`
- `vllm-ascend/vllm_ascend/_310p/worker/v2/rope.py`

