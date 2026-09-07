# TransferQueue / Mooncake 接口调研

# 



日期：2026\-07\-25



本文只记录当前 pinned 源码的接口事实，以及它们如何映射到 Prefix Artifact

生产 backend。总体架构见

\[V3 设计\]\(\./prefix\-execution\-artifact\-store\-design\-v3\.md\)。



## 1\. 调研版本



|项目|Revision|本地路径|
|---|---|---|
|TransferQueue|`b75d570d88c50bbfcbe2171baa727fadd7216f76`|`reference/TransferQueue`|
|Mooncake|`ac010838926e3cff2659465bd9b7bf6c0e9656bf`|`reference/Mooncake`|



开始生产实现前必须重新 pin 最新 revision；本文不能替代届时的接口复核。



关键源码：



- \[TQ high\-level interface\]\(\.\./\.\./reference/TransferQueue/transfer\_queue/interface\.py\)

- \[TQ metadata\]\(\.\./\.\./reference/TransferQueue/transfer\_queue/metadata\.py\)

- \[TQ KV storage manager\]\(\.\./\.\./reference/TransferQueue/transfer\_queue/storage/managers/base\.py\)

- \[TQ Mooncake manager\]\(\.\./\.\./reference/TransferQueue/transfer\_queue/storage/managers/mooncake\_manager\.py\)

- \[TQ Mooncake client\]\(\.\./\.\./reference/TransferQueue/transfer\_queue/storage/clients/mooncake\_client\.py\)

- \[TQ StreamingDataset\]\(\.\./\.\./reference/TransferQueue/transfer\_queue/dataloader/streaming\_dataset\.py\)

- \[TQ StreamingDataLoader\]\(\.\./\.\./reference/TransferQueue/transfer\_queue/dataloader/streaming\_dataloader\.py\)

- \[Mooncake structured object API\]\(\.\./\.\./reference/Mooncake/mooncake\-wheel/mooncake/structured\_object\_store\.py\)

    

## 2\. 实际分层



```Plain Text
ArtifactConnector adapter
        │
        │ tq.kv_put / kv_batch_put
        │ async_kv_put / async_kv_batch_put
        ▼
TransferQueueClient
        ├─ Controller metadata / sampler / status
        └─ StorageManager
              │
              ▼
       MooncakeStorageManager
              │
              ▼
       MooncakeStoreClient
              │
              ▼
       MooncakeDistributedStore
```



TQ 同时承担：



- user key 到 sample/global index 的映射；

- partition；

- field schema；

- per\-field production status；

- custom metadata；

- sampler/consumption tracking；

- StreamingDataset/DataLoader；

- storage backend 调用。

    

Mooncake 只承担 payload 的分布式 KV 存储。



## 3\. High\-level KV API



当前 TQ 已经提供 user\-defined key API：



```Python
kv_put(
    key: str,
    partition_id: str,
    fields: TensorDict | dict[str, Any] | None,
    tag: dict[str, Any] | None,
) -> KVBatchMeta

kv_batch_put(
    keys: list[str],
    partition_id: str,
    fields: TensorDict | None,
    tags: list[dict[str, Any]] | None,
) -> KVBatchMeta

kv_batch_get(
    keys: list[str] | str,
    partition_id: str,
    select_fields: list[str] | str | None,
) -> TensorDict
```



也有对应 async API：



```Plain Text
async_kv_put
async_kv_batch_put
async_kv_batch_get
async_kv_list
async_kv_clear
```



因此 Artifact Connector 可以直接使用自己的 logical key，不需要把 TQ

`global_index` 暴露给 vLLM API 或 VIME。



## 4\. Logical key 与 physical key



`kv_put(key=...)` 的流程：



```Plain Text
user key
  -> controller kv_retrieve_meta(create=True)
  -> TQ global_index
  -> StorageManager physical field keys
```



Mooncake physical key 由 `KVStorageManager._generate_keys()` 生成：



```Plain Text
global_index@field_name
```



例如：



```Plain Text
42@routed_experts
42@prompt_logprobs
42@manifest
```



结论：



- Artifact Connector 控制 TQ logical key；

- TQ 控制 `global_index`；

- TQ Mooncake backend 控制 physical key；

- 不需要修改 Mooncake，让它直接接受 artifact block hash 作为 physical key；

- `global_index` 不进入 HTTP/VIME contract。

    

## 5\. Put 与 ready 顺序



TQ storage manager 的顺序是：



```Plain Text
generate physical keys/values
  -> storage_client.put(...)
  -> 收集 per-field backend metadata
  -> notify_data_update(...)
  -> controller 更新 production_status
```



`notify_data_update` 等待 controller ACK。Controller 对 sample/field 维护

production status；`BatchMeta` 只有在请求的 fields 全部 produced 时才是 ready。



这已经提供本设计需要的基本 publish fence：



```Plain Text
payload backend put 完成
  -> requested field status ready
  -> dataloader 可选中 sample
```



首版不需要额外设计

`PROCESSING -> COMPLETE -> FAILED -> ACKED` 状态机。



## 6\. 三种读取状态



对 Artifact Connector 来说可以归一为：



```Plain Text
NOT_FOUND
NOT_READY
READY
```



映射：



- TQ logical key/row 不存在：`NOT_FOUND`；

- row 存在，但 requested fields 的 `production_status` 不是全 1：`NOT_READY`；

- requested fields 全 ready 且 backend get 成功：`READY`。

    

Mooncake 底层还可能返回：



- `OBJECT_NOT_FOUND`；

- `REPLICA_IS_NOT_READY`；

- 成功；

- 其他永久或暂时错误。

    

这些错误应由 TQ Mooncake client/adapter 做 retry 和归一化。业务层不需要再发明一套

artifact control state；永久错误必须让对应 request/sample fail closed。



## 7\. Mooncake backend 的实际接口



TQ 当前没有通过 `MooncakeBundleTransfer` structured\-object API 写数据，而是使用：



```Plain Text
MooncakeStoreClient
  -> MooncakeDistributedStore
  -> batch_upsert_from(...)
```



所以旧设计中围绕 structured bundle、随机 object ID、PutStart/PutEnd 的讨论不能

直接套到当前 TQ backend。



### 7\.1 Tensor path



CPU tensor path：



```Plain Text
Tensor values
  -> contiguous/merge regions
  -> register buffers
  -> batch_upsert_from
  -> unregister buffers
```



特征：



- field\-major batch；

- batch limit；

- 多线程 worker；

- per\-key retry；

- 调用完成后 source buffer 可释放。

    

### 7\.2 Non\-tensor path



非 tensor 值：



```Plain Text
serialize into uint8 buffers
  -> register
  -> upsert
  -> record packed_size in custom backend metadata
```



Request manifest 可以作为非 tensor field 保存，但大规模 manifest 的序列化成本要

单独测量。



### 7\.3 GDR path



Mooncake client 支持可选 GDR staging：



- `use_gdr=True` 要求 RDMA；

- persistent GPU staging buffer；

- 普通 tensors pack 到 staging；

- 超大 tensor 拆成 `:c{i}` sub\-keys；

- 当前实现包含 CUDA synchronize/staging synchronization。

    

Prefix Artifact 首版采用 CPU pinned ring 路径。是否启用 GDR 必须比较：



- GPU capture 到 CPU ring 的 D2H；

- GDR staging copy/synchronize；

- GPU memory 占用；

- 推理 TPOT；

- RDMA throughput。

    

不能仅因为少一次 CPU copy 就默认选择 GDR。



### 7\.4 Upsert 语义



当前 TQ Mooncake client 使用 upsert，不是 create\-if\-absent immutable put。



影响：



- 稳定 logical key 的重试可以幂等；

- 多 writer 对同一 field 并发写时可能发生覆盖；

- full block 必须由 deterministic owner 单写；

- 相同 key 的 payload 必须 deterministic；

- concurrent create/upsert 需要 E2E，不能依赖“最后写入者正确”。

    

当前不需要立即修改 Mooncake API；先通过 owner 和 adapter 保证单写。



## 8\. StreamingDataLoader 能力



`StreamingDataset`/`StreamingDataLoader` 已提供：



- partition；

- required `data_fields`；

- batch/micro\-batch；

- sampler 与 DP rank 协调；

- ready sample polling；

- consumption status；

- prefetch buffer；

- reset/step；

- custom `fetch_batch_fn`；

- custom `process_batch_fn`。

    

这正是使用 TQ 而不是直接调用 Mooncake 的主要价值。



## 9\. Artifact 数据映射



为了同时实现 full\-block dedup 和 request/sample dataloader，建议使用两类 logical

row。



### 9\.1 Content block row



```Plain Text
partition = artifact_blocks/<model>/<profile>/<policy_epoch>
key       = stable full-block identity
fields    = routed_experts, optional prompt_logprobs, schema/checksum
```



性质：



- 跨 request 去重；

- deterministic owner put；

- immutable；

- 不直接作为训练 sample。

    

### 9\.2 Request sample row



```Plain Text
partition = artifact_requests/<rollout_or_policy_epoch>
key       = artifact_sample_id
fields    = manifest, tail, request-level metadata
```



Manifest 引用 content block logical keys。



VIME 从 request partition 采样，再通过 custom `fetch_batch_fn`：



```Plain Text
fetch request rows
  -> parse block refs
  -> kv_batch_get content blocks
  -> dedup block reads
  -> assemble actual training batch
```



这能保留 TQ dataloader 抽象，同时不为每个 request 重复存 full\-block payload。



## 10\. “只 put 一次”的准确含义



业务上只有一条 producer 路径：



```Plain Text
ArtifactConnector -> TransferQueue -> Mooncake
```



它不意味着整个 request 永远只有一次 API 调用。一个请求可能：



- 增量 put 新完成的 content blocks；

- finalize 时 put request tail/manifest。

    

这些 put 保存的是不同 logical objects，不是把同一 payload 分别写 TQ 和

Mooncake 两次。



明确禁止：



- vLLM 直接写 Mooncake 后再向 TQ put handle；

- VIME 收到 response 后再把相同 artifact put 到 TQ；

- 两个 writer 无 ownership 地 upsert 同一个 block field。

    

## 11\. `artifact_sample_id`



映射关系：



```Plain Text
vLLM artifact_sample_id
  = TQ request-sample logical key

TQ global_index
  = controller/internal storage identity

Mooncake global_index@field
  = physical storage key
```



`artifact_sample_id`：



- 由 vLLM 生成；

- 在 production response 返回；

- VIME 只记录/消费；

- 简单 SHM mode 返回 actual value 时不需要公开；

- 不由 VIME 预分配；

- 不等于 TQ global index。

    

## 12\. 当前接口能直接满足的要求



无需上游改动即可完成：



- user\-defined logical keys；

- batch put/get；

- tensor/non\-tensor fields；

- partition；

- per\-field ready；

- requested\-fields readiness；

- sampler/StreamingDataLoader；

- custom fetch/materialize；

- Mooncake RDMA/TCP backend；

- retry；

- vLLM 生成 sample ID。

    

## 13\. 必须通过 E2E 确认的缺口



### 13\.1 Content block 与 request sample 的两级读取



确认 custom `fetch_batch_fn` 能：



- 高效批量解析 manifest；

- 跨 samples dedup block refs；

- batch get content fields；

- 保持训练 sample 顺序；

- 正确 checkpoint/consumption。

    

### 13\.2 Multi\-writer concurrent logical key



确认 controller `kv_retrieve_meta(create=True)` 对同 key 并发创建只产生一个稳定

row，以及 owner 策略能避免 concurrent field upsert。



### 13\.3 Readiness



确认：



- content row required fields 未齐时不被读取；

- request row manifest 只在引用 block ready 后发布；

- producer crash 不会留下“manifest ready、block missing”；

- permanent Mooncake error 能让 sample fail closed。

    

### 13\.4 Worker critical path



当前 high\-level put 最终等待 storage put 和 notify。生产 adapter 需要验证 async

API 或独立 writer queue 能把等待移出 model execution critical path，并明确：



- source ring slot 何时可释放；

- queue 满时如何 backpressure；

- shutdown 如何 drain/cancel；

- request terminal response 等待 put 到哪一个边界。

    

## 14\. 暂不提出的上游修改



在上述 E2E 完成前，不提出：



- Mooncake 自定义 physical key API；

- structured\-bundle deterministic object ID；

- 新 TQ artifact state machine；

- VIME publish handle；

- 独立 Mooncake materializer 服务。

    

如果 E2E 失败，issue/PR 必须以具体接口缺口为依据，例如：



- 无法原子创建同 logical key；

- 无法表达 requested\-fields ready；

- custom fetch 无法 checkpoint；

- backend put 没有可用的 source ownership completion。

    

## 15\. 最终接口选择



首个 production adapter 优先使用：



```Plain Text
full blocks:
  async_kv_batch_put(
      keys=block_keys,
      partition_id=artifact_block_partition,
      fields=batched block fields,
  )

request finalize:
  async_kv_put(
      key=artifact_sample_id,
      partition_id=artifact_request_partition,
      fields={manifest, tail, ...},
  )

consumer:
  StreamingDataset(custom fetch_batch_fn)
```



这条路径保留 TQ 的 dataloader 能力，让 Mooncake 继续作为 TQ backend，并满足

Artifact Connector 唯一 producer、payload 不重复写入的最终要求。

