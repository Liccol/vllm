# Prefix Artifact 统一 Core 骨架设计

日期：2026-09-07

状态：**基于 [vllm-project/vllm#45635](https://github.com/vllm-project/vllm/pull/45635) 真实实现校准后的骨架设计**。
head 分支 `xhx1022/vllm:r3_offload`，head SHA `df09778c`。

> 注：本文件标题里的「SHM」沿用旧命名，但 #45635 的实际落地已把 store 从**跨进程文件 SHM**
> （`LocalSharedMemoryArtifactStore`，见更早的 aoshen02/vllm#11）改成**单 owner 进程内匿名 mmap arena**
> （`InProcessArtifactStore`）。真正的 shm / mooncake / kvcc 后端是下一阶段的事，见 §11。
> 本文是 #45635 的权威基准，**取代**之前基于 aoshen02/vllm#11 的版本（那份的 key 公式、store 语义、
> 文件布局均已过时）。

---

## 0. 一句话结论

PR #45635 的「统一 Core」是一个**进程内、KV-cache 生命周期对齐**的 artifact 管线：

```text
scheduler 侧  ArtifactSchedulerConnector   （control plane：增量 block-hash 打包 / emit cursor / generation）
worker 侧     ArtifactWorkerConnector       （data plane：capture / tail buffer / publish / materialize）
field 侧      routed_experts.py             （R3 专用 key + buffer + publish + materialize —— 未来 logprobs 镜像）
backend 侧    store.py                      （opaque bytes + 定长 slot + 匿名 mmap arena + 引用计数）
```

`RoutedExpertsArtifactBuffer` 从独立 `buffer.py` **并入** `routed_experts.py`；
`LocalSharedMemoryArtifactStore`（文件 SHM）**删除**，换成 `InProcessArtifactStore`（匿名 mmap）。
与 aoshen02#11 的 8 处差异见 §8。

---

## 1. 真实文件布局（head SHA `df09778c` 最终态）

```text
vllm/distributed/artifact_connector/
├── __init__.py            # 仅 SPDX 头（空）
├── store.py               # ArtifactObject / ArtifactStoreError / BackgroundArtifactStore / InProcessArtifactStore
├── connector.py           # PackedBlockHashes / ArtifactConnectorMetadata / ArtifactRequestOutput / ArtifactSchedulerConnector
├── routed_experts.py      # _RequestTail / RoutedExpertsArtifactBuffer / routed_experts_keys / publish / materialize
└── worker.py              # _WorkerRequestState / PendingArtifactOutput / ArtifactWorkerConnector

vllm/config/artifact.py    # ArtifactConfig
```

**对比 aoshen02#11**：`shm.py`、`buffer.py` 两个文件消失（职责分别并入 `store.py` 与
`routed_experts.py`）；`store.py` 里不再有 `ArtifactStore` Protocol（只留 `BackgroundArtifactStore`
+ `InProcessArtifactStore`）。

集成改动（非新目录，但必须一起看）：
`vllm/config/vllm.py`（`_verify_artifact_compatibility`）、`vllm/v1/core/sched/output.py`
（`artifact_connector_metadata` 字段）、`scheduler.py`（构造 + 3 个调用点）、
`vllm/v1/worker/gpu/model_runner.py`（构造 + begin_step/prepare_output）、
`vllm/v1/worker/gpu/async_utils.py`（`AsyncOutput` 承载）、
`vllm/model_executor/layers/fused_moe/routed_experts_capturer.py`（capture 源，本 PR 基本不动）。

---

## 2. store.py —— 定长 slot + 匿名 mmap arena + 引用计数

```python
@dataclass(frozen=True)
class ArtifactObject:
    key: str
    payload: bytes

class ArtifactStoreError(RuntimeError): ...

class BackgroundArtifactStore:           # 后台线程序列化 put，读自己写
    def __init__(self, store: InProcessArtifactStore, *, max_pending_batches: int): ...
    def put(self, objects, *, retain_keys=(), release_keys=()) -> None: ...  # 入队，coalesced
    def get_concatenated(self, keys: list[str]) -> bytes: ...                # queue.join() 后读
    def close(self) -> None: ...

class InProcessArtifactStore:            # 单 owner，淘汰后 fail-closed
    def __init__(self, *, max_bytes: int, object_nbytes: int): ...
    def put(self, objects, *, retain_keys=(), release_keys=()) -> None: ...
    def get_concatenated(self, keys: list[str]) -> bytes: ...
    def close(self) -> None: ...
```

要点（**与 aoshen02#11 的 `LocalSharedMemoryArtifactStore` 本质不同**）：

- **匿名 mmap，非文件**：`self._arena = mmap.mmap(-1, num_slots * object_nbytes)`——进程内匿名映射，
  无 `arena.bin` 文件、无 `flock` writer lock、无 `make_shm_store_id(instance_id, dp_rank)`、无 stale GC。
  store 是**单 owner、单进程**的。
- **定长 slot 化**：`object_nbytes` 在构造时固定（= 一个 hash block 的 R3 数据字节数），
  `num_slots = max_bytes // object_nbytes`；`_lru: OrderedDict[str, int]` 记 key→slot
  （`_UNALLOCATED_SLOT = -1` 表示已登记未落盘），`_free_slots` 回收空闲 slot，`_next_slot` 顺序分配。
- **引用计数对齐 KV cache 生命周期**（commit `22d5ec7`「Align artifact lifetime with KV cache」）：
  `_references: dict[str, int]` + `_retain(key)`/`_release(key)`；`_release` 返回
  `terminal_order`（引用归零的 key，移到 LRU 最尾成为下一轮淘汰候选）。这取代了 aoshen02#11 的
  TTL stale GC——artifact 的存活直接绑定到「还有多少 KV block 引用它」。
- **淘汰**：`_evict_to_fit(protected)` 只淘汰「无引用且不在 protected 集合」的最旧对象；
  凑不够就 `raise ArtifactStoreError`（fail-closed）。
- **幂等**：`put` 内 `unique = {obj.key: obj}` 去重，已落盘 key 只 `move_to_end`（touch）不重写；
  未落盘（`_UNALLOCATED_SLOT`）才 `_allocate_slot` + 写 `arena[offset:offset+object_nbytes] = payload`。
- **读**：`get_concatenated(keys)` 把多个 key 的 slot 拼接成一段 bytes（`b"".join(...)`），
  读不到抛 `ArtifactStoreError`（提示增大 `max_bytes`），读后 `move_to_end` touch。
- **后台线程**：`BackgroundArtifactStore` 用 daemon 线程 `"vllm-artifact-writer"` + `queue.Queue`，
  把 `(objects, retain_keys, release_keys)` 三元素 batch 序列化到 store；`get_concatenated` 先
  `queue.join()` 保证「读自己刚发的写」；`_error` 记录首次失败、后续 `put/get` 经
  `_raise_if_failed` fail-closed。

---

## 3. connector.py —— scheduler 控制面

```python
@dataclass
class PackedBlockHashes:            # 连续 bytes 打包，跨 scheduler/worker IPC
    data: bytes
    item_size: int
    def __iter__(self) -> Iterator[bytes]: ...   # 按 item_size 切片

@dataclass
class ArtifactConnectorMetadata:    # scheduler -> worker（挂 SchedulerOutput）
    generation: int
    requests: dict[str, int]                    # request_id -> emit_start
    block_hashes: dict[str, PackedBlockHashes]  # 增量新增 hash
    finished_requests: tuple[str, ...]          # 本步终止的 request_id

@dataclass
class ArtifactRequestOutput:
    token_start: int
    rows: np.ndarray

class ArtifactSchedulerConnector:
    def __init__(self) -> None: ...             # _sent_hash_counts / _finished_requests / _generation
    def build_connector_meta(self, scheduler_output, requests) -> ArtifactConnectorMetadata: ...
    def take_output(self, request, output) -> np.ndarray | None: ...
    def request_finished(self, request) -> None: ...
    def reset(self) -> None: ...
    @staticmethod
    def _pack_new_hashes(block_hashes, num_sent) -> PackedBlockHashes | None: ...
```

关键行为（**与 aoshen02#11 的差异**：`requests` 从 `list[ArtifactRequestMetadata]` 精简为
`dict[str, int]`，`finished_requests` 从 `dict` 精简为 `tuple[str, ...]`）：

- **emit_start 语义**（`build_connector_meta` 内）：
  ```python
  scheduled_requests[request_id] = max(
      request.sampling_params.routed_experts_prompt_start,
      0 if request.num_output_tokens == 0 else request.num_tokens - 1,
  )
  ```
  即「R3 从 `routed_experts_prompt_start`（默认 0）开始 emit，直到最后一个已生成 token」。
- **增量 hash 打包**：`_pack_new_hashes` 只发 `block_hashes[num_sent:]` 的新增段（`_sent_hash_counts`
  记录每 request 已发数量），避免每 step 重传全部 hash；已 settle 但未重调度的 request 会补发
  hash-only 更新（见 `build_connector_meta` 第二段）。
- **termination**：`request_finished` 把 terminal 事件 + 最终新增 hash 塞进 `_finished_requests`，
  下一次 `build_connector_meta` 时作为 `finished_requests` 元组转交给 worker。
- **generation**：`reset()` 清空 `_sent_hash_counts`/`_finished_requests` 并 `_generation += 1`；
  worker 见 generation 变化即清空本地状态（reset invalidation 的控制面实现）。
- **不持 payload、不持 store**：`ArtifactSchedulerConnector.__init__` 无参，纯游标 + hash 记账。

---

## 4. routed_experts.py —— R3 field 模块 + tail buffer + 明文 key

### 4.1 key 公式（**明文，不再 sha256**）

```python
def routed_experts_keys(block_hashes, artifact_namespace) -> list[str]:
    prefix = f"vllm-artifact/{artifact_namespace}/"
    return [prefix + block_hash.hex() for block_hash in block_hashes]
```

`artifact_namespace = str(generation)`。key 是 `vllm-artifact/{generation}/{block_hash.hex()}`，
**直接拼接十六进制 block_hash，无哈希、无 field/dtype/shape**。这是与 aoshen02#11
（`sha256(generation + "\0" + block_hash)`）最直观的差异：更简单，但**更暴露 field 冲突风险**
（见 §9）。

### 4.2 publish / materialize（无 JSON envelope）

```python
def materialize_routed_experts(store, artifact_keys, *, shape_per_token, dtype) -> np.ndarray:
    payload = store.get_concatenated(artifact_keys)
    return np.frombuffer(payload, dtype=dtype).reshape((-1, *shape_per_token))

def publish_routed_experts(store, *, batches, block_size, retain_keys=(), release_keys=()) -> None:
    objects = []
    for artifact_keys, blocks in batches:
        for block_start, array in blocks:
            # 校验：block_start 对齐 block_size；block_index < len(keys)；len(array)==block_size
            objects.append(ArtifactObject(
                key=artifact_keys[block_start // block_size],
                payload=array.tobytes(order="C"),
            ))
    store.put(objects, retain_keys=retain_keys, release_keys=release_keys)
```

payload 是 `array.tobytes(order="C")` 的 raw bytes，**无 header**；shape/dtype 由调用方（worker）传入，
materialize 只靠 `get_concatenated` 的字节数 + `reshape((-1, *shape_per_token))` 做隐式校验。

### 4.3 RoutedExpertsArtifactBuffer（从旧 `buffer.py` 迁入）

`RoutedExpertsArtifactBuffer(dtype, shape_per_token, block_size, max_num_seqs, max_num_batched_tokens, max_concurrent_batches)`：

- `_rows: np.ndarray[(max_blocks, block_size, *shape_per_token)]` 的 numpy 预分配池，
  `max_blocks = max_concurrent_batches * ceil(max_num_batched_tokens/block_size) + max_num_seqs`；
  `_free_slots` / `_owned_slots` / `_requests: dict[Hashable, _RequestTail]` 管理 slot 生命周期。
- `capture(request_id, token_start, rows) -> list[(block_start, block)]`：把 rows 追加进 tail，
  凑满 `block_size` 即产出完整 block；**全对齐输入块**（`local_start==0 and len>=block_size`）不占
  tail pool 直接产出。
- `read` / `retain_block`（无 key 的 pending block 保留到下一次 hash 更新）/ `release_block` /
  `discard` / `reset`。
- **已参数化 dtype/shape_per_token**：类名带 R3，但本质是通用 tail buffer，logprobs 复用只需改名
  `ArtifactBuffer`（见 §9）。

---

## 5. worker.py —— worker 数据面

```python
@dataclass
class _WorkerRequestState:
    artifact_keys: list[str]                 # 已 keyed（有 hash）的 block key 序列
    pending_blocks: list[tuple[int, np.ndarray]]  # 无 key、待 hash 更新的完整 block
    capture_cursor: int | None
    scheduled_cursor: int
    emit_cursor: int

@dataclass
class PendingArtifactOutput:
    connector: ArtifactWorkerConnector
    token_starts: np.ndarray
    query_start_loc: np.ndarray
    routed_experts: torch.Tensor
    finished: Event
    def complete(self) -> None: ...          # connector._pending_output = None; finished.set()

class ArtifactWorkerConnector:
    def __init__(self, *, vllm_config, model, kv_cache_config, max_num_batched_tokens): ...
    def prepare_output(self, request_ids, token_starts, query_start_loc) -> PendingArtifactOutput | None: ...
    def process_output(self, request_ids, token_starts, query_start_loc, routed_experts,
                       num_sampled, num_rejected) -> dict[str, ArtifactRequestOutput]: ...
    def begin_step(self, metadata) -> None: ...
    def close(self) -> None: ...
```

关键设计：

- **TP 对称**：`__init__` 无条件 `RoutedExpertsCapturer` + `bind_routed_experts_capturer`（所有 TP rank
  参与 capture collective），但 `if not get_tp_group().is_first_rank: return`——只有 TP output rank
  建 store/buffer 持有 data plane。
- **block 粒度拆分**（`__init__`）：`scheduler_block_size, hash_block_size =
  resolve_kv_cache_block_sizes(kv_cache_config, vllm_config)`；`hashes_per_kv_block =
  scheduler_block_size // hash_block_size`；`block_nbytes = hash_block_size * prod(shape_per_token) *
  dtype.itemsize`。store 的 `object_nbytes` 以 **hash block** 为粒度（一个 hash 对应一段 R3 rows），
  `max_bytes` 缺省 = `kv_cache_config.num_blocks * hashes_per_kv_block * block_nbytes`（与 KV cache
  同容量）。
- **capture 源**：`prepare_output` 调 `self._capturer.snapshot_routing_data(num_rows)` 拿到**稳定 GPU
  snapshot**（`device_buffer[:num_rows].to(output_dtype)`），交给 `AsyncOutput` 在 copy stream 上做
  异步 D2H。
- **process_output 主流程**：按 `query_start_loc` 切 rows → `rows = routed_experts[start:end-rejected]`
  （spec reject 截断）→ 处理 capture/emit/scheduled 三个 cursor（含「乐观调度的后缀被拒后重贴」分支）
  → `buffer.capture` → `_publish_blocks`（keyed 的 ready 直接 publish，无 key 的进 `pending_blocks`
  retain 等下一次 hash 更新）→ materialize `[emit_start, token_end)` 组装 `ArtifactRequestOutput`。
- **generation 检测**：`begin_step` 见 `metadata.generation > self._generation` 即 release 全部旧 key、
  `buffer.reset()`、`_requests.clear()`——reset invalidation 的 worker 侧实现。

---

## 6. config/artifact.py + 兼容性校验

```python
@config
class ArtifactConfig:
    enable_return_routed_experts: bool = False
    max_bytes: int | None = Field(default=None, gt=0)   # None -> 由 KV cache 容量推导
    @property
    def enabled(self) -> bool: return self.enable_return_routed_experts
    def compute_hash(self) -> str: ...
```

`VllmConfig._verify_artifact_compatibility`（`vllm/config/vllm.py`）在 `enabled` 时强制：

- 必须：Model Runner V2（`use_v2_model_runner`）、`runner_type == "generate"`、`is_moe`、
  prefix caching 开启；
- 禁止：adaptive speculative verification、PP > 1、DCP/PCP > 1、KV connectors（PD 分离 / KV offload）。

> logprobs 接入时注意：`is_moe` 这个 gate 对 logprobs 是**错的**（logprobs 适用于任意模型），见 §9.4。

---

## 7. 端到端数据流（PR #45635 实际时序）

### 7.1 scheduler 控制面

```text
scheduler.__init__:  artifact_connector = ArtifactSchedulerConnector() if artifact_config.enabled else None
scheduler.schedule:  scheduler_output.artifact_connector_metadata = build_connector_meta(output, self.requests)
                     （增量打包 hash + 计算 emit_start + 转交 finished_requests）
scheduler._preempt_request: artifact_connector.request_finished(request)   # 释放前先记账
scheduler.update_from_output: routed_experts = artifact_connector.take_output(request, output)
                     （从 ArtifactRequestOutput 切 [emit_start, token_end)）
```

### 7.2 worker 数据面

```text
model_runner.init_artifact_connector(kv_cache_config):
    artifact_connector = ArtifactWorkerConnector(model, kv_cache_config, max_num_tokens, vllm_config)
model_runner.execute_model:
    artifact_connector.begin_step(scheduler_output.artifact_connector_metadata)  # forward 前
    ...（GPU forward，capturer 在各 MoE layer 内 capture topk_ids 到 device_buffer）
model_runner.sample_tokens:
    pending = artifact_connector.prepare_output(req_ids, num_computed_tokens_np, query_start_loc_np)
    AsyncOutput(..., pending_artifact_output=pending)
AsyncOutput.__init__:
    with stream(copy_stream, main_stream):
        routed_experts = async_copy_to_np(pending.routed_experts)      # 异步 D2H
        num_rejected   = async_copy_to_np(sampler_output.num_rejected)
AsyncOutput.get_output:
    artifact_connector_output = pending.connector.process_output(
        req_ids, token_starts, query_start_loc, routed_experts,
        num_sampled_tokens_np, num_rejected)                          # 提交 + materialize
    pending.complete()                                                # finally 保证不卡死
    model_runner_output.artifact_connector_output = artifact_connector_output
```

数据平面流转：`RoutedExpertsCapturer.device_buffer (GPU int32)` → `snapshot_routing_data` 窄化为
`uint8/uint16` → `AsyncOutput` 异步 D2H 成 numpy → `process_output` 切分/截断 → `buffer.capture`
凑满 block → `publish_routed_experts`（`array.tobytes` → `BackgroundArtifactStore.put` 入队）→
后台线程写 `InProcessArtifactStore` 匿名 mmap → `materialize_routed_experts` 读回组装 consumer output。

---

## 8. 差异清单：aoshen02#11 → #45635（本次校准核心）

| 维度 | aoshen02#11（旧文档基准，已过时） | #45635（权威） |
|---|---|---|
| key 公式 | `vllm-artifact/sha256(generation+"\0"+block_hash)` | `vllm-artifact/{generation}/{block_hash.hex()}`（**明文 hex，无哈希**） |
| store | `LocalSharedMemoryArtifactStore`（文件 arena.bin + flock + TTL GC + 多进程） | `InProcessArtifactStore`（**匿名 mmap** + 单 owner + 引用计数，无 flock/GC） |
| `shm.py` / `buffer.py` | 独立文件 | **删除**，职责并入 `store.py` / `routed_experts.py` |
| 读接口 | `get(keys) -> list[bytes]` | `get_concatenated(keys) -> bytes` |
| 对象尺寸 | 变长（逐 object payload） | **定长 slot**（`object_nbytes` = 一个 hash block） |
| 生命周期 | TTL stale GC | **引用计数**（`_references` / `retain` / `release`）对齐 KV cache |
| metadata | `requests: list[ArtifactRequestMetadata]`、`finished_requests: dict`、带 `block_size` | `requests: dict[str,int]`、`finished_requests: tuple[str,...]`、无 `block_size` |
| block 粒度 | 单一 `block_size` | `scheduler_block_size` vs `hash_block_size`（`resolve_kv_cache_block_sizes`） |
| ArtifactConfig | `enable_return_routed_experts` + `shm_dir` + `max_shm_bytes` + `shm_ttl_seconds` | `enable_return_routed_experts` + `max_bytes` 两项 |

**结论**：commit `f6406bd`「remove unused store abstractions」删掉了尚未使用的 SHM/Protocol 抽象，
`22d5ec7`「Align artifact lifetime with KV cache」把「外部 TTL GC」换成「KV block 引用计数」。
方向从「独立跨进程共享存储」收敛为「**跟随 KV cache 存活周期的进程内 artifact 缓存**」，为后续
shm/mooncake/kvcc 多后端留了 `store.py` 这个唯一挂点，但**当前只有 `InProcessArtifactStore` 一个实现**。

---

## 9. logprobs 接入路径（基于 #45635 真实结构）

PR #45635 之后，logprobs 接入仍「不重新发明统一 Core」，但有**三处比 aoshen02#11 时代更硬的结构约束**：

### 9.1 key 冲突（明文 key 下更尖锐）

`routed_experts_keys` 是 `vllm-artifact/{generation}/{block_hash.hex()}`，**不含 field 维度**。
logprobs 与 R3 共用同一 `block_hash`（同一个 KV prefix block）时，key 必然撞。接入时必须：

```python
def logprobs_keys(block_hashes, artifact_namespace) -> list[str]:
    prefix = f"vllm-artifact/{artifact_namespace}/logprobs/"   # 加 field 段
    return [prefix + block_hash.hex() for block_hash in block_hashes]
```

（或 `f"vllm-artifact/logprobs/{artifact_namespace}/"`，任选其一，但**必须**与 R3 前缀区分。）

### 9.2 store 定长 slot → 每 field 一个 store（关键新约束）

`InProcessArtifactStore(max_bytes, object_nbytes)` 的 `object_nbytes` 是**构造时固定**的：
- R3：`hash_block_size * num_layers * num_experts_per_tok * sizeof(uint8|uint16)`
- logprobs：`hash_block_size * width * sizeof(float32)`（width 由 `sampling_params.logprobs` 决定）

两者尺寸不同，**同一个定长 store 装不下两种 object**。因此 worker 不能只持一个 `self._store`，需改为
**每 field 一个 store**（`self._r3_store` / `self._logprobs_store`），或把 store 泛化为「按 field 分槽」
的 `dict[str, InProcessArtifactStore]`。这是 #45635 时代 logprobs 接入的**第一道结构性改动**，
比 aoshen02#11 时代的「只加 field 到 key」更重。

### 9.3 buffer 已泛化，capture 源切换

`RoutedExpertsArtifactBuffer` 已参数化 `dtype`/`shape_per_token`，改名 `ArtifactBuffer` 即可复用给
logprobs 的 tail staging（全对齐块直出 / tail 凑 block 逻辑不变）。capture 源从
`capturer.snapshot_routing_data`（GPU 端 `topk_ids` 窄化）换成
`sampler_output.logprobs_tensors`（方向 1 §10 的 `ingest_logprobs` 落点），在 `prepare_output` 里
snapshot、在 `process_output` 里复用 `num_sampled`/`num_rejected` 边界处理。

### 9.4 兼容性 gate 需放宽

`_verify_artifact_compatibility` 的 `is_moe` 检查对 logprobs **不成立**（logprobs 任意模型都可用）。
接入 logprobs 时要改成：R3 要求 `is_moe`，logprobs 不要求；`bind_routed_experts_capturer`（必须找到
MoE router，否则 raise）也要在非 MoE + 仅 logprobs 场景下**跳过**。

### 9.5 logprobs 专用 emit 语义

R3 的 `emit_start = max(routed_experts_prompt_start, ...)` 是「从 prompt_start 到最后一 token」。
logprobs 的 PREDICTED_TOKEN 坐标（`logprob(token_i) <- forward(token_{i-1})`）需要一个不同的
首 token sentinel + valid_len 处理，放在 `materialize_logprobs` 侧，不进 key（见方向 1 §10）。

---

## 10. 验收清单（以 #45635 为基线，增补 logprobs 项）

- [ ] `InProcessArtifactStore`：定长 slot 幂等、`_evict_to_fit` fail-closed、`get_concatenated` 读自己写；
- [ ] 引用计数：`retain`/`release` 正确驱动 `terminal_order`，KV block 释放后 artifact 可被淘汰；
- [ ] `BackgroundArtifactStore`：coalesced batch、`_error` fail-closed、`close` 干净退出；
- [ ] R3 端到端：capture → snapshot → D2H → buffer → publish → materialize 与 slot 直出逐元素一致；
- [ ] prefix cache 命中：同 `block_hash` 的 R3 block 跨 request 复用（幂等 skip put）；
- [ ] reset：generation 变化后 worker 清空 buffer + release 旧 key、旧 object 不被误读；
- [ ] spec/MTP：`num_rejected` 截断后 rejected rows 不进 commit；
- [ ] `_verify_artifact_compatibility`：V2/generate/MoE/prefix-cache 缺一即拒，PP/DCP/PCP/KV-connector 拒；
- [ ] **（新增）logprobs 与 R3 分 store**：不同 `object_nbytes` 各得各的 arena，互不覆盖；
- [ ] **（新增）logprobs key 前缀与 R3 不冲突**：同 `block_hash` 下两 field 各得各的 object；
- [ ] **（新增）非 MoE 模型 + 仅 logprobs**：绕过 `bind_routed_experts_capturer` 的 raise 路径；
- [ ] 关闭 artifact 开关零开销（`artifact_connector is None` 短路）。

---

## 11. 遗留（骨架阶段不解决）

- **真·SHM / Mooncake / kvcc 后端**：当前只有 `InProcessArtifactStore`（匿名 mmap 单 owner）。真正的
  `LocalSharedMemoryArtifactStore`（跨进程）与 mooncake 是下一阶段，`store.py` 是唯一挂点；届时
  `get_concatenated`/`put(retain/release)` 契约需保持（引用计数语义可能退化为 no-op 或后端原生）。
- **store 多 field 泛化**：logprobs 接入后若「每 field 一个 store」重复，再提炼 `dict[field, store]`
  或把 `object_nbytes` 改成 per-key 尺寸索引（引入变长管理复杂度）。
- **`RoutedExpertsManager`（slot buffer）退役**：capture 已切到 logical-object 发布，slot buffer 仅作
  capture 中间态，逐步移除。
- **`dp_sync_interval`**：review 讨论里提到的新 engine/CLI 选项（对应 commit `5968940`「Allow
  intentional routed-expert output sync」），未在本文件已核实的 head 源码里定位到，待确认后再补。
- **multi-rank writer**：单 owner 正确性后按 token/layer logical ranges 并行（见
  [Prefix Artifact Multi-rank Writer 拓扑](Prefix Artifact Multi-rank Writer 拓扑.md)）。
