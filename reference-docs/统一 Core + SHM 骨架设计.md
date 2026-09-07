# Prefix Artifact 统一 Core + SHM 骨架设计

日期：2026-09-07

状态：方向 2 骨架设计稿，待评审。依据 [可落地设计 V3](Prefix Execution Artifact Store：可落地设计 V3.md)
§2/§5/§6/§7/§9/§13 与 [PR Roadmap ZH](Prefix Execution Artifact Store：PR Roadmap ZH.md) 阶段 1，
并对照 fork 现有 R3 capture baseline 逐行核实后落定。

---

## 0. 一句话目标

阶段 1 要建立**一个 field-agnostic 的统一 Artifact Connector**，SHM 是第一个 backend。
R3 与 logprobs 是两个「field descriptor 配置」，而不是两套 capture/store 代码。
骨架先把「统一 Core + SHM」立起来，让 R3 先挂上（复用已有 capture baseline），logprobs 按
[Logprobs 捕获设计 §10](Prefix Artifact Logprobs 捕获设计.md) 的 ingest 接口随后挂上。

本文交付：文件/类/接口清单、每个类的职责边界、关键方法签名与伪代码、两条端到端时序、
以及「R3 与 logprobs 如何收敛到同一个 Core」的证明。

---

## 1. 文件布局（对应 V3 §13）

```text
vllm/distributed/artifact_connector/
├── __init__.py
├── protocol.py      # scheduler/worker IPC dataclasses（挂 SchedulerOutput / ModelRunnerOutput）
├── fields.py        # FieldDescriptor + R3 / Logprobs 两个 descriptor 实例（唯一 field-specific）
├── store.py         # ArtifactStore Protocol + factory（opaque bytes，不理解 R3/logprobs）
├── shm.py           # ShmArtifactStore（/dev/shm + atomic rename + checksum + capacity + TTL）
├── request_core.py  # ArtifactRequestCore（唯一 request state machine）
├── connector.py     # ArtifactSchedulerConnector + ArtifactWorkerConnector（编排）
└── mooncake.py      # S2 阶段再加，本骨架只留 factory 挂点
```

与 fork 现状的对应关系：

| fork 已有（R3 capture baseline） | 骨架阶段 | 去向 |
|---|---|---|
| `RoutedExpertsCapturer`（worker GPU hook） | 保留不动 | 仍负责 GPU 截取 |
| `RoutedExpertsTensors/Lists`（传输结构） | 保留不动 | 仍负责 D2H 传输 |
| `AsyncOutput.routed_experts` / `ModelRunnerOutput.routed_experts` | 保留不动 | 数据到达 scheduler 的载体 |
| `RoutedExpertsManager`（slot buffer） | **新建骨架后逐步被取代** | logical-object 发布取代 physical-slot 直出 |
| —（缺）`artifact_connector/` | **本次新增** | 统一 Core + SHM |

---

## 2. 核心抽象：FieldDescriptor（fields.py）

**这是 R3 与 logprobs 差异的唯二收敛点**（另一个是 `pack_block`）。其余 store/core/connector
完全 field-agnostic。任何新 token-wise field（DSA、top-p/k ids）只需新增一个 descriptor。

```python
# fields.py
from dataclasses import dataclass
from collections.abc import Sequence

@dataclass(frozen=True)
class FieldDescriptor:
    field_name: str              # "routed_experts" | "logprobs"
    logical_coordinate: str      # "EXECUTED_TOKEN" | "PREDICTED_TOKEN"
    dtype: str                   # "|u1" | "|f4"（numpy dtype 字符串，进 envelope header）
    shape_per_token: tuple[int, ...]   # R3: (num_layers, topk)；logprobs: (width,)
    artifact_block_size: int
    mode: str | None = None      # logprobs 的 "raw"|"processed"；R3 为 None

    # ---- 唯一 field-specific 的三个方法 ----

    def first_valid_token_index(self) -> int:
        """EXECUTED_TOKEN 从 token 0 起有效；PREDICTED_TOKEN 的 token 0 无前驱。"""
        return 0 if self.logical_coordinate == "EXECUTED_TOKEN" else 1

    def pack_block(self, suffix_rows: Sequence[np.ndarray], block_idx: int
                   ) -> tuple[np.ndarray, int]:
        """把 logical suffix 的 block_idx 段打包成 payload，返回 (payload, valid_len)。

        首 block 若含 sentinel（logprobs 的 token 0），sentinel 行不进 payload，
        valid_len 减 1。这是 §5.5 envelope 的 valid_len 唯一来源。
        """
        raise NotImplementedError

    def materialize(self, payloads: Sequence[np.ndarray]) -> np.ndarray:
        """SHM 模式下，把按序读回的 block/tail payload 拼成实际值（base64 npy 的原料）。"""
        raise NotImplementedError
```

### 2.1 R3 descriptor

```python
# fields.py
R3_FIELD = FieldDescriptor(
    field_name="routed_experts",
    logical_coordinate="EXECUTED_TOKEN",
    dtype="|u1",                      # num_experts <= 256 时 uint8；否则 |u2
    shape_per_token=(num_layers, topk),
    artifact_block_size=B,
    mode=None,
)

# R3 的 pack_block：EXECUTED_TOKEN，无 sentinel
#   payload = stack(suffix_rows[lo:hi])          # [B, layers, topk] uint8
#   valid_len = hi - lo
```

### 2.2 Logprobs descriptor（复用方向 1 的 §10.5）

```python
# fields.py
LOGPROBS_FIELD = FieldDescriptor(
    field_name="logprobs",
    logical_coordinate="PREDICTED_TOKEN",
    dtype="|f4",
    shape_per_token=(width,),          # 1（默认）或 1+k
    artifact_block_size=B,
    mode=raw_or_processed,             # 参与 content key，raw/processed 不得混用
)

# logprobs 的 pack_block：PREDICTED_TOKEN，首 block 去掉 token 0 sentinel
#   payload = stack(suffix_rows[lo:hi])          # [hi-lo, width]
#   if block_idx == 0: payload = payload[1:]; valid_len -= 1
```

**关键性质**：`pack_block` 是唯一知道「首 block 有 sentinel」的地方；`derive_full_block_key`
(§5.3) 对两个 field 完全一样，因为 key 只依赖 `field_name` + `dtype` + `shape_per_token` +
`mode` + `artifact_block_size` + `kv_block_hash`，不依赖 `logical_coordinate`。

---

## 3. Store 契约（store.py）

Store 只处理 **opaque bytes + envelope header**，不解析 R3/logprobs。

```python
# store.py
from typing import Protocol

class ArtifactStore(Protocol):
    def put(self, key: str, envelope: bytes) -> bool:
        """幂等发布。已存在返回 False（调用方据此判断 skip put）。"""
        ...

    def exists(self, key: str) -> bool:
        """prefix admission 的存在性查询。"""
        ...

    def get(self, keys: list[str]) -> list[bytes]:
        """按序取回。逐 object 校验 envelope 的 payload_sha256，坏则 raise。"""
        ...

    def delete(self, keys: list[str]) -> None:
        """GC / capacity 淘汰用。"""
        ...


def create_store(config: ArtifactConfig) -> ArtifactStore:
    """factory：SHM 或 Mooncake。S2 之前 Mooncake 分支 raise NotImplemented。"""
    if config.backend == "shm":
        from .shm import ShmArtifactStore
        return ShmArtifactStore(config)
    raise NotImplementedError(config.backend)
```

### 3.1 envelope 编解码（request_core 提供，store 只存 bytes）

```text
uint32 header_length | canonical JSON header | raw contiguous tensor bytes
```

```json
{"schema_version":1,"kind":"block|tail","field":"routed_experts|logprobs","object_id":"…",
 "dtype":"|u1| |f4","shape":[16,32,8],"valid_len":16,"payload_sha256":"…","header_sha256":"…"}
```

store 不生成 header；`put(key, envelope)` 里 envelope 已由 request_core 组装好。store 的
`get` 只做 `payload_sha256` 校验（防 `/dev/shm` 损坏/半写），不解释 `field`/`shape`。

---

## 4. SHM backend（shm.py）

对应 V3 §9 的六项能力：trusted namespace、content-addressed、atomic 发布、checksum、
capacity limit、TTL/lease cleanup。

```python
# shm.py
class ShmArtifactStore:
    def __init__(self, config: ArtifactConfig) -> None:
        self.root = Path(config.shm_root or "/dev/shm/vllm-artifact")
        self.root.mkdir(parents=True, exist_ok=True)
        self.capacity_bytes = config.shm_capacity_bytes
        self.ttl_seconds = config.shm_ttl_seconds

    def _path(self, key: str) -> Path:
        # key 本身已是 content hash；文件名用 sha256(key) 防路径注入/超长
        return self.root / hashlib.sha256(key.encode()).hexdigest()

    def put(self, key: str, envelope: bytes) -> bool:
        path = self._path(key)
        if path.exists():
            return False                       # 幂等：immutable，已存在即完成
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(envelope)
        tmp.rename(path)                       # atomic publish（同目录 rename）
        self._maybe_evict()                    # capacity 超限时淘汰最旧 / 过 TTL
        return True

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def get(self, keys: list[str]) -> list[bytes]:
        out = []
        for key in keys:
            raw = self._path(key).read_bytes()
            out.append(self._verify_and_strip(raw))   # payload_sha256 校验
        return out

    def delete(self, keys: list[str]) -> None:
        for key in keys:
            self._path(key).unlink(missing_ok=True)
```

要点：

- **atomic**：同目录 `tmp` 写满 `rename`，读者永远看不到半写对象；
- **幂等**：`put` 对已存在 key 直接 `False`，对应 V3 §7.1「backend 对已存在 immutable key 跳过 put」；
- **`put` 返回 True 才算 publish 成功**：SHM 是同步 publish 后 ACK（V3 §7.4）；
- **capacity/TTL**：`_maybe_evict` 用 `mtime` + `ttl`，先淘汰过期，再淘汰最旧，不逐 object 加锁。

---

## 5. Request Core（request_core.py）

V3 §6 已定边界：只持有不可变依赖，负责 key/coverage/envelope/ordered refs/tail/materialize，
不碰 GPU capture、scheduler admission、HTTP、Mooncake lifecycle。

```python
# request_core.py
@dataclass(frozen=True)
class FinalizeResult:
    artifact_keys: list[str]          # ordered full-block keys + tail key（Mooncake 返回）
    materialized: np.ndarray | None   # SHM 模式下 materialize 出的实际值
    block_write_count: int            # 本次实际 put 的 object 数（记账用）


class ArtifactRequestCore:
    def __init__(self, store: ArtifactStore, field: FieldDescriptor,
                 materialize: bool, model_namespace: str) -> None:
        self.store = store
        self.field = field
        self.materialize_mode = materialize      # SHM=True 读回组装；Mooncake=False 返回 keys
        self.model_namespace = model_namespace

    # ---- key 派生（§5.3 / §5.4）----

    def derive_full_block_key(self, kv_block_hash: str) -> str:
        return hash_key(
            schema_version=1,
            model_namespace=self.model_namespace,
            field=self.field.field_name,
            dtype=self.field.dtype,
            shape_per_token=self.field.shape_per_token,
            artifact_block_size=self.field.artifact_block_size,
            kv_block_hash=kv_block_hash,
            # logprobs 额外并入 self.field.mode；R3 为 None 不参与
        )

    def derive_tail_key(self, request_id: str, attempt_id: str,
                        terminal_boundary: int) -> str:
        return hash_key(
            schema_version=1, model_namespace=self.model_namespace,
            field_profile_id=self.field.field_name,   # 或 field 的 profile id
            request_id=request_id, request_attempt_id=attempt_id,
            terminal_executed_boundary=terminal_boundary,
        )

    # ---- 发布 ----

    def commit_full_blocks(self, suffix: LogicalSuffix,
                           kv_block_hashes: Sequence[str],
                           executed_len: int) -> list[str]:
        """只发布新完成的 full block（V3 §7.1）。"""
        B = self.field.artifact_block_size
        keys = []
        for k in range(executed_len // B):            # 只到已完成整块
            payload, valid_len = self.field.pack_block(suffix.rows, k)
            key = self.derive_full_block_key(kv_block_hashes[k])
            envelope = self._encode(payload, valid_len, key, kind="block")
            if self.store.put(key, envelope):         # 幂等，重复 key 跳过
                self._account_write()
            keys.append(key)
        return keys

    def finalize(self, suffix: LogicalSuffix, kv_block_hashes: Sequence[str],
                 executed_len: int, request_id: str, attempt_id: str
                 ) -> FinalizeResult:
        """V3 §7.2：从 KV hashes 确定性派生 ordered keys，tail 单独 put。"""
        B = self.field.artifact_block_size
        num_full = executed_len // B
        has_tail = (executed_len % B) != 0

        full_keys = self.commit_full_blocks(suffix, kv_block_hashes, executed_len)

        tail_key = None
        if has_tail:
            tail_payload, tail_len = self.field.pack_block(suffix.rows, num_full)
            tail_key = self.derive_tail_key(request_id, attempt_id, executed_len)
            self.store.put(tail_key, self._encode(tail_payload, tail_len,
                                                  tail_key, kind="tail"))

        ordered = full_keys + ([tail_key] if tail_key else [])   # 顺序即 token 顺序
        materialized = None
        if self.materialize_mode:
            materialized = self._materialize(ordered)
        return FinalizeResult(ordered, materialized, len(ordered))

    def _materialize(self, ordered_keys: list[str]) -> np.ndarray:
        payloads = [self._decode(self.store.get([k])[0]) for k in ordered_keys]
        return self.field.materialize(payloads)      # field-specific 拼接
```

**为什么 key 派生放 core 而不是 scheduler**：V3 §7.2「full-block key 从 KV hashes 确定性派生」，
`kv_block_hash` 由 scheduler 提供（KVCacheManager），但 hash 组合与 field profile 绑定，放 core
保持单一事实源；scheduler 只做 admission / existence / commit plan / finalize 触发。

---

## 6. Connector 编排（connector.py）+ IPC（protocol.py）

### 6.1 protocol.py —— 挂在既有 SchedulerOutput / ModelRunnerOutput 上

```python
# protocol.py
@dataclass
class ArtifactConnectorMetadata:
    """scheduler -> worker 下发（随 SchedulerOutput）。"""
    executed_len: int                      # authoritative executed boundary（stop/acceptance 后）
    kv_block_hashes: list[str]             # 已分配 KV block 的 content hash（顺序即 token 顺序）
    commit_blocks: list[int]               # 本次要 commit 的 full block index（增量）
    finalize: bool                         # 是否触发 finalize
    request_id: str
    attempt_id: str

@dataclass
class ArtifactConnectorAck:
    """worker -> scheduler 上返（随 ModelRunnerOutput）。"""
    request_id: str
    status: Literal["ok", "fail"]
    block_write_count: int
```

### 6.2 ArtifactSchedulerConnector

```python
# connector.py —— scheduler 侧
class ArtifactSchedulerConnector:
    def __init__(self, store: ArtifactStore, field: FieldDescriptor): ...

    def before_step(self, requests) -> None:
        # 为每个 request 维护 accepted/executed progress，标记新完成的 full block
        ...

    def build_metadata(self, req) -> ArtifactConnectorMetadata:
        # executed boundary = stop-trim / spec acceptance 之后的值（不能含 in-flight 行）
        # kv_block_hashes 来自 KVCacheManager
        ...

    def on_ack(self, ack: ArtifactConnectorAck) -> None:
        # 记录每个 object 的独立记账；fail 不把同 batch 其它 item 误标 ready（V3 §7.1）
        ...
```

### 6.3 ArtifactWorkerConnector

```python
# connector.py —— worker 侧
class ArtifactWorkerConnector:
    def __init__(self, field: FieldDescriptor): ...

    def ingest_step(self, req_state, rows: list[np.ndarray | None], token_start: int):
        """把本 step accepted rows 追加进 uncommitted per-request logical suffix。

        rows 的来源：
          - R3: ModelRunnerOutput.routed_experts 按 slot_mapping 重建 logical rows
          - logprobs: Logprobs 捕获设计 §10.2 按 cu_num_generated_tokens 切出的 rows
        token_start 是 rows[0] 对应的绝对 token index（scheduler metadata 提供）。
        """
        req_state.suffix.append_step(self.field, rows, token_start)

    def commit(self, req_state, metadata) -> ArtifactConnectorAck:
        core = req_state.core                       # ArtifactRequestCore
        keys = core.commit_full_blocks(req_state.suffix,
                                       metadata.kv_block_hashes,
                                       metadata.executed_len)
        return ArtifactConnectorAck(req_state.request_id, "ok", len(keys))

    def finalize(self, req_state, metadata) -> ArtifactConnectorAck:
        result = req_state.core.finalize(req_state.suffix,
                                         metadata.kv_block_hashes,
                                         metadata.executed_len,
                                         metadata.request_id, metadata.attempt_id)
        req_state.artifact_keys = result.artifact_keys   # scheduler 校验后组装 HTTP
        return ArtifactConnectorAck(req_state.request_id, "ok",
                                    result.block_write_count)
```

### 6.4 LogicalSuffix —— worker 侧 uncommitted 区（field-agnostic 包装）

```python
# request_core.py（或 connector.py）
class LogicalSuffix:
    """per-request、按绝对 token index 排列的 uncommitted 区（V3 §2.2/§3）。

    rows[j] = token t_j 的字段行；R3 从 j=0 起有效，logprobs 的 j=0 恒为 sentinel。
    """
    def __init__(self, field: FieldDescriptor) -> None:
        self.field = field
        self.rows: list[np.ndarray | None] = [None] * field.first_valid_token_index()
        # R3: [] 空起点；logprobs: [None]（token 0 sentinel）

    def append_step(self, field, step_rows, token_start):
        for offset, row in enumerate(step_rows):
            if row is None:
                continue
            j = token_start + offset
            while len(self.rows) <= j:
                self.rows.append(None)
            self.rows[j] = row
```

---

## 7. 两条端到端时序

### 7.1 commit（full block 增量，V3 §7.1）

```text
forward 完成，capture D2H 到达 scheduler
  → ArtifactSchedulerConnector.before_step 计算新完成 full block 增量
  → SchedulerOutput 携带 ArtifactConnectorMetadata(commit_blocks=[k..], finalize=False)
  → worker ArtifactWorkerConnector.ingest_step 把 rows 追加进 suffix
  → worker commit → ArtifactRequestCore.commit_full_blocks
  → ShmArtifactStore.put(key, envelope) 幂等（重复 key 跳过）
  → ArtifactConnectorAck(ok, block_write_count) 上返
  → scheduler 独立记账；一个 object 失败不误标同 batch 其它 ready
```

### 7.2 finalize（V3 §7.2）

```text
scheduler 得到 stop/abort/spec acceptance 的 authoritative executed end
  → ArtifactConnectorMetadata(finalize=True, executed_len, kv_block_hashes)
  → worker finalize → ArtifactRequestCore.finalize
      ├─ full keys 从 kv_block_hashes 确定性派生并幂等 put
      ├─ 有 partial tail → derive_tail_key + put（request-scoped，不进入 prefix lookup）
      └─ SHM: materialize actual R3 / logprobs；Mooncake: 返回 ordered keys
  → worker 返回 finalize ACK
  → scheduler 校验后生成 ordered list[artifact_key]（或 SHM 直接带实际值）
  → 释放暂存 terminal HTTP output
```

---

## 8. R3 与 logprobs 的收敛证明（这是统一性的核心验收）

| 关注点 | R3 | Logprobs | 统一 Core 处理 |
|---|---|---|---|
| capture | GPU hook 截 topk_ids | sampler 既有输出 | connector.ingest_step 统一入口 |
| 坐标 | EXECUTED_TOKEN | PREDICTED_TOKEN | `FieldDescriptor.first_valid_token_index` |
| 首 token sentinel | 无 | 有（token 0） | `FieldDescriptor.pack_block` 首 block 处理 |
| dtype / shape | `|u1` `[layers,topk]` | `|f4` `[width]` | `FieldDescriptor` 字段 + envelope header |
| key 公式 | `field=routed_experts` | `field=logprobs`+`mode` | `derive_full_block_key` 复用同一 hash_key |
| 发布 | full-block put | full-block put | `ArtifactRequestCore.commit_full_blocks` |
| materialize | 拼 topk_ids | 拼 logprob 值 | `FieldDescriptor.materialize` |

**新增一个 token-wise field 的代价 = 写一个 FieldDescriptor（3 个方法）+ 一个 ingest 来源，
不动 store / core / connector 任何一行。**

---

## 9. 骨架的验收清单（对应 Roadmap 阶段 1）

- [ ] `ShmArtifactStore.put/exists/get/delete` 单测：幂等、atomic（并发读无半写）、checksum 坏则 raise；
- [ ] `ArtifactRequestCore.derive_full_block_key` 对 R3 与 logprobs 同 `kv_block_hash` 得到同 hash 键前缀差异仅 `field`；
- [ ] R3 端到端：capture → suffix → core → SHM → materialize，与 `RoutedExpertsManager` slot 直出结果逐元素一致；
- [ ] prefix cache 命中：同 `kv_block_hash` 的 R3 full-block 跨 request 复用（存在性查询命中即 skip put）；
- [ ] logprobs（方向 1 §10 ingest）挂上后：首 block sentinel 不进 payload、`valid_len` 正确、materialize 值与 sampler 原始输出一致；
- [ ] finalize 时 ordered key list 长度 = `ceil(executed_len / B)`，tail 单独 request-scoped；
- [ ] abort/preemption：已提交 full block 保留，未提交 suffix 丢弃，不产生 tail key；
- [ ] 关闭 artifact 开关时零开销（R3 既有 capture 路径不受影响）。

---

## 10. 遗留（转后续 PR，骨架阶段不解决）

- Mooncake backend（S2）：`mooncake.py` 只留 factory 挂点，`create_store` 抛 `NotImplementedError`；
- multi-rank writer：单 writer 正确性之后按 token/layer logical ranges 并行（见 [Multi-rank Writer 拓扑](Prefix Artifact Multi-rank Writer 拓扑.md)）；
- `RoutedExpertsManager` 的退役：骨架立起后，R3 从 slot-buffer 直出迁到 logical-object 发布，slot buffer 仅作为 capture 阶段的中间态保留；
- DSA / top-p/k ids 两个 field descriptor：等各自 capture 稳定后按 §8 追加。
