# Prefix Artifact Logprobs 捕获设计

日期：2026-09-07

状态：设计稿，待评审。基于 [可落地设计 V3](Prefix Execution Artifact Store：可落地设计 V3.md)、
[功能与并行兼容性](Prefix Artifact 功能与并行兼容性.md)、[PR Roadmap](Prefix Execution Artifact Store：PR Roadmap ZH.md)，
以及 vLLM 三个既有 PR 的捕获范式推导而来。

---

## 0. 关联与依据

本文不重复总体架构、public contract 和 backend 协议，只回答一个问题：**logprobs 作为一个
`PREFIX_BLOCK` token-wise artifact field，如何捕获、如何与既有 R3 管道对齐。**

参考的三个 PR 范式：

| PR | 内容 | 对 logprobs 的价值 |
|---|---|---|
| [vllm#45635](https://github.com/vllm-project/vllm/pull/45635) | R3 捕获 baseline：`RoutedExpertsCapturer` + 异步 D2H + KV-block-keyed SHM store | 捕获→D2H→persist→encode 的总骨架；异步输出流复用点 |
| [vllm#50721](https://github.com/vllm-project/vllm/pull/50721) | R3 迁移 MRV2：`ModelRunnerOutput.routed_experts` 字段 + 快照封装 | MRV2 异步输出路径的接入方式 |
| [vllm#49555](https://github.com/vllm-project/vllm/pull/49555) | indexer top-k（DSA）返回：镜像 `--enable-return-routed-experts` 的第二个 token-wise artifact | 第二字段如何复用 R3 管道、prefill/decode/spec 三路切片收敛 |

Roadmap 中 logprobs 的定位（[PR Roadmap ZH](Prefix Execution Artifact Store：PR Roadmap ZH.md)）：
阶段 1（Artifact Connector + SHM）Token-wise 里的 `Logprobs @刘荣`；S2（Mooncake）迁移 `Logprobs`。

---

## 1. 一句话结论

**logprobs 的捕获与 R3 有本质区别：R3 需要在 MoE router 后新增 GPU hook 去截取 `topk_ids`；
logprobs 在 vLLM 里本来就是 sampler 的既有输出。** 因此 decode 阶段的捕获本质是「分流既有输出」
而非「新增计算」，真正难点只有两处：

1. **prompt logprobs 的 boundary hidden-state restore**（prefix 命中时无 forward，需重算）；
2. **`PREDICTED_TOKEN` 坐标与 full-block 对齐**（与 R3 错开一格）。

---

## 2. 坐标语义与 full-block 对齐

### 2.1 语义坐标（遵守既有文档）

[可落地设计 V3 §5.2](Prefix Execution Artifact Store：可落地设计 V3.md) 已定死：

```text
KV / R3 / DSA of token_i  <- forward(token_i)
logprob(token_i)          <- forward(token_{i-1})
```

对序列 `t_0, t_1, …, t_{n-1}`：

| 字段 | 坐标 | 覆盖 | 含义 |
|---|---|---|---|
| R3 | `EXECUTED_TOKEN` | `[0, n-1]` | forward 该 token 产生的路由 |
| logprob | `PREDICTED_TOKEN` | `[1, n-1]` | 生成该 token 的概率 |

三个衍生后果：

1. `t_0`（prompt 首位）**没有 logprob**，也无前驱 row；
2. 最后一个仅 sampled、未 forward 的 token **有 logprob 但无 R3/DSA**；
3. logprob 与 R3 的 token 覆盖错开一格。

### 2.2 存储对齐：推荐「token 对齐 KV block + sentinel」

`PREDICTED_TOKEN` 描述的是 logprob **值的语义来源**（来自「预测」而非「被执行 forward」），
不是要求存储 index 物理错位。为保持 full-block 与 KV/R3 同 hash、同边界，建议存储层按
**token index 对齐 KV block**：

```text
KV block k 覆盖 executed tokens [kB, (k+1)B)
  -> 对应 logprob block 覆盖「这些 token 的 logprob」：logprob(t_kB) … logprob(t_(k+1)B-1)
  -> block 内 index 0（对应 t_0，仅首个 block 存在）填 sentinel
```

好处：

- logprob full-block 的 `kv_block_hash` 与 R3 完全一致，prefix 复用时不产生三 field 边界错位；
- `block_key = H(schema, model_namespace, field="logprobs", dtype, shape_per_token, raw/processed_mode, artifact_block_size, kv_block_hash)` 可直接套用 [§5.3](Prefix Execution Artifact Store：可落地设计 V3.md) 的 key 公式；
- 冷/热路径拼接时不用做 +1/-1 位移。

代价：首个 block 恒有一个 sentinel 位置。若后续证明 sentinel 破坏消费者契约，再回退到
「独立 `PREDICTED_TOKEN` 坐标、block 边界错位 1」的方案（成本更高，不首选）。

### 2.3 与 R3 的联合覆盖关系（用于 finalize 的 ordered key list）

对一个 request，logprob 的 key list 长度与 R3 的关系：

```text
L_r3      = 实际执行 forward、应返回 R3 的 token 数
L_logprob = 实际被采样（predicted）的 token 数 = L_r3（近似，差在「末尾仅采样未 forward」的 1 个 token）

num_logprob_blocks = ceil(L_logprob / B)      # 与 R3 同公式，仅覆盖 token 集合差一格
```

finalize 时两个 field 各自派生 ordered key list，共享同一个 completion barrier：
任一 required field 失败，request 不返回任何 production handle（[§1.4](Prefix Execution Artifact Store：可落地设计 V3.md)）。

---

## 3. 捕获点设计（三条路径）

### 3.1 Decode logprobs —— 从 Sampler 输出分流（零额外计算）

[vllm#45635](https://github.com/vllm-project/vllm/pull/45635) 已确认：R3 是「与 tokens、
**logprobs** 一起放到既有 async output copy stream 上拷贝」。即 **logprobs 已经在异步输出流上传输**。

因此 decode 阶段捕获 = sampler 产出 `SamplerOutput` 之后，把 `logprobs` / `logprob_token_ids`
按 request 切分，挂到与 R3 相同的 `PendingArtifactOutput`（`vllm/v1/worker/gpu/async_utils.py`）
上，走同一条 D2H 流。

- **不新增 GPU capture buffer**；
- **不 hook kernel**；
- 与 R3 共用一个 per-step completion fence（`begin_step()` 等待上一 pending output，防止覆盖）。

### 3.2 Prompt logprobs —— 机制已从源码核实

已核实（`vllm/v1/worker/gpu/sample/prompt_logprob.py`）：vLLM 现有 `PromptLogprobsWorker.compute_prompt_logprobs`
在 **prefill 的每个 chunk 用该 chunk 的最后一层 `hidden_states` 重新走 `model.compute_logits` 算 logprobs**：

```text
forward(chunk) -> hidden_states
   -> logits_fn(hidden_states[:num_tokens])   # logits_fn = self.model.compute_logits
   -> compute_topk_scores(...)                 # logprob 对「下一个 token」算
```

关键事实：

- 分 chunk 计算（`CHUNK_SIZE=1024`），chunked prefill 用 `in_progress_prompt_logprobs` 缓存中间结果最后 `cat`；
- `logits_fn` 注释明确「can be slow because it involves all-gather」——prompt logprobs 是有实际开销的重算，不是 free 的；
- **坐标印证**：`get_prompt_logprobs_token_ids` kernel 注释「shift the pos by one because the logprob is
  computed for the next token」，`target_pos = num_computed_tokens + 1 + block` —— 即 `PREDICTED_TOKEN` 坐标；
- **prefix 命中的关键缺口**：`compute_prompt_logprobs` 依赖 `hidden_states`（当前 step forward 产出），
  命中前缀部分不 forward、无 `hidden_states`，于是该部分的 prompt logprobs 无法重算。

这正解释了本项目要做 logprobs artifact 的动机：**命中前缀的 prompt logprobs 必须作为 `PREFIX_BLOCK`
artifact 持久化复用，否则跨 request 复用前缀时拿不到 hidden_states 去重算。**

分工边界（[§13](Prefix Execution Artifact Store：可落地设计 V3.md)）不变：boundary hidden-state 的
capture/restore 留在 sampler 路径；ArtifactConnector 只负责把 sampler 最终算出的 logprobs 存起来。

仍在 sampler 侧、不属于 ArtifactConnector 的机制：`resumed_after_prompt`（preemption 恢复后跳过重算）、
`in_progress_prompt_logprobs` 的 chunk 合并。

### 3.3 Speculative —— 只保留 accepted

参考 [vllm#45635](https://github.com/vllm-project/vllm/pull/45635) 的
`process_output(req_ids, …, num_sampled_tokens_np, num_rejected)` 与
[功能与并行兼容性 §5.2](Prefix Artifact 功能与并行兼容性.md)：

```text
draft/verification capture
  -> scheduler 决定 accepted range
  -> 只 commit accepted logical positions
  -> rejected 位置被覆盖
```

logprobs 同样：draft token 的 logprob 只在被 accepted 后进入 commit；rejected 丢弃。
stop-trim 后、async in-flight 的行同样不得进入 committed range。

---

## 4. 数据结构与 field profile

### 4.1 shape / dtype（由请求参数决定，进入 field profile）

已核实的 vLLM 实际结构（`vllm/v1/outputs.py` 的 `LogprobsTensors`）：

| 字段 | shape / dtype | 含义 |
|---|---|---|
| `logprob_token_ids` | `[num_tokens, max_num_logprobs + 1]` int32 | 第 0 列 sampled token，后接 top-k |
| `logprobs` | `[num_tokens, max_num_logprobs + 1]` float32 | 对应位置的 logprob |
| `selected_token_ranks` | `[num_tokens]` int64 | sampled token 的排名 |
| `cu_num_generated_tokens` | `[num_reqs + 1]` | per-request 切片边界 |

注意是**扁平化 `[num_tokens, width]` 布局**，靠 `cu_num_generated_tokens` 做 per-request 切分，
不是按 request 分块。capture 时直接复用该布局，ArtifactConnector 按 `cu_num_generated_tokens`
还原 per-request 边界即可。

| 请求 | 内容 | 对应 width |
|---|---|---|
| 默认（仅 sampled token 的 logprob） | `logprob(t_i)` | `1`（只有第 0 列） |
| `logprobs=k` | top-k logprob + 对应 token id | `1 + k` |

### 4.2 raw / processed mode

[§5.2](Prefix Execution Artifact Store：可落地设计 V3.md) 要求 logprobs profile 区分
**raw / processed mode**：

- raw = 采样器原始 logprob；
- processed = 经 temperature/penalty 等 logits processor 之后的值。

该标记必须进入 field descriptor 并参与 content key hash，否则 raw/processed 混用会污染 prefix reuse。

### 4.3 object envelope

复用 [§5.5](Prefix Execution Artifact Store：可落地设计 V3.md) 的统一逻辑 header，`field` 改为
`"logprobs"`，`dtype` 为 `|f4`（float32），`valid_len` 为 block 内有效 logprob 行数
（首 block 因 sentinel 减 1）。

---

## 5. 接入 ArtifactConnector

照搬 [vllm#49555](https://github.com/vllm-project/vllm/pull/49555) 复刻 R3 的范式，logprobs
作为第二个 token-wise field 接入统一 Core：

```text
Sampler 输出 (logprobs / logprob_token_ids)
   -> 按 request 切分（prefill / decode / spec 三路，参考 IndexerTopkManager.get_request_topk 的切片收敛）
   -> ArtifactWorkerConnector 收进 uncommitted per-request logical suffix
   -> ArtifactRequestCore（统一 field adapter，坐标语义 = PREDICTED_TOKEN，存储按 token 对齐）
   -> ShmArtifactStore / MooncakeArtifactStore
```

必须遵守的既有约束：

- **不建独立 `prompt_logprobs.py`**、不建 backend-specific materializer、不建 Codec 继承树或
  manager hierarchy（[§13](Prefix Execution Artifact Store：可落地设计 V3.md)）；
- logprobs 是 `PREFIX_BLOCK` 下「一个统一 `logprobs` field」，开 `prompt_logprobs` 只是给该
  field 加 prompt coverage，**不产生第二种 artifact**（[§1.4](Prefix Execution Artifact Store：可落地设计 V3.md)）；
- 启用 artifact backend 后 logprobs 也从 backend 读取，HTTP 不再重复返回完整 logprobs value
  （[§1.4](Prefix Execution Artifact Store：可落地设计 V3.md)）；
- 命中条件升级为三态原子 `KV ready AND R3 ready AND prompt-logprobs ready`，**不允许 KV 命中
  但 prompt-logprobs 不命中的降级**，视为 connector miss 或一致性错误
  （[功能与并行兼容性 §5.1](Prefix Artifact 功能与并行兼容性.md)）；
- `fields.py` 中 logprobs 是同一 Core 的 field descriptor 配置，不建立 per-field Manager/生命周期
  （[§13](Prefix Execution Artifact Store：可落地设计 V3.md)）。

---

## 6. 与 R3 捕获的差异清单

| 维度 | R3 | Logprobs |
|---|---|---|
| 捕获方式 | 新增 GPU hook（MoE router 后截 topk_ids） | 复用 sampler 既有输出，分流即可 |
| 坐标语义 | `EXECUTED_TOKEN` | `PREDICTED_TOKEN`（错 1 格） |
| dtype | uint8 / int32 | float32 |
| 特有难点 | slot mapping 对齐 | boundary hidden-state restore |
| 首 token | 有 | 无（无前驱，sentinel） |
| 额外计算 | 无（forward 副产品） | prompt logprobs 冷/热均需 logits→logprob |
| 与 logits processor 耦合 | 无 | 有（raw/processed mode） |

---

## 7. 验证矩阵

- decode 冷启动：logprob 与 `logprobs=k` 的 top-k 逐项一致；
- prompt 首 token 无 logprob、sentinel 处理正确；
- prefix full/partial hit 时 boundary restore 后 logprob 与冷启动 exact
  （对齐 [§11.3](Prefix Execution Artifact Store：可落地设计 V3.md) 的 Prompt Logprobs 验收）；
- speculative：rejected token 的 logprob 不进入 commit；
- `PREDICTED_TOKEN` 语义下 full-block key 派生与拼接正确（首 block sentinel 不进 payload）；
- 三态原子命中（KV + R3 + logprobs）缺一 fail closed；
- SHM 实际 logprob 与 Mooncake key list materialization 逐元素一致；
- raw/processed mode 不一致时不得复用同一 content key。

---

## 8. 开放问题（生产 PR 前需核实）

1. ~~boundary hidden-state~~ **（已核实，见 §3.2）**：命中前缀无 `hidden_states` 正是 logprobs
   artifact 的核心动机，无需再重算。
2. **sentinel 选择**：logprob 是 float32，需选一个不参与数值范围的 sentinel（如 `-inf` 或独立
   valid_len 标记），确认不会与合法 logprob（`-inf` 本身是合法值，表示概率 0）冲突——优先用
   header 的 `valid_len` 表达，而非数值 sentinel。
3. **raw/processed 与 content key**：确认哪个 mode 是消费者实际需要的，避免两种 mode 并存导致
   prefix 复用命中率减半。
4. **末尾 token 归属**：`L_logprob` 与 `L_r3` 在「末尾仅采样未 forward」的 1 个 token 上的差异，
   需在 finalize 的 ordered key list 中显式规定。

---

## 9. 已核实代码锚点（vllm fork @ main）

| 关注点 | 文件 | 关键符号 |
|---|---|---|
| logprob 计算（decode + top-k） | `vllm/v1/worker/gpu/sample/logprob.py` | `compute_token_logprobs`, `compute_topk_scores` |
| prompt logprobs（boundary hidden-state） | `vllm/v1/worker/gpu/sample/prompt_logprob.py` | `PromptLogprobsWorker`, `compute_prompt_logprobs`, `get_prompt_logprobs_token_ids` |
| sampler 输出结构 | `vllm/v1/worker/gpu/sample/output.py` | `SamplerOutput`（含 `logprobs_tensors`） |
| 传输结构 | `vllm/v1/outputs.py` | `LogprobsTensors`, `RoutedExpertsTensors`, `LogprobsLists` |
| prompt logprobs 调用点 | `vllm/v1/worker/gpu/model_runner.py` | `compute_prompt_logprobs`（约 L1954，传入 `hidden_states` + `model.compute_logits`） |
| R3 捕获范式（参照） | `vllm/model_executor/layers/fused_moe/routed_experts_capturer.py` | `RoutedExpertsCapturer`, `RoutedExpertsManager` |

**现状提醒**：本 fork 尚未合入 `artifact_connector` 目录（无 `ArtifactConnector`/`ArtifactRequestCore`），
当前只有 R3 的 capture baseline。logprobs 需在 R3 baseline 之上，按 [PR Roadmap 阶段 1](Prefix Execution Artifact Store：PR Roadmap ZH.md)
补出「统一 Core + SHM」后再接入本字段。

---

## 10. 方向 1：伪代码级实现设计（可直接照着写）

日期：2026-09-07（对照 fork 源码逐行核实后落定）

本节回答三个具体问题：**capture 接入点落在哪一行**、**如何按 `cu_num_generated_tokens`
做 per-request 切分**、**full-block key 怎么用 `PREDICTED_TOKEN` 范围派生**。

### 10.0 一句话结论：worker 侧零改动，接入点在 scheduler 侧

logprobs 与 R3 的本质区别决定了接入点不同：

| | R3 | Logprobs |
|---|---|---|
| worker 侧是否需要新增 hook | **是**（MoE router 后截 `topk_ids`） | **否**（sampler 既有输出） |
| capture 传输链路 | 需新建 `AsyncOutput.routed_experts` 字段 | **已存在**（`AsyncOutput.logprobs_tensors`） |
| 新增接入点 | `model_runner.py` + `async_utils.py` + scheduler | **仅 scheduler 侧 `output_processor.py`** |

logprobs 的 GPU→CPU 异步 D2H **现有代码已经做完**：

- `vllm/v1/worker/gpu/async_utils.py` L141–145：`sampler_output.logprobs_tensors.to_cpu_nonblocking()`
- `vllm/v1/worker/gpu/async_utils.py` L192–193：`self.model_runner_output.logprobs = self.logprobs_tensors.tolists()`

所以方向 1 的真正落点是 **scheduler 侧分流**，不是 worker 侧新增。唯一需要动 worker 侧的是
「强制产出 logprobs」（§10.1.3），因为 artifact 模式下即使用户没开 `logprobs=k`，也要为
`PREFIX_BLOCK` 持久化至少 `width=1` 的 sampled-token logprob。

### 10.1 capture 接入点（逐行）

#### 10.1.1 worker 侧：已就绪，无需改动

现有链路（[model_runner.py](vllm/v1/worker/gpu/model_runner.py) `sample_tokens`）：

```text
L1940  sampler_output, num_sampled, num_rejected = self.sample(hidden_states, ...)
         └─ sampler.__call__ 产出 sampler_output.logprobs_tensors (LogprobsTensors, GPU)
L1974  async_output = AsyncOutput(..., sampler_output=sampler_output, ...)
         └─ __init__ L141–145: logprobs_tensors.to_cpu_nonblocking()   # 异步 D2H
L1996  self.postprocess_sampled(...)                                    # 与 D2H 重叠
        └─ get_output() L192–193: model_runner_output.logprobs = ...tolists()
```

`LogprobsTensors`（GPU）→ `LogprobsLists`（CPU）的结构已在 [outputs.py](vllm/v1/outputs.py) 核实：
`LogprobsTensors.tolists()`（L94–105）产出 `LogprobsLists(logprob_token_ids, logprobs,
sampled_token_ranks, cu_num_generated_tokens)`，其中 `cu_num_generated_tokens` 透传自
`compute_topk_scores`（[logprob.py](vllm/v1/worker/gpu/sample/logprob.py) L180–187）。

#### 10.1.2 scheduler 侧：新增分流点（本设计的核心改动）

消费 `ModelRunnerOutput.logprobs` 的位置在 [output_processor.py](vllm/v1/engine/output_processor.py)：

- L670–672：R3 的消费点，`req_state.routed_experts_chunks.append(engine_core_output.routed_experts)`
- L690–705：logprobs 的**既有**消费点，`req_state.logprobs_processor.update_from_output(...)`

新增分流落在 **L690–705 旁**，与既有 logprobs 消费并列（不要改现有组装 response 的路径）：

```python
# output_processor.py —— 在消费 engine_core_output 的循环内，logprobs 既有消费旁新增：
if engine_core_output.logprobs is not None:
    # (现有) 组装 response 的 logprobs，不动
    req_state.logprobs_processor.update_from_output(engine_core_output)
    # (新增) 分流进 ArtifactConnector 的 per-request logical suffix
    if self.artifact_connector is not None:
        self.artifact_connector.ingest_logprobs(
            req_state=req_state,
            logprobs=engine_core_output.logprobs,      # LogprobsLists
            token_start=req_state.num_computed_tokens_prev,  # 本 step 起始 token index
        )
```

`token_start` 的语义（见 §10.2）：本 step 新生成 token 的绝对 index 起点。scheduler 侧
`req_state` 已追踪 `num_computed_tokens`，step 开始时值为 `token_start`，step 结束后
`num_computed_tokens_new = token_start + 本 step 生成 token 数`。这两者由既有
`postprocess_sampled` / output processor 维护，connector 直接取用即可，不需要自己重建。

#### 10.1.3 worker 侧唯一改动：强制产出 logprobs

artifact 模式下，即使用户 `sampling_params.logprobs is None`，也必须产出至少 `width=1` 的
sampled-token logprob。改动点：[sampler.py](vllm/v1/worker/gpu/sample/sampler.py) 的
`get_logprobs_dims`（L106–120）：

```python
# sampler.py get_logprobs_dims —— 现状：无 request 要 logprobs 时返回 None，导致
#   __call__ L174 置 logprobs_tensors=None，async_utils L142 跳过传输。
def get_logprobs_dims(self, idx_mapping_np, include_token_ids=True):
    max_num_logprobs = self.sampling_states.max_num_logprobs(idx_mapping_np)
    max_token_ids = self.logprob_token_ids_state.max_num_token_ids(idx_mapping_np) \
        if include_token_ids else 0
    if max_num_logprobs == NO_LOGPROBS and max_token_ids == 0:
        # (新增) artifact 模式：至少返回 (0, 0)，让 compute_topk_scores 走
        #   logprob.py L122–124 的 width=1 fast path（只算 sampled token 的 logprob）。
        if self.capture_logprobs_for_artifact:
            return 0, 0
        return None
    num_logprobs = max_num_logprobs if max_num_logprobs != NO_LOGPROBS else 0
    return num_logprobs, max_token_ids
```

`capture_logprobs_for_artifact` 是新增的开关（对齐 R3 的 `--enable-return-routed-experts`），
由 artifact backend 启用时置 `True`。这样 `__call__` L157 `if logprobs_dims is not None`
恒为真，`logprobs_tensors` 恒非 None，后续传输链路零改动。

### 10.2 按 `cu_num_generated_tokens` 做 per-request 切分

`LogprobsLists` 是**扁平化 `[total_generated_tokens, width]` 布局**（非按 request 分块），
靠 `cu_num_generated_tokens`（`[num_reqs + 1]` 累积边界）还原 per-request 边界。

**关键语义（已核实 sampler.py L161–162）**：`cu_num_generated_tokens` 为 `None` 当且仅当
`expanded_logits == False`（普通 decode，无 spec，`logits.shape[0] == num_reqs`），此时每个
request 恰好生成 1 个 token，flat 布局的行顺序天然等于 `arange(num_reqs + 1)`；非 None 仅出现在
spec decode / 多 token / expanded logits，值为 `cu_num_logits_np.tolist()`。

```python
# artifact_connector / logprobs_field.py
def slice_logprobs_per_request(
    logprobs: LogprobsLists,
    num_reqs: int,
) -> list[np.ndarray | None]:
    """把 step-level flat logprobs 切成 per-request 的 `[gen_i, width]` 行。

    Args:
        logprobs: ModelRunnerOutput.logprobs（LogprobsLists），本 step 全部
            request 全部新生成 token 的 logprob 拼接。
        num_reqs: 本 step 的 request 数。

    Returns:
        rows[i] = request i 本 step 新生成 token 的 logprob，shape
        ``[gen_i, width]``；``None`` 表示该 request 本 step 未生成 token
        （chunked-prefill 中间块）。
    """
    cu = logprobs.cu_num_generated_tokens
    if cu is None:
        # 普通 decode：每 request 恰 1 行，边界即 arange(num_reqs + 1)
        cu = list(range(num_reqs + 1))
    rows: list[np.ndarray | None] = []
    for i in range(num_reqs):
        s, e = cu[i], cu[i + 1]
        rows.append(logprobs.logprobs[s:e] if e > s else None)
    return rows
```

注意：切分后得到的是 **step-level 增量**，不是 request 历史累积。要得到 logical suffix 的
绝对 token index，必须叠加 `token_start`（§10.3）。

### 10.3 追加进 per-request logical suffix（token index 对齐）

R3 用 `slot_mapping` 把 routing 数据 fancy-index 到 **physical slot buffer**
（[routed_experts_capturer.py](vllm/model_executor/layers/fused_moe/routed_experts_capturer.py) L381
`store_batch`）。logprobs **没有 slot_mapping**，只有 logical token 顺序，因此它的 uncommitted
区是 **per-request、按 token index 排列的 logical suffix**，不是 slot buffer。

```python
# artifact_connector / logprobs_field.py
class LogprobsLogicalSuffix:
    """Per-request 的 logprob 累积区，按绝对 token index 对齐。

    token 0 位置恒为 SENTINEL（logprob(t_0) 无前驱），占用 index 0 但不进 payload。
    每个新生成 token t_j 的 logprob 落在 index j —— 这正是 §2.2 的
    「token 对齐 KV block + sentinel」：logprob(t_j) 存 t_j 所在 block。
    """
    SENTINEL = None  # 用 None 表达「无值」，不占用 float32 数值范围（§8.2）

    def __init__(self) -> None:
        self.rows: list[np.ndarray | None] = [self.SENTINEL]  # token 0 占位

    def append_step(
        self,
        rows: list[np.ndarray | None],  # §10.2 的切分结果
        token_start: int,               # 本 step 起始 token index
    ) -> None:
        for offset, row in enumerate(rows):
            if row is None:
                continue
            j = token_start + offset
            assert j == len(self.rows) or j < len(self.rows)  # 单调递增，可覆盖未 forward 的领先行
            while len(self.rows) <= j:
                self.rows.append(None)
            self.rows[j] = row
```

**与 R3 的坐标对齐**（PREDICTED_TOKEN 的体现）：`rows[offset]` 是 token
`t_{token_start + offset}` 的 logprob。`token_start` 是 step 开始时已 forward 的 token 数，
所以 `rows[0]` 对应 `logprob(t_{token_start})`（其值来自 `forward(t_{token_start - 1})`）。

### 10.4 full-block key 派生（PREDICTED_TOKEN 范围）

finalize 时，把 logical suffix 按 token index 对齐 KV block，用 **与 R3 完全相同的
`kv_block_hash`** 派生 key（§2.2 的核心收益：三 field 同 hash 同边界）。

坐标关系（B 为 block size）：

```text
KV block k  覆盖 executed tokens [kB, (k+1)B)
R3 block k  覆盖 R3(t_kB) … R3(t_(k+1)B-1)     # EXECUTED_TOKEN，覆盖 [kB, (k+1)B)
logprob block k 覆盖 logprob(t_kB) … logprob(t_(k+1)B-1)
                                                    # PREDICTED_TOKEN，同样对齐 [kB, (k+1)B)
                                                    # 首 block index 0 (t_0) 是 sentinel
```

```python
# artifact_connector / logprobs_field.py —— finalize 阶段
def derive_logprob_block_keys(
    req_state,
    field_profile: LogprobsFieldProfile,
) -> list[str]:
    """派生 ordered key list，长度与 R3 相同（共享 completion barrier）。"""
    B = req_state.artifact_block_size
    n_forwarded = req_state.num_computed_tokens      # 已 forward 的 token 数
    num_blocks = ceil(n_forwarded / B)               # 与 R3 同公式
    keys = []
    for k in range(num_blocks):
        kv_block_hash = req_state.kv_block_hashes[k]  # KV block k 的 content hash
        keys.append(hash_key(
            schema=SCHEMA_VERSION,
            model_namespace=req_state.model_namespace,
            field="logprobs",
            dtype="f4",                               # float32
            shape_per_token=field_profile.width,      # 1（默认）或 1+k
            mode=field_profile.mode,                  # raw / processed
            artifact_block_size=B,
            kv_block_hash=kv_block_hash,
        ))
    return keys
```

**PREDICTED_TOKEN 范围如何进入 key**：logprob block k 与 KV/R3 block k 共享同一
`kv_block_hash`（都是 KV block `[kB, (k+1)B)` 的 hash），差异只在 payload 的 `valid_len` 与
首 block 的 sentinel，**不改变 key 公式**。这避免了「独立 PREDICTED_TOKEN 坐标、block 边界错位
1」方案（§2.2 已否决）的 +1/-1 位移。

### 10.5 payload 组装（首 block sentinel + valid_len）

```python
# artifact_connector / logprobs_field.py —— finalize 阶段
def pack_logprob_block(req_state, k: int) -> tuple[np.ndarray, int]:
    """组装 block k 的 payload，返回 (payload, valid_len)。

    首 block (k=0) 的 index 0 是 t_0 sentinel，不进 payload，valid_len 减 1。
    """
    B = req_state.artifact_block_size
    n_forwarded = req_state.num_computed_tokens
    lo = k * B
    hi = min((k + 1) * B, n_forwarded)
    suffix = req_state.logprob_suffix          # §10.3 的 logical suffix
    payload = suffix.rows[lo:hi]               # logprob(t_lo) … logprob(t_hi-1)
    valid_len = hi - lo
    if k == 0:
        payload = payload[1:]                  # 去掉 token 0 sentinel
        valid_len -= 1
    # payload 是 list[np.ndarray]，stack 成 [valid_len, width] float32 后写入
    return np.asarray(payload, dtype=np.float32), valid_len
```

**末尾领先 token 的处理**：`logprob(t_{n_forwarded})`（最后一个 sampled 未 forward 的 token）
其 KV block 尚未分配，`logprob_suffix` 里该 index 已写入但 finalize 只 commit 到
`num_computed_tokens`，领先行自然暂缓，等下一 step forward 后进入 commit。这闭合了
[§8 开放问题 4](#8-开放问题生产-pr-前需核实) 的「末尾仅采样未 forward 的 1 个 token」。

### 10.6 与 R3 的接入点逐行对照

| 步骤 | R3（已有 baseline） | Logprobs（本设计） |
|---|---|---|
| GPU 截取 | `RoutedExpertsCapturer.capture`（MoE hook） | 无（sampler 既有输出） |
| 强制产出开关 | `--enable-return-routed-experts` | `capture_logprobs_for_artifact`（§10.1.3） |
| worker 传输 | `AsyncOutput.routed_experts`（新建字段） | `AsyncOutput.logprobs_tensors`（已存在） |
| D2H | `async_utils.py` L155–157 | `async_utils.py` L141–145（已存在） |
| → Lists | `async_utils.py` L195–196 | `async_utils.py` L192–193（已存在） |
| scheduler 消费 | `output_processor.py` L670–672 `routed_experts_chunks` | `output_processor.py` L690–705 旁新增 `ingest_logprobs` |
| 存储 key | `kv_block_hash`（EXECUTED_TOKEN，slot 对齐） | `kv_block_hash`（PREDICTED_TOKEN，token index 对齐 + sentinel） |
| 恢复 | `RoutedExpertsManager.get(block_ids)`（slot fancy-index） | 从 backend 按 key 读 logprob block，token index 对位拼接 |

### 10.7 遗留（本方向不解决，转方向 2）

- `artifact_connector` 目录尚不存在：`ingest_logprobs` / `LogprobsLogicalSuffix` /
  `derive_logprob_block_keys` / `pack_logprob_block` 都挂在统一 Core 之上，需先有
  「统一 Core + SHM」骨架（方向 2）。
- `kv_block_hashes` / `num_computed_tokens` / `model_namespace` 的获取方需由方向 2 的
  `ArtifactRequestCore` 统一暴露，本文只假定其存在。
