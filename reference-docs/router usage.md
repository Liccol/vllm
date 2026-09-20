# vllm-router / SGLang Model Gateway 使能

> 多 DP（Data Parallel）场景下，用 **vllm-router** 做请求路由层，通过
> **consistent_hash + X-Session-ID** 把同一个 session/对话的请求固定路由到同一个 DP rank，
> 从而让多 DP 后端的 **KV Cache 能够复用**（避免跨 rank 的重复计算）。
>
> 本文档覆盖两部分：
> - **vllm-router 使能**（原理 + 代码链路 + 验证结果）
> - **sglang model gateway 使能**（框架，待补充）

---

## 1. vllm-router 使能

### 1.1 背景与作用

- vllm 服务端以 **多 DP** 方式部署（如 `--data-parallel-size 2 --tensor-parallel-size 2`），对外可以通过
  **一个 vllm-router** 统一暴露入口。
- vllm-router 采用 **consistent_hash 策略**：根据请求头 `X-Session-ID` 做哈希，
  **同一个 Session-ID 的请求永远路由到同一个 DP rank**。
- 好处：同一对话的多次请求命中同一 rank 的 **KV Cache 命中/复用**，跨请求无需重新计算
  已生成的历史 token，降低重复计算开销、提升多轮对话吞吐。

### 1.2 原理：vllm-router 如何与 vLLM 交互、如何转发到不同 DP rank

#### 1.2.1 总体架构

vllm-router 本身**不参与** vLLM 内部的数据并行（DP）调度。它把每个 DP rank 看成
一个独立的「逻辑 worker」，通过一次普通的 **HTTP 请求 + 一个特殊 HTTP 头
（`X-data-parallel-rank`）** 来指定要把请求送到哪个 rank。完整链路：

```
客户端 --HTTP--> vllm-router --HTTP + X-data-parallel-rank: N--> vLLM APIServer (host:port)
```

关键点：**同一个 vLLM 服务端的所有 DP rank 都监听在同一个 `host:port` 上**
（单个 APIServer 进程内起了多个 DP engine core）。所以 router 并不按"端口"区分 rank，
而是把 rank 号放进请求头里，由 vLLM 服务端把请求派到对应的 DP engine。

这也是为什么 `--intra-node-data-parallel-size N` 只影响 router 对同一个 worker URL 的
展开份数，而 **worker 的物理地址始终是同一个 `host:port`**。

涉及的关键文件：

| 文件 | 作用 |
|---|---|
| `src/routers/http/dp_utils.rs` | DP 工具函数（URL 展开 / rank 提取 / 加请求头） |
| `src/core/worker.rs` | `DPAwareWorker`（DP 感知 worker 抽象） |
| `src/routers/http/router.rs` | 普通（非 PD）路由器的请求转发逻辑 |
| `src/routers/http/vllm_pd_router.rs` | Prefill/Decode 分离（PD）路由器的转发逻辑 |
| `src/config/types.rs` / `py_src/vllm_router/router_args.py` | `intra_node_data_parallel_size` 配置 |

#### 1.2.2 DP rank 的编址方式（`URL@rank`）

vllm-router 用 `http://host:port@rank` 这种带 `@rank` 后缀的字符串唯一标识某一个 DP rank，
并作为一个独立的 worker 注册进 worker 注册表。

- `dp_utils::get_dp_aware_workers`（`dp_utils.rs:34`）把 `http://host:8000` + dp_size 展开成
  `["http://host:8000@0", "...@1", ..., "...@N-1"]`。
- `dp_utils::extract_dp_rank`（`dp_utils.rs:67`）从 `URL@rank` 里拆出 `(base_url, dp_rank)`。

> 提示：`@rank` 只是 **router 内部**用来标识 rank 的约定。真正发给 vLLM 时，
> router 会先把 `@rank` 去掉，用干净的 `base_url` 作为目标地址，rank 单独走请求头。

#### 1.2.3 启动时 DP worker 的展开

在 `router.rs` 的 `Router::new` 里，当 `intra_node_data_parallel_size > 1` 时：

```rust
let worker_urls = if ctx.router_config.intra_node_data_parallel_size > 1 {
    // worker address now in the format of "http://host:port@dp_rank"
    dp_utils::get_dp_aware_workers(&worker_urls, ..., dp_size).await?
} else { worker_urls };
```

每个 worker URL 会被展开成 N 个 `@0..@N-1` 的 `DPAwareWorker`（`worker.rs:497`）。
`DPAwareWorker` 记录两项关键信息：

- `base_url`：真实地址 `http://host:port`（**不含 `@rank`**）
- `dp_rank`：该 worker 对应的 rank 号

`DPAwareWorker::endpoint_url`（`worker.rs:606`）返回**去掉 `@rank` 的干净 URL**，避免 reqwest
把 `@` 误解析成 user:pass（参见 `tests/test_dp_routing.rs` 头部注释里的历史 bug）。

#### 1.2.4 请求转发时如何落到某个 DP rank

每来一个推理请求（如 `/v1/chat/completions`），普通 router 走
`route_typed_request` → `select_worker_for_model`（`router.rs:547 / :513`）：

1. 在可用 worker 里，用负载均衡策略（consistent_hash / round_robin / random 等）
   **选出一个 `DPAwareWorker`**，即选中了一个 rank。
2. 在 `send_typed_request`（`router.rs:774`）里，用 `extract_dp_rank` 取出该 worker 的
   rank，然后给 HTTP 请求加上头（`router.rs:842`）：

   ```rust
   // Add X-data-parallel-rank header for DP-aware routing
   if let Some(dp_rank) = extracted_dp_rank {
       request_builder = request_builder.header("X-data-parallel-rank", dp_rank.to_string());
   }
   ```

   即：**目标 URL 用干净的 base_url**（`router.rs:811`），**rank 号通过
   `X-data-parallel-rank` HTTP 头携带** —— router 只负责「选 rank + 加头」，
   剩下的路由到具体 DP engine 由 vLLM 服务端完成。

#### 1.2.5 健康检查与 GET 类请求

- 健康检查只对**去重后的 base URL** 执行（`router.rs:226` 用 `HashSet` 去掉 `@rank`），
  因为 `@0..@N` 指向同一个端口，只查一次。
- GET 类的 proxy 请求同样会带上 `X-data-parallel-rank`（`router.rs:424` `proxy_get_request`），
  并**明确剥掉客户端自带的同名字段**（`router.rs:442-445`），保证 worker 只能看到
  router 选定的那一个 rank。

#### 1.2.6 PD（Prefill/Decode 分离）模式

启动时加 `--vllm-pd-disaggregation` 会走 `src/routers/http/vllm_pd_router.rs`，把请求拆成
prefill 段与 decode 段，同样用 `X-data-parallel-rank` 指定 rank：

- Prefill：`vllm_pd_router.rs:809` 若未显式给 rank，用 `prefill_dp_round_robin
  .fetch_add(...) % intra_node_data_parallel_size` 轮询选 rank；请求经
  `build_prefill_request_builder`（`:94/:104`）加头。
- Decode：`:965-972` 沿用 prefill 的 rank（保证 P/D 同属一个 DP 组）。
- 同时还会往请求体注入 `kv_transfer_params`（含 `remote_dp_size` 等，`:245`），用于
  Nixl/Mooncake/MoriIO 等 KV 传输协调。

#### 1.2.7 vLLM 服务端如何响应

Router 用的是标准 `reqwest::Client`（普通模式的 `router.rs`、以及 `openai_router.rs`），
对 vLLM 的 APIServer（`host:port`）发起一次普通 HTTP POST。vLLM 服务端收到带
`X-data-parallel-rank: N` 的请求后，把请求派到其内部第 N 个 DP engine。

> **小结（回答两个核心问题）**
> 1. **如何与 vLLM 服务端交互**：vllm-router 不做特殊协议，就是标准的 HTTP 反向代理
>    （`reqwest` POST 到 `http://host:port`，适配 `/v1/chat/completions`、`/v1/completions`、
>    `/generate` 等 OpenAI / vLLM 端点），支持流式 SSE、健康检查、熔断、重试等。
> 2. **如何转发到不同 DP rank**：`--intra-node-data-parallel-size N`（大于 1 自动启用），
>    router 把 base URL 展开成 N 个 `@0..@N-1` 的 `DPAwareWorker`，用负载均衡策略选一个，
>    然后**向同一个 base URL 发请求，并在 `X-data-parallel-rank` 头写入所选 rank**。
>    vLLM 据此把请求派到对应 DP engine。
>
> 注：DP rank 不再放进请求 body（早期实现用 `data_parallel_rank` body 字段），
> 现在统一走 `X-data-parallel-rank` HTTP 头（`worker.rs:601` 注释）。

### 1.3 代码链路（结构总览）

```
外部客户端
   │  X-Session-ID 请求头
   ▼
vllm-router (Rust, vllm_router_rs)
   │  consistent_hash 策略: hash(header:x-session-id) → 映射到某个 DP-aware worker URL
   │    ├─ 单 worker URL → 按 --intra-node-data-parallel-size 扩展为多个 "URL@rank"（DP-aware）
   │    └─ 多节点/超多节点 → --worker-urls 提供多个访问地址，同样按 DP size 展开
   ▼
后端 vllm 服务 (port 9000, DP 2 × TP 2)
   ├─ http://127.0.0.1:9000@0   （DP rank 0）
   └─ http://127.0.0.1:9000@1   （DP rank 1）
```

关键环节（对应日志里的每一步）：

| 环节 | 日志/代码位置（Rust `vllm_router_rs`） | 说明 |
|---|---|---|
| 路由开始 | `src/server.rs:847` Starting router | 启动参数、policy、kv_connector（Nixl） |
| worker 健康检查 | `src/routers/http/router.rs:251/:321` | 等待后台 worker 变为 healthy |
| **DP 展开** | `src/routers/http/dp_utils.rs:34` | 把 `http://host:port` 展开成 `@0/@1` 多个 DP-aware URL |
| 分配策略 | `src/policies/registry.rs:83` | 给 model 分配 `consistent_hash` |
| **哈希取 key** | `src/policies/consistent_hash.rs:359/:364` | 提取 hash key = `header:x-session-id:*` |
| 路由到 worker | `src/policies/consistent_hash.rs:413` | key → 固定 worker（`index=0/1…`） |
| 就绪/启动 | `src/server.rs:1008/:1037` | Router ready + 对外监听 |

> 补充：`intra-node-data-parallel-size` 告诉 router「它连的每一个 worker URL，内部实际上暴露了几个 DP rank」；
> router 据此把该 URL 展开成 `worker@0 … worker@N-1` 的 **DP-aware 集合**，再参与 consistent_hash 环。

### 1.4 安装

```bash
pip install vllm-router
```

### 1.5 启动 vllm 服务端（多 DP 场景）

```bash
vllm serve /home/data/Qwen3-30B-A3B \
  --max_model_len 8192 \
  --safetensors-load-strategy 'prefetch' \
  --served-model-name auto \
  --gpu-memory-utilization 0.9 \
  --enable-expert-parallel \
  --async-scheduling \
  --max-num-seqs 48 \
  --port 9000 \
  --data-parallel-size 2 \
  --tensor-parallel-size 2 \
  --api_server_count 1 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
```

要点：
- `--port 9000`：后端服务暴露端口（router 通过 `--worker-urls` 指向它）。
- `--data-parallel-size 2`：单节点起 2 个 DP rank，供 router 按
  `--intra-node-data-parallel-size 2` 展开。
- `--tensor-parallel-size 2`：每个 rank 内部再切 2 路张量并行。

### 1.6 启动 vllm-router，对接服务端

```bash
vllm-router \
  --host 127.0.0.1 \
  --port 30000 \
  --worker-urls http://127.0.0.1:9000 \
  --policy consistent_hash \
  --intra-node-data-parallel-size 2 \
  --prometheus-port 29000
```

参数说明：

| 参数 | 作用 |
|---|---|
| `--host` / `--port` | vllm-router **对外暴露**的端口；外部使用服务需切换到该地址 |
| `--worker-urls` | 各个 worker 的暴露端口。单节点 `http://127.0.0.1:8000/`；多节点 `http://node0:8000 http://node1:8000`；超多节点 `${WORKERS}`（配置各自访问地址+端口清单）。均由这一个配置解决 |
| `--intra-node-data-parallel-size` | 每个 worker URL 内部实际暴露的 DP rank 数 |
| `--policy` | 路由策略，此处 `consistent_hash` |
| `--prometheus-port` | 暴露 metrics 的端口；可查询 router 往哪些 DP 实例转发了请求 |

### 1.7 效果验证

通过 `--prometheus-port` 暴露 metrics，用测试脚本对比不同 `X-Session-ID`
请求在 DP 之间是如何被分发的（看 `vllm_router_processed_requests_total` 的增量）。

#### 测试脚本 `test_router.sh`

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

# vllm-router 的请求地址和 Prometheus metrics 地址。
ROUTER="${ROUTER:-http://127.0.0.1:30000}"
METRICS="${METRICS:-http://127.0.0.1:29000}"

# 根据现场模型名称修改。
MODEL="${MODEL:-auto}"
N="${N:-40}"

# 从 metrics 中读取每个 DP-aware worker 已处理的请求数。
# 输出示例：
#   http://127.0.0.1:8000@0 100
#   http://127.0.0.1:8000@1 120
snapshot() {
  curl -fsS "${METRICS}/metrics" |
    awk '
      /^vllm_router_processed_requests_total\{/ {
        if (match($0, /worker="[^"]*"/)) {
          worker = substr($0, RSTART + 8, RLENGTH - 9)
          print worker, $NF
        }
      }'
}

# 发送一次请求。
# $1 是本次请求使用的 X-Session-ID。
send_one() {
  local session_id="$1"

  curl -fsS -o /dev/null \
    -H "Content-Type: application/json" \
    -H "X-Session-ID: ${session_id}" \
    -X POST "${ROUTER}/v1/completions" \
    -d "{
      \"model\": \"${MODEL}\",
      \"prompt\": \"Once upon a time\",
      \"max_tokens\": 1,
      \"stream\": false
    }"
}

# 对比请求前后的 metrics，打印每个 DP-aware worker 的请求增量。
report_delta() {
  local before="$1"
  local after="$2"

  awk -v before="$before" -v after="$after" '
    BEGIN {
      before_count = split(before, before_lines, "\n")
      for (i = 1; i <= before_count; i++) {
        split(before_lines[i], fields, " ")
        old[fields[1]] = fields[2]
      }

      after_count = split(after, after_lines, "\n")
      for (i = 1; i <= after_count; i++) {
        split(after_lines[i], fields, " ")
        new[fields[1]] = fields[2]
      }

      for (worker in new) {
        delta = new[worker] - old[worker]
        if (delta > 0) {
          printf "%-45s %d\n", worker, delta
        }
      }
    }'
}

echo "Scenario 1: 所有请求使用相同 X-Session-ID"
before="$(snapshot)"
for ((i = 0; i < N; i++)); do
  send_one "same-episode-123"
done
after="$(snapshot)"
report_delta "$before" "$after"

echo
echo "Scenario 2: 每个请求使用不同 X-Session-ID"
before="$(snapshot)"
for ((i = 0; i < N; i++)); do
  # $$ 是当前脚本的进程号；与 i 组合后，本轮测试里的 session id 唯一。
  send_one "episode-$$-${i}"
done
after="$(snapshot)"
report_delta "$before" "$after"
```

#### 测试结果

```
[root@node2 l00606955]# bash test_router.sh
Scenario 1: 所有请求使用相同 X-Session-ID
http://127.0.0.1:9000@0                       40

Scenario 2: 每个请求使用不同 X-Session-ID
http://127.0.0.1:9000@0                       18
http://127.0.0.1:9000@1                       22
```

结果解读：
- **Scenario 1（相同 Session-ID）**：40 个请求全部路由到 `9000@0` —— 说明 **consistent_hash
  把同一 session 钉在同一个 DP rank**，保证 KV Cache 在该 rank 内复用。
- **Scenario 2（不同 Session-ID）**：40 个请求分散到 `@0`(18) / `@1`(22) —— **不同 session 按哈希散布**，
  实现 DP 间的负载均衡。

#### vllm-router 日志示例

```log
DEBUG: Server startup function called
DEBUG: Initializing logging
DEBUG: Logging initialized
DEBUG: Initializing Prometheus metrics
DEBUG: Prometheus metrics initialized
2026-09-20 08:28:07  INFO vllm_router_rs::server: src/server.rs:847: Starting router on 127.0.0.1:30000 | mode: Regular { worker_urls: ["http://127.0.0.1:9000"] } | policy: ConsistentHash { virtual_nodes: 160 } | kv_connector: Nixl | max_payload: 512MB
DEBUG: Creating HTTP client
DEBUG: HTTP client created
DEBUG: Creating AppContext
DEBUG: AppContext created
2026-09-20 08:28:07  INFO vllm_router_rs::server: src/server.rs:945: Single router mode (enable_igw=false)
2026-09-20 08:28:07  INFO vllm_router_rs::routers::http::router: src/routers/http/router.rs:251: Waiting for 1 unique hosts (representing 1 workers) to become healthy (timeout: 600s)
2026-09-20 08:28:07  INFO vllm_router_rs::routers::http::router: src/routers/http/router.rs:321: 1 out of 1 unique hosts are healthy (representing 1 workers)
2026-09-20 08:28:07  INFO vllm_router_rs::routers::http::dp_utils: src/routers/http/dp_utils.rs:34: Expanding worker http://127.0.0.1:9000 to 2 DP-aware URLs (ranks 0..1)
2026-09-20 08:28:07  INFO vllm_router_rs::policies::registry: src/policies/registry.rs:83: Assigning policy consistent_hash to new model unknown
2026-09-20 08:28:07  INFO vllm_router_rs::server: src/server.rs:957: Started health checker for workers with 60s interval
2026-09-20 08:28:07  INFO vllm_router_rs::server: src/server.rs:972: Started request queue with size: 100, timeout: 60s
2026-09-20 08:28:07  INFO vllm_router_rs::middleware: src/middleware.rs:421: Starting concurrency queue processor
2026-09-20 08:28:07  INFO vllm_router_rs::server: src/server.rs:1008: Router ready | workers: ["http://127.0.0.1:9000@0", "http://127.0.0.1:9000@1"]
2026-09-20 08:28:07  INFO vllm_router_rs::server: src/server.rs:1037: Starting server on 127.0.0.1:30000
2026-09-20 08:28:18  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:282: Updated consistent hash ring with 2 workers and 320 virtual nodes
2026-09-20 08:28:18  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:359: CONSISTENT_HASH_DEBUG: Extracted hash key: header:x-session-id:same-episode-123
2026-09-20 08:28:18  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:364: CONSISTENT_HASH_DEBUG: Hash key 'header:x-session-id:same-episode-123' mapped to worker: http://127.0.0.1:9000@0
2026-09-20 08:28:18  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:413: Consistent hash routing: key='header:x-session-id:same-episode-123' -> worker='http://127.0.0.1:9000@0' (index=0)
```

---

## 2. sglang model gateway 使能

> todo：补充 sglang model gateway 的代码链路与测试验证。

### 2.1 代码链路

todo

### 2.2 安装

```bash
pip install sglang-router
```

### 2.3 启动与配置

todo

### 2.4 测试验证

todo
