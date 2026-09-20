vllm-router使能

代码链路

验证
安装vllm-router
pip install vllm-router
启动vllm服务端，多DP场景
vllm serve /home/data/Qwen3-30B-A3B   --max_model_len 8192   --safetensors-load-strategy 'prefetch'   --served-model-name auto   --gpu-memory-utilization 0.9   --enable-expert-parallel   --async-scheduling   --max-num-seqs 48   --port 9000   --data-parallel-size 2   --tensor-parallel-size 2   --api_server_count 1   --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'
启动vllm-router，对接服务端
vllm-router \
  --host 127.0.0.1 \
  --port 30000 \
  --worker-urls http://127.0.0.1:9000 \
  --policy consistent_hash \
  --intra-node-data-parallel-size 2 \
  --prometheus-port 29000
其中
1. --host和--port指明vllm-router对外暴露的端口信息，外部使用服务需要切换到这个地址
2. --worker-urls填写各个worker的暴露端口信息，这里涉及到单节点（--worker-urls http://127.0.0.1:8000/）、多节点（--worker-urls http://node0:8000 http://node1:8000）、超多节点（--worker-urls ${WORKERS}，配置各自的访问地址和端口清单）场景，均由这一个配置解决。
3. intra-node-data-parallel-size指明它所连接的每一个后端 worker URL，在内部实际上暴露了多少个DP rank
4. 如果需要进行metric信息查询，比如需要查询到当前vllm-router往哪些dp示例路由转发了请求，可以通过--prometheus-port来指定metrics的对外暴露端口

效果验证
通过--prometheus-port暴露metrics，通过测试脚本测试不同session-id的请求在DP间是如何控制分发的
脚本
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

测试结果
[root@node2 l00606955]# bash test_router.sh
Scenario 1: 所有请求使用相同 X-Session-ID
http://127.0.0.1:9000@0                       40

Scenario 2: 每个请求使用不同 X-Session-ID
http://127.0.0.1:9000@0                       18
http://127.0.0.1:9000@1                       22

vllm-router日志示例：
[root@node2 ds-v3.2]# vllm-router   --host 127.0.0.1   --port 30000   --worker-urls http://127.0.0.1:9000   --policy consistent_hash   --intra-node-data-parallel-size 2 --prometheus-port 29000
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
2026-09-20 08:28:18  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:359: CONSISTENT_HASH_DEBUG: Extracted hash key: header:x-session-id:same-episode-123
2026-09-20 08:28:18  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:364: CONSISTENT_HASH_DEBUG: Hash key 'header:x-session-id:same-episode-123' mapped to worker: http://127.0.0.1:9000@0
2026-09-20 08:28:18  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:413: Consistent hash routing: key='header:x-session-id:same-episode-123' -> worker='http://127.0.0.1:9000@0' (index=0)
2026-09-20 08:28:18  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:359: CONSISTENT_HASH_DEBUG: Extracted hash key: header:x-session-id:same-episode-123
2026-09-20 08:28:18  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:364: CONSISTENT_HASH_DEBUG: Hash key 'header:x-session-id:same-episode-123' mapped to worker: http://127.0.0.1:9000@0
2026-09-20 08:28:18  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:413: Consistent hash routing: key='header:x-session-id:same-episode-123' -> worker='http://127.0.0.1:9000@0' (index=0)
2026-09-20 08:28:18  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:359: CONSISTENT_HASH_DEBUG: Extracted hash key: header:x-session-id:same-episode-123
2026-09-20 08:28:18  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:364: CONSISTENT_HASH_DEBUG: Hash key 'header:x-session-id:same-episode-123' mapped to worker: http://127.0.0.1:9000@0
2026-09-20 08:28:18  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:413: Consistent hash routing: key='header:x-session-id:same-episode-123' -> worker='http://127.0.0.1:9000@0' (index=0)
2026-09-20 08:28:18  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:359: CONSISTENT_HASH_DEBUG: Extracted hash key: header:x-session-id:same-episode-123
2026-09-20 08:28:18  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:364: CONSISTENT_HASH_DEBUG: Hash key 'header:x-session-id:same-episode-123' mapped to worker: http://127.0.0.1:9000@0
2026-09-20 08:28:18  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:413: Consistent hash routing: key='header:x-session-id:same-episode-123' -> worker='http://127.0.0.1:9000@0' (index=0)
2026-09-20 08:28:19  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:359: CONSISTENT_HASH_DEBUG: Extracted hash key: header:x-session-id:same-episode-123
2026-09-20 08:28:19  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:364: CONSISTENT_HASH_DEBUG: Hash key 'header:x-session-id:same-episode-123' mapped to worker: http://127.0.0.1:9000@0
2026-09-20 08:28:19  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:413: Consistent hash routing: key='header:x-session-id:same-episode-123' -> worker='http://127.0.0.1:9000@0' (index=0)
2026-09-20 08:28:19  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:359: CONSISTENT_HASH_DEBUG: Extracted hash key: header:x-session-id:same-episode-123
2026-09-20 08:28:19  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:364: CONSISTENT_HASH_DEBUG: Hash key 'header:x-session-id:same-episode-123' mapped to worker: http://127.0.0.1:9000@0
2026-09-20 08:28:19  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:413: Consistent hash routing: key='header:x-session-id:same-episode-123' -> worker='http://127.0.0.1:9000@0' (index=0)
2026-09-20 08:28:19  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:359: CONSISTENT_HASH_DEBUG: Extracted hash key: header:x-session-id:same-episode-123
2026-09-20 08:28:19  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:364: CONSISTENT_HASH_DEBUG: Hash key 'header:x-session-id:same-episode-123' mapped to worker: http://127.0.0.1:9000@0
2026-09-20 08:28:19  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:413: Consistent hash routing: key='header:x-session-id:same-episode-123' -> worker='http://127.0.0.1:9000@0' (index=0)
2026-09-20 08:28:19  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:359: CONSISTENT_HASH_DEBUG: Extracted hash key: header:x-session-id:same-episode-123
2026-09-20 08:28:19  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:364: CONSISTENT_HASH_DEBUG: Hash key 'header:x-session-id:same-episode-123' mapped to worker: http://127.0.0.1:9000@0
2026-09-20 08:28:19  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:413: Consistent hash routing: key='header:x-session-id:same-episode-123' -> worker='http://127.0.0.1:9000@0' (index=0)
2026-09-20 08:28:19  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:359: CONSISTENT_HASH_DEBUG: Extracted hash key: header:x-session-id:same-episode-123
2026-09-20 08:28:19  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:364: CONSISTENT_HASH_DEBUG: Hash key 'header:x-session-id:same-episode-123' mapped to worker: http://127.0.0.1:9000@0
2026-09-20 08:28:19  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:413: Consistent hash routing: key='header:x-session-id:same-episode-123' -> worker='http://127.0.0.1:9000@0' (index=0)
2026-09-20 08:28:19  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:359: CONSISTENT_HASH_DEBUG: Extracted hash key: header:x-session-id:same-episode-123
2026-09-20 08:28:19  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:364: CONSISTENT_HASH_DEBUG: Hash key 'header:x-session-id:same-episode-123' mapped to worker: http://127.0.0.1:9000@0
2026-09-20 08:28:19  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:413: Consistent hash routing: key='header:x-session-id:same-episode-123' -> worker='http://127.0.0.1:9000@0' (index=0)
2026-09-20 08:28:20  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:359: CONSISTENT_HASH_DEBUG: Extracted hash key: header:x-session-id:same-episode-123
2026-09-20 08:28:20  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:364: CONSISTENT_HASH_DEBUG: Hash key 'header:x-session-id:same-episode-123' mapped to worker: http://127.0.0.1:9000@0
2026-09-20 08:28:20  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:413: Consistent hash routing: key='header:x-session-id:same-episode-123' -> worker='http://127.0.0.1:9000@0' (index=0)
2026-09-20 08:28:20  INFO vllm_router_rs::policies::consistent_hash: src/policies/consistent_hash.rs:359: CONSISTENT_HASH_DEBUG: Extracted hash key: header:x-session-id:same-episode-123



sglang model gateway使能

代码链路

测试验证
安装sglang-router
pip install sglang-router