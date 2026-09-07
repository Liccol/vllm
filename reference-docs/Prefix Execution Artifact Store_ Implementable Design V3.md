# Prefix Execution Artifact Store: Implementable Design V3



Date: 2026\-07\-26



Status: Rewritten according to the latest agreed design\. The current phase implements only R3 and two Artifact Connector backends: SHM and direct Mooncake\. The SHM rewrite, direct Mooncake fake\-client validation, and H200 TP1/TCP E2E against a real master have been completed\. TP2, RDMA, and failure\-path tests remain explicit acceptance gates\.



## 0\. Document Navigation



This document defines the overall architecture, public contract, data model, module boundaries, and core protocols\. Topic\-specific documents:



- See \[Data Plane and Performance Analysis\]\(\./prefix\-artifact\-data\-plane\-performance\.md\) for GPU/CPU rings, PCIe/RDMA traffic, and interference with inference\.

- See \[Writer Topology\]\(\./prefix\-artifact\-writer\-topology\.md\) for multi\-rank writer ownership, aggregation, and the ready protocol\.

- See \[Feature and Parallelism Compatibility\]\(\./prefix\-artifact\-compatibility\.md\) for parallelism modes, feature support, and startup guards\.

- See \[Storage Interface Survey\]\(\./mooncake\-interface\-survey\.md\) for source\-level facts about the Mooncake API and its reuse boundaries\.

- See \[Direct Mooncake R3 \+ Logprobs E2E\]\(\./mooncake\-r3\-logprobs\-e2e\.md\) for results from a real master, an independent consumer, SHM parity, and numerical logprobs validation\.

- See \[Roadmap\]\(\./prefix\-execution\-artifact\-store\-roadmap\.md\) for the implementation and PR sequence\.

    

## 1\. Current Scope



### 1\.1 Baseline



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



Current stacked PRs:



- [aoshen02/vllm\#4](https://github.com/aoshen02/vllm/pull/4): ordered\-key Artifact Core \+ SHM;

- [aoshen02/vllm\#7](https://github.com/aoshen02/vllm/pull/7): direct Mooncake\.

    

PR4 and PR7 have been rewritten as a compact stack\. PR7 retains the critical fix found by the real\-backend E2E test: Mooncake 0\.3\.10 requires the exact object length for a registered\-buffer get\. The store calls `get_size(key)` internally and packs the batch using the actual lengths, while the public caller still passes only keys\.



Key source references:



- [vllm\-project/vllm\#45635](https://github.com/vllm-project/vllm/pull/45635): physical\-slot R3 capture and CPU\-offload baseline;

- [vllm\-project/vllm\#39568](https://github.com/vllm-project/vllm/pull/39568): reference semantics for scheduler accepted ranges and speculative overwrite;

- vLLM `vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py`: reference for the Mooncake client, registered buffers, batch I/O, and lifecycle\.

    

### 1\.2 Deliverables for This Phase



Only the following are delivered:



- R3 capture in Model Runner V2;

- R3 full\-block prefix reuse;

- SHM simple backend;

- direct Mooncake production backend;

- joint KV \+ R3 prefix readiness;

- ordered artifact\-key consumer contract\.

    

The following are not delivered:



- logprobs, DSA, multimodal data, or Top\-p token IDs;

- VIME dataloader;

- multi\-writer, PP, DCP/PCP, and other subsequent topologies\.

    

### 1\.3 Deployment Assumptions and Boundaries



- A deployment uses a consistent model, tokenizer, parallel configuration, artifact block size, and field profile\. Incompatible configurations do not share a key namespace\.

- An engine does not mix ordinary serving requests and artifact rollout requests\. Once Artifact Connector is enabled, every request that may enter the prefix cache must produce all mandatory fields required by the deployment\.

- An engine enables exactly one backend at a time: `shm` or `mooncake`; it never dual\-writes\.

- The current version does not implement a policy epoch/fence state machine\. In\-place weight updates fail closed when the R3 Artifact Connector is enabled\. Weight changes require draining and restarting the engine\. If hot updates within one engine are needed later, a generation/epoch contract must be designed separately\.

- When Artifact Connector is disabled, the engine creates no capture state, SHM store, Mooncake client, or staging buffer, and constructs no artifact metadata on ordinary requests or steps\.

    

The following remain request/control\-plane metadata and are not written to the ArtifactStore:



- reward, tool result, finish reason, and group metadata;

- loss mask and consumer lease/ack;

- scheduler request status, commit/finalize commands, and worker ACKs\.

    

Other producers such as value and reward models may reuse the object/store protocol in the future, but they must use their own producer/model namespaces and retention policies rather than being coupled to policy\-KV invalidation rules\.



### 1\.4 Preserved Model for Future Fields



Only R3 is implemented in this phase, but future extensions reuse the same lifecycle:



- Artifact Connector integrates only with MRV2\. MRV1 retains the upstream R3 behavior and is not connected to this store lifecycle\.

- `PREFIX_BLOCK`: R3, DSA, and one unified `logprobs` field\. Enabling `prompt_logprobs` merely extends prompt coverage for that field; it does not create a second prompt\-logprobs artifact\. When an artifact backend is enabled, logprobs are also read from that backend, and HTTP does not return a duplicate complete logprobs value\.

- `REQUEST_ONLY`: multimodal data and Top\-p token IDs, which do not participate in cross\-request prefix reuse\.

- All fields share `ArtifactRequestCore`, the backend object contract, and finalize\. A field adapter describes only coordinates, shape, and codec\.

- Multiple mandatory fields use separate per\-field objects/key lists but share one completion barrier\. If any required field fails, the request returns no production handle\.

    

Upstream DSA implementation tracking:

[vllm\-project/vllm\#47279](https://github.com/vllm-project/vllm/pull/47279)\.



### 1\.5 Existing Baseline Evidence



These results establish the historical R3 capture/Core baseline; they do not mean that direct Mooncake has passed acceptance:



- H200 TP8: exact equality between cold HTTP R3 from the old \#45635 and the fixed version, shape `[1016, 32, 8]`, `uint8`, with all 40 experts represented;

- local prefix cache: exact 992\-token R3 prefix;

- CPU KV offload: exact 992\-token external R3 prefix;

- R3 \+ prompt\-logprobs prototype: shape `[372, 32, 8]`, 352\-token local/external prefix hit, with exact cold/hot logprobs and cached R3;

- latest stacked head: H200 TP1/TP2 MRV2 async cold/full\-hit, with an exact 64\-token cached R3 prefix\.

    

See \[PR3 validation\]\(\./final\_validation\.md\) for the complete historical commands and environment records\.



## 2\. Overall Architecture



### 2\.1 vLLM Process Boundaries



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
│ - executed boundary                                          │
│ - KV-compatible block hashes                                 │
│ - joint KV + R3 readiness                                    │
│ - commit/finalize ordering                                   │
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



### 2\.2 Inside Artifact Connector



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



`connector.py` only orchestrates scheduler/worker metadata\. Keys, coverage, ordered references, tails, and materialization live exclusively in `request_core.py`\.



### 2\.3 End\-to\-End Consumer Path



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



The consumer does not write dataset/sample state back to vLLM\. The SHM path consumes actual R3 directly; the Mooncake path reads objects using the HTTP keys\. The final tensors from both paths must be element\-wise identical\.



## 3\. Capture vs\. ArtifactStore



Artifact mode reuses the router capture hook and stable GPU snapshot from `#45635`, but not its physical\-slot mmap:



```Plain Text
GPU router IDs
  -> authoritative rank-0 worker
  -> current-step stable GPU snapshot
  -> async D2H
  -> per-request logical suffix buffer
  -> ArtifactRequestCore
```



The two layers have different meanings:



|Layer|Identity|Lifetime|Purpose|
|---|---|---|---|
|worker logical buffer|request \+ logical token range|release a full block after commit; discard the tail at terminal/abort|accepted\-row staging|
|ArtifactStore|logical KV hash \+ field profile|store retention|prefix reuse/consumer|



Therefore:



- Artifact Connector performs no second GPU capture;

- a physical slot cannot be an artifact key;

- both SHM and Mooncake publish logical objects;

- an artifact write task permits only one destination and no longer writes a physical\-slot mmap at the same time;

- because the SHM simple backend itself uses `/dev/shm`, it requires the writer and EngineCore to be colocated;

- the Mooncake production backend does not depend on EngineCore `/dev/shm`; cross\-node writer/EngineCore placement still requires deployment validation, but no longer has a shared\-mmap architectural constraint\.

    

## 4\. Public Contract



### 4\.1 SHM Backend



SHM targets simple, immediate\-return usage:



```JSON
{
  "routed_experts": "<base64 npy>",
  "artifact_keys": null
}
```



- The caller does not read `/dev/shm`\.

- The caller receives no path, mmap handle, or internal object key\.

- The worker materializes actual R3 through the unified Core\.

- Actual R3 continues through EngineCore/API IPC\.

    

### 4\.2 Mooncake Backend



Mooncake targets production consumers:



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



`artifact_keys`:



- is an ordered `list[str]`;

- each key represents a contiguous range of logical R3 tokens;

- full\-block keys are reusable across requests;

- the final partial block uses a request\-scoped tail key;

- is returned only after all object puts have completed;

- is not a Mooncake segment/replica handle;

- does not contain an `artifact_sample_id`\.

    

### 4\.3 Key\-List Length



Let:



```Plain Text
L = number of tokens that actually executed forward and whose R3 must be returned
B = artifact hash block size
```



Then:



```Plain Text
num_full_blocks = floor(L / B)
has_tail = (L % B) != 0

len(artifact_keys)
  = num_full_blocks + int(has_tail)
  = ceil(L / B)
```



It is therefore the number of logical blocks covering this request's R3, and does not exceed the upper bound given by the request's KV block table\.



The last sampled token, which has not yet executed a forward pass, is excluded from `L`\.



### 4\.4 API Delivery



- Non\-streaming: each terminal choice returns its own actual R3 or ordered keys\.

- Streaming: artifacts are returned exactly once in the terminal chunk; intermediate chunks contain no partial artifacts\.

- Batch/`n > 1`: each child request independently computes `L`, its tail, and its key list; child requests may share identical full\-block keys\.

- A backend failure returns no partial key list and does not misrepresent an otherwise terminal success as an artifact success that can be consumed\.

    

## 5\. R3 Data Model



### 5\.1 Logical Coordinate



R3 uses:



```Plain Text
reuse_policy       = PREFIX_BLOCK
logical_coordinate = EXECUTED_TOKEN
shape              = [executed_tokens, num_moe_layers, topk]
```



### 5\.2 Forward Alignment



Field alignment for one `forward(token_i)` must be explicitly fixed:



```Plain Text
KV / R3 / DSA of token_i  <- forward(token_i)
logprob(token_i)          <- forward(token_{i-1})
```



Therefore:



- R3/DSA use the `EXECUTED_TOKEN` coordinate;

- logprobs use the `PREDICTED_TOKEN` coordinate;

- the first prompt token has no preceding row;

- the last token that was only sampled and has not been used as forward input has no R3/DSA;

- speculative rejected rows, rows after stop trimming, and async in\-flight rows must never enter the committed range;

- R3 stores global expert IDs, DSA stores logical top\-k positions, and the logprobs profile must distinguish raw and processed modes\.

    

### 5\.3 Full\-Block Key



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



It does not include a request ID\. The same model/profile/prefix produces the same key\. A weight change currently creates a new deployment namespace by drain/restart; the scheduler does not maintain a policy epoch\.



### 5\.4 Partial\-Tail Key



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



The tail:



- belongs only to the current request;

- does not participate in prefix lookup;

- is not returned on abort;

- expires through the backend's natural eviction policy\.

    

### 5\.5 Object Envelope



Mooncake and SHM objects use the same logical header:



```Plain Text
uint32 header_length
canonical JSON header
raw contiguous tensor bytes
```



The header contains at least:



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



Because every object self\-describes its dtype, shape, valid length, and checksum, no request manifest is required\.



A Mooncake get requires the caller to supply a registered receive buffer\. Mooncake 0\.3\.10's `batch_get_into_multi_buffers` does not accept maximum capacity in place of the actual object length: the real E2E test returns `TRANSFER_FAIL (-800)`\. `MooncakeArtifactStore.get(keys)` must call `get_size(key)` internally, verify that the length does not exceed the field\-profile limit, and tightly pack multiple objects into the registered staging buffer using their exact lengths\. Public HTTP and `materialize_routed_experts(store, keys)` still take only ordered keys; neither full blocks nor tails require a request manifest or public length list\.



## 6\. Request Core



`ArtifactRequestCore` stores only immutable dependencies:



```Plain Text
store
writer
namespace / field profile
materialize mode
```



It is responsible for:



- encoding scheduler\-confirmed executed ranges as full blocks;

- keys/profile;

- re\-deriving the ordered key list from all KV hashes during finalize;

- verifying that all full blocks exist and publishing the request\-scoped tail;

- backend puts and simple\-mode materialization\.

    

It is not responsible for:



- the GPU capture hook;

- scheduler admission;

- the Mooncake master lifecycle;

- HTTP serialization;

- an external dataset/sample state machine\.

    

The following are no longer needed:



- public `artifact_sample_id`;

- request manifest;

- any public handle other than `artifact_keys`;

- final\-notify control row;

- worker\-side request assembly, segment tracking, or discard RPC;

- policy epoch/update fence state machine\.

    

## 7\. Publication Protocol



### 7\.1 Full Blocks



Whenever the accepted/executed boundary completes a new full block:



```Plain Text
1. The scheduler sends commits only for newly completed full ranges.
2. The worker reads the current step's stable logical buffer and creates
   content-addressed objects.
3. The backend skips the put for an immutable key that already exists.
4. Missing objects are written with a batch put.
5. SHM ACKs after synchronous publication; Mooncake ACKs after immutable bytes
   enter the bounded publisher.
6. Prefix admission still uses a backend existence lookup. A Mooncake enqueue
   ACK is not treated as ready.
```



Only newly completed full blocks are published\. New blocks from multiple requests in the same step are combined into a batch, but results are accounted for independently per object: a failed item must neither mark other items in its batch as ready nor allow a request to return a key list that references it\.



### 7\.2 Finalize



```Plain Text
1. The scheduler obtains the authoritative executed end after stop/abort/spec
   acceptance.
2. It sends finalize through normal SchedulerOutput metadata.
3. Full-block keys are deterministically derived from KV hashes.
4. If a partial tail exists, SHM puts it synchronously; Mooncake enqueues its
   stable bytes.
5. The worker returns a request-ID-bound finalize ACK.
6. After validating the ACK, the scheduler produces an ordered
   list[artifact_key].
7. SHM: materialize actual R3.
8. Mooncake: return the key list.
9. Release the retained terminal HTTP output.
```



The Mooncake key list is itself the complete handle\. HTTP may return after every object has reached PutStart/enqueue, without waiting for PutEnd/ready; the consumer polls `batch_is_exist` for each key\. No request manifest or additional final notification is required\.



Under async scheduling, the implementation must not directly use a count that includes the next frame's in\-flight tokens\. The speculative path must use the boundary after stop trimming and acceptance\. Finalize may occupy the next zero\-token control step, but the worker must not infer EOS/stop itself because it cannot see frontend stop strings, aborts, or all scheduler state\.



### 7\.3 Abort / Preemption



- Abort/cancel produces neither a tail nor an HTTP key list\.

- The scheduler removes local request state and sends no discard RPC\.

- Already committed full blocks remain\.

- Full blocks are not bound to a request lifetime\.

- Retention/eviction is managed naturally by the backend\.

    

Preemption preserves already published immutable full blocks\. Accepted rows that have not yet been published are copied into the request\-scoped logical buffer after the current step's D2H\. The scheduler releases and reuses physical KV blocks according to the standard KV lifecycle; it maintains no artifact\-specific snapshot or retention\. If recomputation covers the same logical position, the buffer is overwritten with the newly accepted rows, while an already committed prefix is ignored\.



### 7\.4 Put Failures



- SHM reports success only after its synchronous put succeeds\.

- Mooncake may return keys after enqueue/PutStart, but prefix admission must not treat them as ready\.

- The publisher owns immutable encoded bytes, so the original physical KV slot and logical capture buffer may be released immediately\. The background put still owns the registered staging buffer until the Mooncake call returns\.

- Only explicitly retryable transport failures receive bounded retries and backoff\.

- A background failure after PutStart records a fatal error, and subsequent publication fails closed\. Already returned keys remain not ready and are handled by consumer polling timeout/error\.

- Successfully published immutable full blocks are not rolled back; a failed tail naturally becomes an unreferenced object\.

- Backend retention/GC reclaims unreferenced objects left by crashes\.

    

## 8\. Prefix Cache \+ R3



### 8\.1 Joint Readiness



The hit condition is:



```Plain Text
KV block ready
AND corresponding R3 block key ready
```



Mooncake mode cannot trust only a scheduler\-local ACK catalog because:



- Mooncake objects expire naturally;

- another vLLM instance may already have published the object;

- the local catalog is empty after an engine restart\.

    

The current implementation maintains no cross\-request ready catalog\. `MooncakeArtifactReader` in EngineCore calls `batch_is_exist` directly\. The reader registers no staging buffer; only the rank\-0 GPU worker's writer store registers staging, whose default size is 64 MiB\.



Correct admission:



```Plain Text
KV candidate prefix
  -> derive ordered R3 block keys
  -> Mooncake batch_is_exist(keys)
  -> longest KV-ready AND R3-ready prefix
```



The KV\-only candidate is not yet a prefix hit\. The scheduler recognizes a hit only after joint lookup:



- A candidate that is not yet ready may wait or be treated as a joint miss and re\-executed\.

- There is no valid state in which the scheduler has recognized a KV hit but mandatory R3 is absent\.

- After the scheduler has recognized a hit, missing/corrupt/profile\-mismatched data is a consistency error\. The implementation must not silently shorten the hit or recompute it\.

    

### 8\.2 Hit Path



After a joint hit is accepted:



- the cached span is not forwarded again;

- cached R3 is not captured again;

- no block put is called;

- the request binds to the existing keys;

- only newly completed blocks are put incrementally\.

    

```Plain Text
request A -> [K0, K1, K2, TailA]
request B -> [K0, K1, K2, TailB]
```



### 8\.3 Eviction Race



- Missing before joint admission: the candidate has not become a hit and may wait or re\-execute\.

- Missing after admission but before finalize: fail closed\.

- Object evicted after HTTP returns keys: consumer get returns not found, governed by the deployment retention SLA\.

    

The same artifact\-enabled engine must not allow an ordinary request to warm only KV and then call that KV\-only state a hit for an artifact request\. If the implementation observes a split between already accepted KV and artifact readiness, it must report an error and invalidate the affected cache entry\.



## 9\. SHM Backend



The SHM store provides:



- a trusted `/dev/shm` namespace;

- content\-addressed full blocks;

- atomic temporary\-file \+ rename publication;

- checksums;

- a capacity limit;

- TTL/lease cleanup;

- ordered\-key materialization\.

    

The SHM public contract does not return keys, but internally it still reuses the same ordered references\.



## 10\. Direct Mooncake Backend



### 10\.1 Deployment Ownership



Reuse the deployment convention of vLLM's `MooncakeStoreConnector`:



```Bash
mooncake_master --port 50051
export MOONCAKE_CONFIG_PATH=/path/to/mooncake_config.json
```



Artifact Connector:



- does not start or stop the master;

- does not modify the Mooncake deployment;

- has the authoritative rank\-0 GPU worker create `MooncakeDistributedStore`;

- reuses the master, protocol, segment, and local\-buffer settings from the configuration;

- releases only the current process's store/TransferEngine on close\.

    

### 10\.2 Store Operations



Use Mooncake's existing interfaces:



```Plain Text
batch_is_exist
batch_put_from_multi_buffers
batch_get_into_multi_buffers
register_buffer
close
```



The logical object key generated by Artifact Core is used directly as the Mooncake key\.



### 10\.3 Registered Staging Buffer



`MooncakeArtifactStore` maintains a bounded registered CPU staging pool:



```Plain Text
encode object
  -> acquire slot
  -> copy bytes
  -> batch_put_from_multi_buffers
  -> transfer complete
  -> release slot
```



Reads work analogously\. `MooncakeArtifactPublisher` places a single writer thread and a bounded one\-batch queue in front of the store:



```Plain Text
encode immutable bytes
  -> bounded enqueue (PutStart / HTTP may return)
  -> background MooncakeArtifactStore.put
  -> PutEnd / object becomes ready
```



A registered staging buffer is not reused until the underlying call returns\. The request's physical KV slot is not part of this lifecycle\. A multi\-slot ring/GDR is a future performance optimization\.



### 10\.4 Duplicate Writers



The first version uses:



```Plain Text
batch_is_exist == 1 -> skip
batch_is_exist == 0 -> put
```



An identical content key implies that a valid payload must be byte\-identical\. Concurrent cold writers may put the same bytes at the same time; the consumer still validates the checksum\.



No lock/control service is introduced for now\. If strict protection against a same\-key race with different payloads becomes necessary, add Mooncake create\-if\-absent/CAS\.



### 10\.5 Retention, GC, and Staleness



- A full block's lifetime is independent of any single request; request completion does not delete a shared block\.

- A tail exists only for the request that returns it and expires according to the backend's natural eviction policy\.

- The current design introduces no consumer ACK/lease control plane\. The deployment must configure a minimum retention SLA long enough for the consumer to complete its get after HTTP returns\.

- The scheduler queries the backend for every admission rather than trusting a stale local catalog\.

- Changes to the field profile/model namespace isolate old objects through a new key namespace\.

- Hot policy updates within the same engine are unsupported; a drain/restart switches the namespace\.

- Under storage pressure, reject new artifact admissions or fail requests explicitly before deleting a source still involved in an in\-flight transfer\.

    

### 10\.6 Crash Semantics



- Worker crash before enqueue: no keys are returned\.

- Worker crash after enqueue/HTTP return but before PutEnd: keys have been returned but remain not ready, and consumer polling times out\. This is an explicit failure mode of the current PutStart contract\.

- Put succeeds but worker crashes before ACK: the object may become a reusable orphan, but it cannot fabricate completion for the current request\.

- EngineCore/API crashes after ACK: published immutable blocks remain, while the original HTTP request fails\.

- Master/client failures propagate as explicit backend errors and must not degrade into a successful response with missing keys\.

    

## 11\. Consumer



Usage:



```Python
r3 = materialize_routed_experts(
    store=MooncakeArtifactStore(...),
    artifact_keys=response.artifact_keys,
)
```



The materializer:



1. batch\-gets the ordered keys;

2. validates object identity, checksum, dtype, shape, and field profile;

3. concatenates full blocks and the tail in order;

4. returns complete R3\.

    

The caller does not manually concatenate tensors\. The key list is a transport handle, not a requirement for users to implement the storage protocol\.



If any required key is not found, cannot be read, or is corrupt, the entire artifact read fails\. It neither returns a shorter R3 nor triggers automatic recomputation by the inference service\.



## 12\. Configuration



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



Mooncake's master address, protocol, device, segment size, and local buffer are read only from `MOONCAKE_CONFIG_PATH`; `ArtifactConfig` does not duplicate them\.



## 13\. Code Layout



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



- `protocol.py`: scheduler/worker IPC dataclasses\.

- `request_core.py`: the only request state machine, plus keys, coverage, envelopes, ordered references, tails, and materialization\.

- `fields.py`: small immutable field descriptors/coordinate adapters\. Future R3, DSA, and logprobs are configurations of the same Core rather than separate per\-field managers/lifecycles\.

- `store.py`: opaque object\-I/O protocol and factory; it does not understand R3 or logprobs\.

- `shm.py` and `mooncake.py`: two backends implementing the same store contract\.

    

Do not create a separate `prompt_logprobs.py`, backend\-specific materializer, Codec inheritance tree, or manager hierarchy\. Prompt\-logprobs\-specific boundary\-hidden capture/restore remains in the sampler path; persistent logprobs still use the unified field adapter/Core\.



## 14\. Current Compatibility Boundaries



Currently allowed:



- CUDA MoE generate;

- MRV2;

- TP;

- a single authoritative writer;

- SHM/direct Mooncake;

- local prefix cache;

- explicitly supported CPU `OffloadingConnector` combinations\.

    

Currently fail closed:



- PP;

- DCP/PCP;

- DBO/executor microbatching;

- elastic EP;

- unsupported routers;

- unsupported KV connectors/PD;

- EC transfer;

- multimodal/encoder\-decoder/diffusion\.

    

See \[the compatibility document\]\(\./prefix\-artifact\-compatibility\.md\) for detailed reasons\.



### 14\.1 Explicitly Accepted Current Tradeoffs



- There is currently only one authoritative rank\-0 writer\. The SHM simple backend requires it to share `/dev/shm` with EngineCore\. The Mooncake production backend has no such colocation requirement, but TP16/cross\-node still requires empirical validation\.

- The first Mooncake version uses one background writer and one pending batch\. HTTP does not wait for PutEnd, but the bounded queue applies backpressure under heavy load\. It does not implement a multi\-slot ring/GDR\.

- There is no create\-if\-absent/CAS\. A cross\-instance cold race relies on deterministic same\-key/same\-bytes behavior; strict collision protection is deferred to the multi\-writer phase\.

- The lifecycle uses natural eviction plus a deployment retention SLA rather than adding a consumer ACK/lease control plane\.

- PP, DCP/PCP, DBO, and unvalidated KV connectors are temporarily guarded for correctness rather than declared permanently incompatible\.

- To return actual R3 immediately, the SHM simple path accepts that terminal payloads pass through ModelRunnerOutput/EngineCore/API IPC\. The production Mooncake path does not do this\.

    

## 15\. Correctness Acceptance



Coverage must include:



- `len(keys) == ceil(executed_tokens / block_size)`;

- an exact\-block request has no tail;

- the last key of a partial request is a tail;

- key order matches the logical token span;

- terminal\-only delivery for non\-streaming/streaming/batch/`n > 1`;

- exact equality between SHM actual R3 and materialization from Mooncake keys;

- cold/full/partial prefix hit;

- the hit path does not call block put;

- missing before joint admission is never recognized as a hit;

- missing/corrupt after admission fails closed;

- an individual batch\-item put failure, exhausted retries, or a worker crash never marks the object ready early;

- async in\-flight rows, stop\-trimmed speculative rows, and rejected rows never cross the executed boundary;

- abort returns no tail/key list, and preemption never reads a released slot;

- in\-place weight update fails closed, and disabled mode has zero overhead;

- retention SLA and eviction/stale local catalog;

- TP1/TP2/TP scale;

- sync/async;

- preemption/abort;

- CPU KV offload;

- Mooncake TCP/RDMA;

- master/worker lifecycle isolation\.

    

## 16\. Design Revision Log



Changes from the old V3:



- The production backend is direct Mooncake\.

- The public handle changes from `artifact_sample_id` to an ordered `list[artifact_key]`\.

- The request manifest is removed\.

- The list length is explicitly `ceil(executed R3 tokens / block_size)`\.

- Full blocks use KV\-compatible content keys\.

- A partial tail uses a request\-scoped key\.

- Keys are returned only after all are ready, so no final notification is required\.

- Mooncake objects are self\-describing, and consumers concatenate them through the unified materializer\.

    

The following are explicit decisions relative to V2 rather than omissions:



- Remove the request manifest because ordered keys plus self\-describing objects are sufficient\.

- Remove the public sample ID and return ordered keys directly over HTTP\.

- Remove the separate final\-notify protocol and reuse finalize metadata plus a worker ACK\.

- Change V2's “the API does not wait for PutEnd” to “return only after all key puts succeed”; otherwise the key list has no ready semantics\.

- Change V2's “inference checks only KV” to “the scheduler recognizes a hit only when KV and mandatory artifacts jointly hit\.”

- Change V2's “combine all fields into one block value” to per\-field immutable objects plus a request\-level completion barrier, avoiding coupling between field codecs/layouts\.

- The V2 selected\-hidden placeholder is not part of the currently confirmed field matrix and is excluded from the Roadmap\. If the requirement is raised again, classify it explicitly as `PREFIX_BLOCK` or `REQUEST_ONLY` before integrating it with the unified Core\.

    

## 17\. Implementation Order



```Plain Text
#45635 + fixes
  -> PR3 MRV2 + R3
  -> PR4 unified Core + SHM
  -> PR7 direct Mooncake + ordered keys
  -> SHM/Mooncake parity
  -> remaining topology/performance matrix
```



## 18\. Legacy\-Design Coverage Audit



|Legacy content|Current location|
|---|---|
|Goals, deployment assumptions, zero overhead when disabled|§1\.2–1\.4|
|Forward/token/field alignment|§5\.1–5\.5|
|vLLM process boundaries, sole writer, simple/production paths|§2–3 and Writer Topology|
|Full\-block batch put, tail finalize, Put/ACK|§7|
|Ring, backpressure, PCIe/RDMA interference|Data Plane and Performance document|
|Manifest/public handle|§4 and §6; manifest explicitly removed|
|Retention, GC, staleness, crash|§7\.3–7\.4 and §10\.5–10\.6|
|Speculative, abort, preemption, KV offload|Compatibility document and §7|
|Scheduler/Worker/Core/backend mapping|§2, §6, and §13|
|Mooncake source interfaces and inflight/buffer ownership|Storage Interface Survey|
|Multi\-rank, PP/CP/DP/EP|Writer Topology and Compatibility documents|
|Experimental data and validation matrix|§1\.5, §15, and Roadmap|



## 10\. Configuration and Fail\-Closed Behavior



With the current baseline and `--enable-return-routed-experts`:



- reject `PP > 1`;

- reject `DCP > 1` or `PCP > 1`;

- allow only `OffloadingConnector + CPUOffloadingSpec + kv_role=kv_both`;

- reject PD and other KV connectors;

- reject configurations with no full\-attention KV group;

- reject monolithic MoE kernels and unknown routers;

- do not blanket\-reject speculative decoding;

- do not reject TP\.

    

One guard gap must still be resolved before release: multiple full\-attention groups currently produce only a warning and select the first group as the anchor\. Unless that selection is proven to cover every logical token that must be reused/offloaded, startup must fail\.



Enablement policy must distinguish between:



- When deployed to test R3 for this project, Artifact Connector \+ SHM must work by default without requiring the user to compose hidden experimental flags\.

- When an ordinary request does not request R3/prompt\-logprobs artifacts, the engine must not initialize capture/store hot paths or pay D2H/serialization overhead\.

    

## 11\. Validation and Acceptance



### 11\.1 MRV2 \+ R3



- MRV1 regression;

- MRV2 TP1/TP2;

- sync/async output copy;

- slot reuse and preemption;

- unsupported router/monolithic\-kernel fail\-closed behavior;

- speculative accepted/rejected overwrite;

- frontend stop boundary;

- element\-wise equality between the HTTP value and direct capture\.

    

### 11\.2 Artifact Connector \+ SHM



- initial miss, full hit, and partial hit;

- incremental full\-block commit;

- request\-local tail;

- terminal assembly;

- same\-key idempotency;

- missing/corrupt/profile\-mismatch fail\-closed behavior;

- capacity/retention/cleanup;

- worker crash and stale SHM;

- TP2;

- CPU KV eviction/reload\.

    

### 11\.3 Prompt Logprobs



- cache\-disabled baseline;

- first miss;

- full/partial prefix hit;

- element\-wise agreement of values and token order;

- boundary hidden\-state restoration;

- mandatory missing/corruption fail\-closed behavior;

- R3 \+ prompt logprobs \+ frontend stop\.

    

### 11\.4 Production Backend



- SHM/TQ backend contract parity;

- one logical publication;

- full\-block deduplication;

- atomically visible sample readiness;

- not\-found/not\-ready/ready;

- producer/consumer crashes;

- restart/checkpoint;

- capacity and backpressure;

- single\-node, multi\-process, and cross\-node;

- VIME dataloader batch equality\.

    

## 12\. Observability



At minimum, expose:



- captured tokens/bytes/time;

- full\-block commit/hit/miss;

- joint\-readiness miss reason;

- tail bytes;

- terminal assembly time;

- corrupt/profile mismatch/invariant violation;

- SHM used/free/evicted/stale;

- backend queue depth/backpressure;

- TQ not\-found/not\-ready/ready;

- Mooncake put/get latency;

- speculative overwrite count;

- CPU\-offload store/load count\.

    

Compare TTFT, TPOT, and throughput for ordinary serving with artifacts disabled against the baseline\.



## 13\. Abandoned or Deferred Designs



The following no longer belong to the current simple mode:



- HTTP returns a request\-scoped mmap handle;

- VIME reads vLLM `/dev/shm` using `openat`;

- VIME calls an ACK endpoint after materialization;

- adding an external lease/TTL/ACKED protocol for simple mode;

- requiring users to concatenate head/full\-block/tail manually;

- treating the `#45635` physical\-slot mmap as an immutable artifact store\.

    

The following are deferred until a production E2E test reveals a concrete need:



- changing Mooncake key generation;

- adding a deterministic Mooncake bundle API;

- adding a TransferQueue artifact control\-state machine;

- a standalone `materialize(handle)` service;

- having VIME publish an artifact handle to TQ\.

    

## 14\. Preserved and Revised Research Conclusions



Three conclusions from earlier source research remain valuable:



1. TransferQueue's primary value is its sample/dataloader/control abstraction, not allowing vLLM to bypass it and use Mooncake directly\.

2. The shared\-slot mmap from `#45635` is well suited to worker/scheduler staging, but its physical\-slot lifetime prevents it from directly providing cross\-request artifact retention\.

3. Whether the current Mooncake/TQ interfaces lack idempotent block deduplication or sample atomic\-ready semantics can only be established by a real adapter E2E test; upstream changes must not be assumed in advance\.

    

The following conclusions from the old V3 have been replaced:



- “Simple mode returns an SHM handle over HTTP” is replaced by “internal SHM backend, actual value returned over HTTP\.”

- “TQ is control plane only, and vLLM writes Mooncake directly” is replaced by “ArtifactConnector \-\> TQ \-\> Mooncake, with a single write\.”

- “The first version always rejects speculative decoding” is replaced by “do not blanket\-reject it; determine support from accepted\-token correctness and E2E results\.”

- “Prompt logprobs are currently excluded from artifacts” is replaced by a mandatory profile in separate PR C\.

- “Mooncake/TQ APIs must be changed first” is replaced by “adapt the existing APIs first\.”

    

## 15\. Researched Source Snapshots



Snapshots used in the earlier external source research are preserved as background\. They do not remove the need to recheck current sources when production PR work begins:



|Project|Researched revision|Purpose|
|---|---|---|
|Ascend/TransferQueue|`b75d570d88c50bbfcbe2171baa727fadd7216f76`|sample/schema/ready/Mooncake client/dataloader|
|kvcache\-ai/Mooncake|`ac010838926e3cff2659465bd9b7bf6c0e9656bf`|structured objects and storage semantics|
|VIME|`5cabf1f3e459e3df276c5b1f21478c475f04fb9d`|rollout/Sample/train\_data consumer|



Current implementation decisions are based on the local `#45635 + PR3 @ 8c01cea0d5` and the review worktree\.



Before PR D/E begins, the latest TransferQueue, Mooncake, and VIME interfaces must be pinned and rechecked\.

