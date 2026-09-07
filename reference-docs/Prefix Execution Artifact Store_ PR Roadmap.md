# Prefix Execution Artifact Store: PR Roadmap



## Final Goal



Retain only two Artifact Connector backends:



```Plain Text
Simple mode
GPU worker capture
  -> ArtifactConnector
  -> local SHM backend
  -> assemble actual R3 / prompt logprobs values
  -> ModelRunnerOutput
  -> EngineCore / HTTP

Production mode
GPU worker capture
  -> ArtifactConnector (sole producer)
  -> Mooncake storage backend
```



Both modes share the same upper\-level Artifact Connector interface, block identity, profile, full\-block/tail organization, and correctness constraints\. They differ only in backend storage, visibility, and lifecycle\.



Final decisions:



- Simple mode is also implemented through Artifact Connector; it is not a temporary capture path that bypasses the connector\.

- Simple mode uses SHM, while HTTP remains compatible and returns actual values\. Users do not need to concatenate blocks, and VIME does not need to implement an SHM\-handle reader\.

- In production mode, Artifact Connector publishes exactly once: `ArtifactConnector -> Mooncake`\.

    

## Roadmap Checklist



* [ ] **Phase 0: Finalize Existing PRs and Migrate MRV1 Work to MRV2**

    

    * [ ] R3

        

        * [ ] B [vllm\-project/vllm\#45635](https://github.com/vllm-project/vllm/pull/45635) — Owner: @申奥

            

            Create [`codex/r3-artifact-integration`](https://github.com/aoshen02/vllm/tree/codex/r3-artifact-integration) \(`8c01cea0d5`\) from the PR3 fixes already merged into the author's branch\.

            

        * [ ] F MRV2 R3 — Owner: @申奥

            

            [aoshen02/vllm\#3](https://github.com/aoshen02/vllm/pull/3)

            

            * [x] MRV1/MRV2 share the capture state, writer, and sync/async snapshot lifecycle\.

            * [x] Targeted CPU tests and static checks pass\.

            * [x] Complete real\-GPU E2E validation on the current PR head: TP, prefix miss/hit, CPU KV offload, sync/async, and preemption\.

            * [ ] Code review\.

                

    * [ ] DSA — Owners: @刘荣, @申奥

        

        * [ ] [vllm\-project/vllm\#47279](https://github.com/vllm-project/vllm/pull/47279)

        * [ ] Migrate to SHM\.

        * [ ] Migrate to MRV2\.

            

    * [ ] Top\-p/k token IDs — Owners: @招行, @申奥

        

        * [ ] [vllm\-project/vllm\#49577](https://github.com/vllm-project/vllm/pull/49577)

            

* [ ] **Phase 1: Establish Artifact Connector and Use SHM as Its First Backend** — Owner: @刘荣

    

    * [ ] Token\-wise

        

        * [ ] R3: [aoshen02/vllm\#4](https://github.com/aoshen02/vllm/pull/4)

        * [ ] Logprobs — Owner: @刘荣

        * [ ] DSA — Owner: @刘荣

        * [ ] Top\-p token IDs — Owner: @招行

            

    * [ ] Request\-wise

        * [ ] Multimodal data\.

            

* [ ] **Phase 2: Mooncake Production Backend** — Owner: @刘荣

    

    * [ ] **Mooncake Artifact Store**

        

        * [ ] Make `ArtifactConnector -> Mooncake` the only production write path, with no duplicate put\.

        * [ ] Preserve the same identity, finalize/discard, manifest, and materialize contracts as SHM\.

            

    * [ ] **Migrate Three Token\-Wise Fields**

        

        * [ ] R3: [aoshen02/vllm\#7](https://github.com/aoshen02/vllm/pull/7)

        * [ ] Logprobs\.

        * [ ] DSA\.

            

    * [ ] **Migrate Two Request\-Wise Fields**

        

        * [ ] Top\-p token IDs\.

        * [ ] Multimodal data\.

            

* [ ] **Phase 3: Integrate into veRL for E2E Testing** — Owner: @刘荣

    

    * [ ] Using the KV put/get interface

        

        * [ ] On put, construct the mapping `sample/request ID (key) -> artifact key list (value) -> artifact`\.

        * [ ] On get, retrieve the value by sample ID\.

            

    * [ ] Using the dataloader interface; relax the single\-node bottleneck

        

        * [ ] TQ may need changes\.

            

* [ ] **Extend Writer Topology Across Parallel Strategies**

    

    * [ ] **Implement a Multi\-Rank Writer** — Owner: \_\_\_\_\_\_\_\_\_\_

        

        * [ ] After single\-writer correctness is complete, write token/layer logical ranges in parallel\.

        * [ ] Verify that single\-writer and multi\-writer results are exactly identical\.

            

    * [ ] **Remove Parallelism/Feature Guards Individually** — Owner: \_\_\_\_\_\_\_\_\_\_

        

        * [ ] PP\.

        * [ ] DCP/PCP\.

        * [ ] DP/EP\.

        * [ ] DBO/microbatching, EC transfer, and other KV connectors\.

        * [ ] Before removing each guard, complete global\-ordering, writer\-placement, finalize/discard, and failure E2E validation\. Continue rejecting unvalidated combinations at startup\.

            

* [ ] **Track Mooncake/TQ\-Related Changes**

    

    * [ ] Mooncake

        

        * [ ] Keep most key puts local: use CUDA memory copies instead of RDMA\. Track DMA\-copy put support in [kvcache\-ai/Mooncake\#1946](https://github.com/kvcache-ai/Mooncake/pull/1946) to reduce NIC traffic contention\.

