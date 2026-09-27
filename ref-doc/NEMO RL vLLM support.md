# NEMO RL vLLM support

1. Rollout only

2. Vllm router added

    1. Add to nemo\-gym

    2. Add to nemo async gen

3. Multi configuration



* [x] https://github\.com/vllm\-project/vllm/pull/50874

    * [x] Almost mergeable, should expect to merge today\.

* [ ] Rollout only

    * [ ] https://github\.com/NVIDIA\-NeMo/RL/pull/3243

* [x] Build the image\.

* [ ] Routing integration

    * [ ] Dynamo https://github\.com/NVIDIA\-NeMo/RL/pull/3391

    * [ ] Vllm router/smg\(\) integration

        * [ ] https://github\.com/NVIDIA\-NeMo/RL/pull/3663

            1. Use worker url to init router

        * [ ] https://github\.com/NVIDIA\-NeMo/Gym/pull/2570

            * [ ] Add session id to head\.

        * [ ] Experiment result

            * [x] Dapo math 17k

            * [x] Codex multi turn replay

            * [ ] https://github\.com/NVIDIA\-NeMo/Gym/pull/2369 solve the sticky session problem when using nemo gymm\.

* [ ] Multi server groups

    * [x] Initial pr: https://github\.com/NVIDIA\-NeMo/RL/pull/3520

        * [x] mergable\.

    * [ ] Mooncake x nemo x vLLM integration

        * [x] https://github\.com/NVIDIA\-NeMo/RL/pull/3530

        * [x] mergeable

            * [ ] https://github\.com/vllm\-project/vllm/pull/53129

                * [ ] heterogeneous TP sharing support merged

    * [ ] Test with a long tail traces 8/20

        * [x] Validate the router \+ worker group

        * [x] A more long tail traces

            * [x] https://github\.com/NVIDIA\-NeMo/RL/pull/3663

            * [ ] 

* [ ] Paper survey

    * [x] https://claude\.ai/code/artifact/933b2e2d\-9987\-422c\-ad7b\-58265c339d4e

    * [ ] conduct experiment

        * [ ] Longest first

        * [ ] Turns down

* [ ] New consistency practice compare with xx replay\. Collab with megatron team\.

    * [ ] [Consistency practice](https://my.feishu.cn/wiki/HMacwKvaki8BqokUmYFcqoAsn3g)





# Multi turn \+ local docker:

# **NeMo\-RL flagship SWE2 recipe, end to end on one ARM64 GB200 node**



One GB200 node \(4 GPUs, aarch64\) running the full chain:

`NeMo-RL (Megatron) → NeMo-Gym → vLLM → OpenHands → ARM64 SWE sandbox → SWE-bench eval → reward → replay buffer → one training step`\. Every number below traces to a run directory under `agent_run/results/`\.



## **Changes \(the real fix is in Gym, not NeMo\-RL\)**



**Blocking bug\.** \`openhands\.sh\` hardcodes \`jq\-linux\-amd64\`\. On aarch64 the download succeeds and setup looks green, but the binary is copied into the SWE container where it dies with \`Exec format error\`\. The entry script's instance lookup then returns nothing and hits \`exit 1\` — and because the runtime *sources* that script, it kills the caller's shell\. OpenHands only sees \`command timed out after 600\.0 seconds\`, three retries, 1800s wasted per rollout\. **The symptom is a timeout; the failure happens at second zero\.** Fix: dispatch on \`uname \-m\`; re\-download when the existing binary cannot execute; drop \`jq \-\-version \|\| true\`, which is what swallowed the error in the first place\.



**Optimization\.** \`NEMO\_GYM\_SWE\_NO\_COPY=1\` replaces \`cp \-al /testbed\` with \`ln \-s\`\. The container's writable layer is RAM\-backed, so the recursive copy costs both time and memory\. It is safe because of \`\-\-writable\-tmpfs\`: the SIF is shared read\-only and each container gets its own in\-memory overlay — measured directly, two concurrent rollouts of the same instance produced different patches\. Branches: \`aoshen02/Gym\` @ \`fix/swe\-arm64\-jq\` \(\`openhands\.sh\` \+17 −4, \`swe\_agents/app\.py\` \+62\) · \`aoshen02/RL\` @ \`review/qwen3\-swe2\-singlenode\` \(harness, docs, evidence\)\.



## **Setup and data**



`Qwen3-30B-A3B-Thinking-2507` \(MoE\) · recipe \`grpo\_qwen3\_30ba3b\_thinking\_swe2\.yaml\` · training Megatron TP2/PP1/CP1/EP1 with full optimizer CPU offload · generation vLLM TP2, non\-colocated, async engine, \`max\_model\_len=32768\` · the agent is **OpenHands** \(Gym's default; \`CodeActAgent\`, 200\-turn cap\) and the evaluator is upstream \`swebench\.harness\.run\_local\_evaluation\` — the two are fully separate\.



Data: **10 django instances** from \`SWE\-bench\_Verified\`, stratified by gold\-patch size \(1–29 lines\) rather than cherry\-picked easy ones\. One ARM64 SIF per instance, converted straight from \`docker://swebench/sweb\.eval\.arm64\.\<id\>\` with no customisation\. GRPO: 10 prompts × 2 samples = 20 rollouts, \`max\_num\_steps=1\`, sandbox concurrency 10\.



## **Results**



Three runs of the same configuration, all \`exit\_code=0\`, reward mean **0\.5000 / 0\.4500 / 0\.3000** \(resolved 10/20, 9/20, 6/20\) — sampling spread at n=20\. Taking the latest \(13206\): reward \`min 0\.0 / max 1\.0 / mean 0\.5000 / std 0\.5130\`, resolved **10/20**, 14m53s wall clock\.



Pass rate falls **monotonically** with gold\-patch size: 1–2 lines 6/8 · 5 lines 2/4 · 11–29 lines 2/8\. **Advantage is non\-zero** \(\`±0\.7071, std 0\.3162\`\), so this step carries real gradient — a group needs both successes and failures to produce signal; an all\-pass or all\-fail group has advantage 0 and updates nothing no matter how healthy the chain is\.



**The reward is not hacked**: ① the \`/root/dataset/data\.jsonl\` mounted into the agent container and into the eval container are two different files — the agent's is 925B with \`patch\`, \`test\_patch\`, \`FAIL\_TO\_PASS\` and \`PASS\_TO\_PASS\` all absent; ② the target test is not in the image, and all 20 rollouts show \`touched\_tests=0\`; ③ \`eval\.sh\` does \`git checkout \<base\_commit\> tests/\.\.\.\` before applying \`test\_patch\`, and only the *model's* patch reaches the source tree; ④ a patch with broken indentation was correctly scored \`resolved=False\`\. The monotonic difficulty curve is itself something a leak could not produce\.



## **Reproduce**



```Bash
git clone https://github.com/aoshen02/RL  -b review/qwen3-swe2-singlenode
git clone https://github.com/aoshen02/Gym -b fix/swe-arm64-jq <RL>/3rdparty/Gym-workspace/Gym
cd <RL>/experiments/qwen3_swe2_arm64
bash   scripts/build_swe_sif_arm64.sh   django__django-17029          # build the SIF, ~2 min
python scripts/make_swe2_dataset_row.py django__django-17029 data/django17029_swe2.jsonl
bash   scripts/prewarm_nemo_gym_qwen3_swe2.sh                          # prewarm venvs, must be serial

SWE_DATASET=/workspace/agent_run/data/qwen3_swe2_smoke/django_mix10_swe2.jsonl \
NUM_PROMPTS=10 NUM_GENERATIONS=2 SWE_CONCURRENCY=10 \
  bash scripts/launch_qwen3_swe2_tp2_e2e.sh
bash handoff_qwen3_swe2_cold_start.sh --verify [run_dir]   # 0=all required pass, 1=some failed, 2=usage error
```



Prerequisites: Slurm \+ pyxis/enroot, one node with 4 GPUs \(aarch64\), image \`nemo\-rl\-vllm\-latest\.sqsh\`\. Docker Hub's arm64 images cover only **part** of the benchmark \(\~141 django instances\); missing ones return 401\. Optional: \`SWE\_INSTANCE=\` for a single instance · \`GYM\_DIR=\` to point at a different Gym tree without touching the main checkout · \`SWE\_VERIFY\_GOLDEN=1\` to calibrate the eval chain with the gold patch \(**calibration, not a deliverable**\)\. Concurrency is 10 rather than 20: a sandbox measures 1\.1–1\.6 GB, but \`apptainer\_memory\_limit\_mb=32768\` has **no cgroup enforcement** under enroot and is only watchdog\-polled, so the worst case is \`concurrency × 32 GB\`; the bulk of memory sits on the training side, leaving \~185 GB of the node's 893 GB once it is resident\.



## **Limits**



**\*\*The only claim this supports is that the chain above completed with \`exit\_code=0\`\.\*\*** It is not a model\-capability evaluation \(10 instances × 2 samples\); it is not evidence that training works \(\`max\_num\_steps=1\`, no convergence, no multi\-step\); evidence items 3/4/5/8 **cover 1 of the 20 rollouts** \(the collector takes \`head \-1\`\) and only the aggregate items cover all of them; item 6 is permanently NO and exempted \(OpenHands' own \`output\.jsonl\` is always 0 bytes on this path, so the trajectory comes from item 7, decoded from \`train\_data\_step\*\.jsonl\`\); and **this was only validated on aarch64 — an x86 regression is required before upstreaming**\. One transient failure is on record: job 13204 lost two vLLM TP workers to the system \(\`SYSTEM\_ERROR\` → \`Executor failed\.\`\); the same code on the same node passed on retry, so it is logged as infrastructure flake\. Note also that the \`batch\` partition is \`OverSubscribe=EXCLUSIVE\` — every job takes a whole node, so N small jobs cost N times the wall clock\.



See `ROOT_CAUSE_swe_entry_600s_timeout.md` \(the jq chain, including two self\-corrections\) · `ROOT_CAUSE_vllm_enginedead_keyerror.md` \(`mamba_cache_mode` inheritance trap, including one retracted attribution\) · `REWARD1_EVIDENCE.md` \(with 9 falsified hypotheses\) · `AUDIT.md` \(independent audit, including the verifier's own false positives and negatives\)\.





Shell script:

```Bash
#!/usr/bin/env bash
# =============================================================================
# workplace_assistant GRPO Training + vllm-router running on Slurm.
#
# Usage (execute on login node):
#   MODEL_PROFILE=nano9b   bash agent_run/scripts/phase4/grpo_workplace_assistant.sh
#   MODEL_PROFILE=super120b bash agent_run/scripts/phase4/grpo_workplace_assistant.sh   # Default
#   MAX_STEPS=5 TIME_LIMIT=02:00:00 MODEL_PROFILE=nano9b bash ...grpo_workplace_assistant.sh
#
# Two execution modes:
#   One-off (STAGE=submit, default): Launch cluster -> Run once -> Entire job exits.
#   Persistent (STAGE=cluster + attach): Cluster stays alive. Each iteration only attaches to run training.
#     STAGE=cluster bash ...          # Spin up persistent cluster, record <jobid>
#     JOB=<jobid> STAGE=attach bash ...  # Run once; can be executed repeatedly
#     scancel <jobid>                 # MUST release manually after use, otherwise nodes stay occupied
#   Persistent mode saves ~90s per run for container initialization + Ray cluster formation;
#   Worktree is a bind mount, so code changes take effect on the next attach without restarting the cluster.
#
# Call chain:
#   This script (STAGE=submit/cluster/attach, login node)
#     └─ sbatch ray.sub         NeMo-RL built-in: spins up container + Ray cluster (unmodified)
#          └─ This script (STAGE=run, inside head node container) <- $COMMAND of ray.sub
#               ├─ vllm-router
#               └─ run_grpo_nemo_gym.py --config <YAML>
#
# Division of responsibility (matches official super_launch.sh):
#   YAML: Machine-agnostic configs — batch shapes, parallelism, node count, Gym environment.
#   This script: Machine-dependent configs — model path, data path, vLLM engine args, cache dirs.
# The script performs NO parameter calculations (neither DP nor batch shapes); it only performs injections.
# =============================================================================
set -uo pipefail

# --- Fixed Paths and Ports ----------------------------------------------------
NMU=/home/inf-aoshen/vllm/projects/vllm-rl-day0-support/nmu
RL_ROOT="$NMU/pr_worktrees/gym-router-url"          # NeMo-RL worktree (uncommitted changes)
SELF="$NMU/agent_run/scripts/phase4/grpo_workplace_assistant.sh"  # Must be hardcoded: $0 cannot be retrieved inside container
GPUS_PER_NODE=4                 # ray.sub asserts this against partition GRES: gpu:nvidia_gb200:4, non-configurable
PORT=30000                      # Router external service port
METRICS_PORT=29000              # Router Prometheus metrics port

# --- Select Model Profile ----------------------------------------------------
# The `++` prefix is used for keys not present in the parent config — Hydra operates in struct mode,
# without `++` it throws "Key not in struct" (Job 12872). Official super_launch.sh:62-68 uses this same pattern.
GYM_DATA=3rdparty/Gym-workspace/Gym/data/workplace_assistant
case "${MODEL_PROFILE:=super120b}" in
  super120b)
    CONFIG=examples/nemo_gym/nemotron-3-super/small_scale/stage1_rlvr_convergence_27node_h100.yaml
    MODEL=/mnt/lustre01/users/inf-aoshen/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
    TRAIN_DATA="$GYM_DATA/train.jsonl"
    VAL_DATA="$GYM_DATA/validation.jsonl"
    TIME_LIMIT="${TIME_LIMIT:-04:00:00}"   # First run requires converting 231GB weights + compiling 120B MoE
    ENGINE_ARGS=(
      ++policy.generation.vllm_kwargs.max_num_batched_tokens=8480
      # Do not override max_new_tokens: in parent config it is ${policy.max_total_sequence_length}.
      # The real constraint on generation is the total sequence length budget; setting an additional cap simply truncates tool calls.
      # Disable FlashInfer's kernel autotune (in Job 12910, "Tuning flashinfer::trtllm_bf16_moe" appeared 334 times).
      # FlashInfer does not persist these results, so cache cannot save it; disabling is the only option. vLLM's -O0 preset
      # disables it by default anyway (config/vllm.py:254). Trade-off: MoE kernel will use default heuristic configs,
      # which might degrade generation throughput — must revert alongside enforce_eager before official training.
      ++policy.generation.vllm_kwargs.kernel_config.enable_flashinfer_autotune=false
      # moe_backend=auto on GB200 selects FlashInfer TRTLLM, which repacks BF16 expert weights into a backend-specific
      # tiled layout (dumped w13_weight is 4D (512,16,768,64), where last dim 64 is tile). Generic load_weights cannot
      # write into it, causing refit shape mismatch. NeMo-RL's own refit_loader.py:73-78 explicitly states to set this to triton.
      ++policy.generation.vllm_kwargs.moe_backend=triton
      # Gym launches servers based on all entrypoint definitions in config, regardless of config_paths.
      # Parent config stage1_rlvr inlined three judge models (two 235B, one 4B) plus a reasoning_off variant.
      # None exist locally nor are GPUs allocated for Gym; must delete them with `~`,
      # otherwise it stalls at "0 / 7 servers ready" (Job 12897). Deleting leaves only 3 servers.
      ~env.nemo_gym.safety_judge_model
      ~env.nemo_gym.nl2bash_judge_model
      ~env.nemo_gym.genrm_model
      ~env.nemo_gym.policy_model_reasoning_off
    )
    ;;
  codex)
    # Trajectory Replay: No training, routing policy benchmark only. Data generated by
    # agent_run/scripts/phase4/build_codex_replay_dataset.py, 1 session per line.
    # Context naturally accumulates across multi-turn loops in codex_replay_agent.
    # Current profile is minimal for fast iteration: 64 sessions x 2 turns, final turn context p50 24K.
    CONFIG=examples/nemo_gym/nemotron-3-super/small_scale/stage1_rlvr_convergence_27node_h100.yaml
    MODEL=/mnt/lustre01/users/inf-aoshen/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
    TRAIN_DATA="$NMU/agent_run/data/codex_replay_t2/train.jsonl"
    VAL_DATA="$NMU/agent_run/data/codex_replay_t2/validation.jsonl"
    TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
    ENGINE_ARGS=(
      # mamba_cache_mode=align requires block_size(2080) <= max_num_batched_tokens;
      # parent config default (2048) is too small, causing engine startup assertion failure.
      ++policy.generation.vllm_kwargs.max_num_batched_tokens=8480
      ++policy.generation.vllm_kwargs.kernel_config.enable_flashinfer_autotune=false
      ++policy.generation.vllm_kwargs.moe_backend=triton
      # Skip CUDA graph capture during iteration phase: measured to increase engine init from 4s to 30s,
      # while single-step generation only takes 1-2 minutes; CUDA graph speedup cannot offset the 24s overhead.
      # Remove this when config stabilizes for long runs.
      policy.generation.vllm_cfg.enforce_eager=true
      # 2-turn replay: final turn context p50 24K / max 50K. Set to 65536 for headroom.
      policy.max_total_sequence_length=65536
      # Replace Gym servers to launch: keep only policy model + replay agent. Tools aren't executed, so resources_server is unneeded.
      # Inline judge models in parent config are also removed (otherwise stuck waiting for servers to be ready).
      "++env.nemo_gym.config_paths=[responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,responses_api_agents/codex_replay_agent/configs/codex_replay_agent.yaml]"
      ~env.nemo_gym.safety_judge_model
      ~env.nemo_gym.nl2bash_judge_model
      ~env.nemo_gym.genrm_model
      ~env.nemo_gym.policy_model_reasoning_off
    )
    ;;
  dapo)
    # Dataset swap only: YAML requires zero changes, reusing super120b's config (parallelism, 8-node/4-GPU, 16384
    # sequence length are sufficient for DAPO — prompt is only ~120 tokens).
    # Gym environment resources_servers/math_with_judge is fully integrated: when should_use_judge
    # is false, it routes to local math_verify without needing a judge model. Only data files are missing;
    # official pipeline uses internal GitLab artifact, generated here from HF via agent_run/scripts/phase4/build_dapo_gym_dataset.py.
    # Filenames and line counts aligned with data/*_metrics.json (1.79M / 960 lines).
    CONFIG=examples/nemo_gym/nemotron-3-super/small_scale/stage1_rlvr_convergence_27node_h100.yaml
    MODEL=/mnt/lustre01/users/inf-aoshen/models/NVIDIA-Nemotron-3-Super-120B-A12B-BF16
    MWJ=3rdparty/Gym-workspace/Gym/resources_servers/math_with_judge
    TRAIN_DATA="$MWJ/data/dapo17k_bytedtsinghua_train.jsonl"
    VAL_DATA="$MWJ/data/aime24_bytedtsinghua_validation.jsonl"
    TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
    # 16 nodes: training side unchanged, inference side 4 -> 12 nodes.
    # Training node count in grpo.py:807 is calculated via cluster.num_nodes - colocated.resources.num_nodes,
    # 16 - 12 = 4 nodes, so TP2/EP16/ETP1/DP8 are retained as-is; vLLM TP4 unchanged, engine count 4 -> 12.
    NODES="${NODES:-16}"
    GEN_NODES="${GEN_NODES:-12}"
    ENGINE_ARGS=(
      ++cluster.num_nodes="$NODES"
      ++policy.generation.colocated.resources.num_nodes="$GEN_NODES"
      # Essential: mamba_cache_mode=align requires max_num_batched_tokens >= Mamba's block_size(2080);
      # default 2048 causes engine startup assertion failure.
      ++policy.generation.vllm_kwargs.max_num_batched_tokens=8480
      ++policy.generation.vllm_kwargs.kernel_config.enable_flashinfer_autotune=false
      ++policy.generation.vllm_kwargs.moe_backend=triton
      "++env.nemo_gym.config_paths=[responses_api_models/vllm_model/configs/vllm_model_for_training.yaml,resources_servers/math_with_judge/configs/dapo17k.yaml]"
      # Parent config stage1_rlvr.yaml:512 hardcodes should_use_judge=true and points judge to
      # nl2bash_judge_model, overriding Gym's dapo17k.yaml (false). Must set back to false here
      # to use local math_verify; judge_model_server.name is filled concurrently to avoid dangling reference after deleting nl2bash_judge_model.
      ++env.nemo_gym.math_with_judge.resources_servers.math_with_judge.should_use_judge=false
      ++env.nemo_gym.math_with_judge.resources_servers.math_with_judge.judge_model_server.name=policy_model
      ~env.nemo_gym.safety_judge_model
      ~env.nemo_gym.nl2bash_judge_model
      ~env.nemo_gym.genrm_model
      ~env.nemo_gym.policy_model_reasoning_off
    )
    ;;
  nano9b)
    CONFIG=examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml
    MODEL=nvidia/NVIDIA-Nemotron-Nano-9B-v2
    TRAIN_DATA="$GYM_DATA/train.jsonl"
    VAL_DATA="$NMU/agent_run/data/wa_validation_48.jsonl"   # Official 545 entries pruned down to 48
    TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
    ENGINE_ARGS=(
      policy.generation.vllm_cfg.enforce_eager=true   # Skip CUDA graph capture for smoke testing
      policy.generation.max_new_tokens="${MAX_NEW_TOKENS:-128}"
    )
    ;;
  *) echo "unknown MODEL_PROFILE=$MODEL_PROFILE (super120b|codex|dapo|nano9b)" >&2; exit 1 ;;
esac

# Default node count extracted from YAML (cluster.num_nodes) for sbatch to avoid duplicating values across two places.
# When changing scale via ++cluster.num_nodes in profile, NODES must be set explicitly;
# otherwise sbatch allocated nodes won't match Hydra's view, causing placement group to wait indefinitely.
NODES="${NODES:-$(awk '/^cluster:/{c=1} c&&/^  num_nodes:/{print $2; exit}' "$RL_ROOT/$CONFIG")}"

STAGE="${STAGE:-submit}"
NEW_RUN_DIR() { echo "$NMU/agent_run/results/wa_${MODEL_PROFILE}_${NODES}n_$(date -u +%Y%m%dT%H%M%SZ)"; }
# Environment variables passed to STAGE=run, shared by both submit and attach paths
RUN_CMD() {
  echo "MODEL_PROFILE=$MODEL_PROFILE STAGE=run RUN_DIR=$1 MAX_STEPS=${MAX_STEPS:-1} \
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-} ROUTER_POLICY=${ROUTER_POLICY:-consistent_hash} \
ROLLOUT_ONLY=${ROLLOUT_ONLY:-0} PROMPTS=${PROMPTS:-} TENSORBOARD=${TENSORBOARD:-false} bash $SELF"
}

# =============================================================================
# Outside Phase 1: Attach (Login node, entering an existing persistent cluster)
# Once cluster is ready, ray.sub writes <jobid>-attach.sh to SLURM_SUBMIT_DIR (i.e. RL_ROOT).
# Internally it runs `srun --overlap --container-name=ray-head` to reuse the same container instance.
# =============================================================================
if [ "$STAGE" = attach ]; then
  set -e
  ATTACH="$RL_ROOT/${JOB:?attach requires JOB=<persistent_cluster_jobid>}-attach.sh"
  [ -f "$ATTACH" ] || { echo "Cannot find $ATTACH (is the cluster ready?)" >&2; exit 1; }
  # If previous attach failed, process may not exit (driver hangs on refit error). Residual Ray
  # actors keep occupying placement group, causing next attach to stall at "Timed out waiting for placement groups to be ready". Clean up first.
  # `|| true`: if no residual processes exist, pgrep returns 1, which under `set -e` would silently exit script.
  STALE=$(pgrep -f "srun .*--jobid $JOB .*STAGE=run" | tr '\n' ' ' || true)
  [ -n "$STALE" ] && { echo "Cleaning up leftover attach process: $STALE"; kill $STALE; sleep 10; }
  RUN_DIR=$(NEW_RUN_DIR)
  mkdir -p "$RUN_DIR"
  echo "attach job=$JOB profile=$MODEL_PROFILE"
  echo "run_dir=$RUN_DIR"
  COMMAND="$(RUN_CMD "$RUN_DIR")" bash "$ATTACH"
  exit
fi

# =============================================================================
# Phase 1: Submit (Login node)
# When STAGE=cluster, COMMAND is empty. ray.sub sleeps indefinitely after spinning up cluster and writes attach script.
# =============================================================================
if [ "$STAGE" = submit ] || [ "$STAGE" = cluster ]; then
  set -e
  if [ "$STAGE" = cluster ]; then
    RUN_DIR="$NMU/agent_run/results/cluster_${MODEL_PROFILE}_${NODES}n_$(date -u +%Y%m%dT%H%M%SZ)"
    COMMAND=""
    TIME_LIMIT="${TIME_LIMIT:-08:00:00}"
  else
    RUN_DIR=$(NEW_RUN_DIR)
    COMMAND="$(RUN_CMD "$RUN_DIR")"
  fi
  mkdir -p "$RUN_DIR"
  echo "stage=$STAGE profile=$MODEL_PROFILE nodes=$NODES model=$MODEL"
  echo "run_dir=$RUN_DIR"

  # Mount explanation:
  #   First two are identity mounts (same path inside/outside container), so RUN_DIR / RL_ROOT need no path mapping.
  #   Third one is critical: nemo_gym inside Ray actor's venv is installed in editable mode pointing to
  #   /opt/nemo-rl/3rdparty/Gym-workspace/Gym inside the image, whereas PYTHONPATH can only redirect nemo_rl.
  #   If worktree isn't used to overlay that path, all Gym-side changes will be silently ignored (Pitfall from Job 12859).
  cd "$RL_ROOT"
  COMMAND="$COMMAND" \
  CONTAINER=/mnt/lustre01/users/inf-aoshen/enroot/containers/nemo-rl-vllm-latest.sqsh \
  MOUNTS="$NMU:$NMU,/mnt/lustre01:/mnt/lustre01,$RL_ROOT/3rdparty/Gym-workspace/Gym:/opt/nemo-rl/3rdparty/Gym-workspace/Gym" \
  GPUS_PER_NODE="$GPUS_PER_NODE" BASE_LOG_DIR="$RUN_DIR" \
  sbatch --nodes="$NODES" --account=inferact --partition=batch --time="$TIME_LIMIT" \
         --job-name="${JOB_NAME:-wa_$MODEL_PROFILE}" --gres="gpu:$GPUS_PER_NODE" --exclusive \
         --output="$RUN_DIR/slurm-%j.log" ray.sub
  exit
fi

# =============================================================================
# Phase 2: Execution (Head node, inside container, invoked by ray.sub)
# =============================================================================
RUN_DIR="${RUN_DIR:?}"
cd "$RL_ROOT"

# --- Pre-validation: Verify Gym uses worktree rather than image built-in copy ---
# session_affinity_header is added by us in worktree and absent in image, used as fingerprint.
# When mount fails, execution proceeds without error, but Gym changes silently fail with zero indication in logs; must intercept early.
grep -q session_affinity_header \
  /opt/nemo-rl/3rdparty/Gym-workspace/Gym/responses_api_models/vllm_model/app.py ||
  { echo "FATAL: Gym not bind-mounted from worktree" >&2; exit 1; }

# --- Environment Variables ----------------------------------------------------
# PERSISTENT_CACHE subdirectories are carried over from official super_launch.sh for cross-job reuse:
#   megatron_ckpt_cache    HF->Megatron weight conversion output (~8 mins for 120B)
#   hf_config_locks        File locks when Megatron-Bridge reads HF config
#   hf_modules             Dynamic modules generated by trust_remote_code
#   vllm_compile_cache     vLLM torch.compile artifacts
#   gym_venvs              venv for each Gym server (Super YAML references this via oc.env)
export HF_HOME=/mnt/lustre01/users/inf-aoshen/huggingface
export PERSISTENT_CACHE=/mnt/lustre01/users/inf-aoshen/nemo_persistent_cache
export PYTHONPATH="$RL_ROOT:/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM:${PYTHONPATH:-}"
export NRL_MEGATRON_CHECKPOINT_DIR="$PERSISTENT_CACHE/megatron_ckpt_cache"
export MEGATRON_CONFIG_LOCK_DIR="$PERSISTENT_CACHE/hf_config_locks"
export HF_MODULES_CACHE="$PERSISTENT_CACHE/hf_modules"
export VLLM_CACHE_ROOT="$PERSISTENT_CACHE/vllm_compile_cache"
# VLLM_CACHE_ROOT only manages vLLM's own torch.compile artifacts, not Triton. Nemotron-H's
# Mamba2 SSD kernel relies on Triton autotune (mamba_mixer2.py:596), which defaults to container's
# ~/.triton/cache and vanishes when job exits — requiring recompilation every run (~50s delay in Job 12910).
export TRITON_CACHE_DIR="$PERSISTENT_CACHE/triton_cache"
# FlashInfer JIT artifacts land in $FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/<ver>/<arch>;
# without persistent path, it writes to container's ~/.cache and disappears post-job.
# Do NOT set FLASHINFER_CUBIN_DIR: that's for precompiled cubin downloads requiring version checks
# (flashinfer/jit/env.py:63-94); without precompiled packages, setting it leaves it empty.
# As for 300+ autotunes during "Tuning flashinfer::trtllm_bf16_moe", FlashInfer does not persist
# results — caching won't help, so autotune must be disabled instead.
export FLASHINFER_WORKSPACE_BASE="$PERSISTENT_CACHE/flashinfer_workspace"
# RAY_DEDUP_LOGS=0: Otherwise Ray collapses identical logs across workers to "[repeated N x]",
# obscuring which rank got stuck during engine startup.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1 RAY_DEDUP_LOGS=0
# NRL_IGNORE_VERSION_MISMATCH: Worktree code version and image baked version don't match strictly.
# UV_LOCK_TIMEOUT: Concurrent Gym servers building the same editable nemo-gym acquire lock under
#   /root/.cache/uv; default 300s timeout is insufficient (Job 12876 failed here).
export NRL_IGNORE_VERSION_MISMATCH=1 UV_LOCK_TIMEOUT=1200
mkdir -p "$PERSISTENT_CACHE"/{megatron_ckpt_cache,hf_config_locks,hf_modules,vllm_compile_cache,triton_cache,gym_venvs}

# --- Warm up HF Dynamic Module Cache -----------------------------------------
# Nemotron requires trust_remote_code, generating dynamic module files on the fly.
# Multiple nodes writing concurrently to the same cache directory on initial run is a known failure point; run standalone on head node first to populate cache.
python -c "from transformers import AutoConfig, AutoTokenizer
AutoConfig.from_pretrained('$MODEL', trust_remote_code=True)
AutoTokenizer.from_pretrained('$MODEL', trust_remote_code=True, use_fast=True)" || exit 1

# Override prompts count only if PROMPTS is explicitly provided, otherwise default to YAML.
PROMPT_ARG=()
[ -n "${PROMPTS:-}" ] && PROMPT_ARG=(grpo.num_prompts_per_step="$PROMPTS")

# --- Start Router ------------------------------------------------------------
# ROUTER_POLICY=none completely disables router startup: nemo_gym.py:233 hands all vLLM replica
# base_urls directly to Gym when router_url is empty, letting Gym handle selection — serving as no-router baseline.
# Other values represent supported policies in vllm-router 0.1.15 (router_args.py:137-142):
#   random / round_robin / cache_aware / power_of_two / consistent_hash
# Router must run on head node as Gym receives the head node IP.
# --request-id-headers is required: 0.1.15 default list is x-request-id / x-correlation-id /
# x-trace-id / request-id, excluding x-session-id. Without explicitly adding it, Gym session headers are ignored,
# causing consistent_hash to silently degenerate into hashing request bodies, breaking session affinity.
ROUTER_POLICY="${ROUTER_POLICY:-consistent_hash}"
ROUTER_ARG=(env.nemo_gym.router_url=null)
if [ "$ROUTER_POLICY" != none ]; then
  python -m pip install --quiet vllm-router==0.1.15
  vllm-router --host 0.0.0.0 --port $PORT --policy "$ROUTER_POLICY" \
    --prometheus-port $METRICS_PORT --request-id-headers x-session-id x-request-id \
    > "$RUN_DIR/router.log" 2>&1 &
  ROUTER=$!
  ROUTER_ARG=(env.nemo_gym.router_url="http://$(hostname -I | awk '{print $1}'):$PORT")
fi

# Snapshot evidence before exit: worker registry + router counters + /metrics for each vLLM replica
# (Prefix cache hit rates exist only on backends, invisible to router side).
# This is the sole evidence chain for routing; training exit code 0 does not imply routing functioned correctly.
snapshot() {
  local tag="${1:-exit}"
  [ "$ROUTER_POLICY" = none ] || {
    curl -s "http://127.0.0.1:$PORT/list_workers" > "$RUN_DIR/list_workers.json"
    curl -s "http://127.0.0.1:$METRICS_PORT/metrics" > "$RUN_DIR/router_metrics.$tag.txt"
  }
  : > "$RUN_DIR/worker_metrics.$tag.txt"
  for u in $(grep -oE "http://[0-9.]+:[0-9]+" "$RUN_DIR/list_workers.json" 2>/dev/null); do
    echo "### $u" >> "$RUN_DIR/worker_metrics.$tag.txt"
    curl -s --max-time 5 "$u/metrics" >> "$RUN_DIR/worker_metrics.$tag.txt"
  done
}
trap 'snapshot; kill ${ROUTER:-} 2>/dev/null' EXIT

# Wait for router to spin up; if router dies during startup, exit immediately to prevent training from waiting pointlessly.
if [ "$ROUTER_POLICY" != none ]; then
  until curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
    kill -0 $ROUTER 2>/dev/null || { echo "router died on startup" >&2; exit 1; }
    sleep 2
  done
fi

# --- Run Training ------------------------------------------------------------
# Override only options known at runtime or intentionally disabled for this run:
# step counts, router URL (head IP available now), log backends, checkpointing.
# Everything else follows YAML defaults.
python -u examples/nemo_gym/run_grpo_nemo_gym.py --config "$CONFIG" \
  policy.model_name="$MODEL" \
  data.train.data_path="$TRAIN_DATA" \
  data.validation.data_path="$VAL_DATA" \
  "${ENGINE_ARGS[@]}" "${PROMPT_ARG[@]}" \
  ++env.nemo_gym.nemo_gym_log_dir="$RUN_DIR/gym_logs" \
  grpo.max_num_steps="${MAX_STEPS:-1}" \
  "${ROUTER_ARG[@]}" \
  logger.wandb_enabled=false logger.tensorboard_enabled="${TENSORBOARD:-false}" \
  checkpointing.enabled=false 2>&1 | tee "$RUN_DIR/nemo_rl.log" &
TRAIN=$!

# ROLLOUT_ONLY=1: Collect rollout data only; training phase (measured at 114s in previous run, up to ~10+ mins with batch expanded to 1024) is skipped.
# NeMo-RL lacks a rollout-only flag, so monitor logs for "Computing logprobs" — the first stage post-generation.
# When detected, dump evidence snapshot and stop execution.
if [ "${ROLLOUT_ONLY:-0}" = 1 ]; then
  while kill -0 $TRAIN 2>/dev/null; do
    grep -q "Computing logprobs" "$RUN_DIR/nemo_rl.log" 2>/dev/null && {
      echo "Rollout completed, capturing evidence and stopping"; snapshot rollout; kill -INT $TRAIN 2>/dev/null; sleep 20; break
    }
    sleep 10
  done
fi
wait $TRAIN
RC=$?

exit $RC
```





Yaml file:

```YAML
defaults: "../stage1_rlvr.yaml"

# GB200 4-GPU/node shape; parameters copied from examples/configs/recipes/llm/
# grpo-nemotron3-super-120BA12B-8n4g-megatron.yaml.
#
# 8 nodes x 4 GPUs = 32 GPUs  ->  16 for training (4 nodes) + 16 for generation (4 nodes)
#   Training    TP2 x CP1 x PP1 = 2 GPUs/replica -> DP=8; ETP1 x EP16 == TP2 x DP8  ✓
#   Generation  vLLM TP4 -> 4 engines
# Only entries that differ from the parent config are written here: EP16 / ETP1 / PP1 /
# colocated.enabled=false / mamba_* / compilation_config are all inherited from
# stage1_rlvr.yaml with the correct values.
#
# Same split of responsibilities as super_launch.sh: model path, data paths, and the
# machine-dependent vLLM engine args are injected by the launch script on the command
# line, so this file stays machine-independent.

grpo:
  num_prompts_per_step: 16
  num_generations_per_prompt: 8
  async_grpo:
    enabled: false

policy:
  train_global_batch_size: ${mul:${grpo.num_prompts_per_step}, ${grpo.num_generations_per_prompt}}
  # Measured initial prompts in this dataset cluster at 4140-4181 tokens (1255 samples,
  # almost no long tail). 8192 would leave only 4k for the agent's multi-turn tool calls,
  # so roughly 1% of requests blow past the context and get 400'd by vLLM.
  # 16384 leaves about 12k of trajectory room. The parent config derives train_mb_tokens /
  # logprob_mb_tokens / max_new_tokens / vllm max_model_len from this one value, so
  # changing it here is enough.
  # Official prod (stage1_rlvr.yaml:96) uses 65536; we don't follow because train_mb_tokens
  # would grow 8x and this cluster can't fit the activation memory.
  max_total_sequence_length: 16384

  megatron_cfg:
    tensor_model_parallel_size: 2
    context_parallel_size: 1
    env_vars:
      PYTORCH_CUDA_ALLOC_CONF: expandable_segments:False
      NRL_REFIT_BUFFER_MEMORY_RATIO: "0.005"
      NRL_REFIT_NUM_BUFFERS: "1"

  generation:
    colocated:
      resources:
        gpus_per_node: 4
        num_nodes: 4
    vllm_cfg:
      tensor_parallel_size: 4
      gpu_memory_utilization: 0.5
      env_vars:
        NRL_REFIT_BUFFER_MEMORY_RATIO: "0.005"
        NRL_REFIT_NUM_BUFFERS: "1"

env:
  nemo_gym:
    router_url: null   # Key must exist for Hydra struct mode to allow a CLI override
    num_gpu_nodes: 0
    config_paths:
    - responses_api_models/vllm_model/configs/vllm_model_for_training.yaml
    - resources_servers/workplace_assistant/configs/workplace_assistant.yaml
    policy_model:
      responses_api_models:
        vllm_model:
          num_workers: 4
          session_affinity_header: X-Session-ID

cluster:
  gpus_per_node: 4
  num_nodes: 8

```







