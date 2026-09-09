# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side execution-artifact data plane."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from threading import Event
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.distributed.artifact_connector.connector import (
    ArtifactConnectorMetadata,
    ArtifactRequestOutput,
)
from vllm.distributed.artifact_connector.logprobs import (
    LogprobRow,
    LogprobsLogicalSuffix,
    logprob_block_nbytes,
    logprobs_keys,
    materialize_logprobs_range,
    publish_logprobs,
    slice_logprobs_per_request,
)
from vllm.distributed.artifact_connector.routed_experts import (
    RoutedExpertsArtifactBuffer,
    materialize_routed_experts,
    publish_routed_experts,
    routed_experts_keys,
)
from vllm.distributed.artifact_connector.store import (
    BackgroundArtifactStore,
    InProcessArtifactStore,
)
from vllm.distributed.parallel_state import get_tp_group
from vllm.model_executor.layers.fused_moe.routed_experts_capturer import (
    RoutedExpertsCapturer,
    bind_routed_experts_capturer,
)
from vllm.v1.core.kv_cache_utils import resolve_kv_cache_block_sizes
from vllm.v1.outputs import LogprobsLists, LogprobsTensors

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig


@dataclass
class _WorkerRequestState:
    artifact_keys: list[str] = field(default_factory=list)
    pending_blocks: list[tuple[int, np.ndarray]] = field(default_factory=list)
    capture_cursor: int | None = None
    scheduled_cursor: int = 0
    emit_cursor: int = 0
    # Logprobs field state: one token-index-aligned suffix per request.
    logprob_suffix: LogprobsLogicalSuffix | None = None
    logprob_keys: list[str] = field(default_factory=list)
    logprob_pending: list[int] = field(default_factory=list)
    # Highest logprob block index already published or staged as pending.
    logprob_done: int = -1
    logprob_width: int | None = None
    logprob_store: BackgroundArtifactStore | None = None


@dataclass
class PendingArtifactOutput:
    """Own one step's GPU snapshot until its asynchronous copy is consumed."""

    connector: ArtifactWorkerConnector
    token_starts: np.ndarray
    query_start_loc: np.ndarray
    routed_experts: torch.Tensor | None = None
    finished: Event = field(default_factory=Event)

    def complete(self) -> None:
        self.connector._pending_output = None
        self.finished.set()


class ArtifactWorkerConnector:
    """Own capture, request tails, and backend resources on the output worker."""

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        model: torch.nn.Module,
        kv_cache_config: KVCacheConfig,
        max_num_batched_tokens: int,
    ) -> None:
        artifact_config = vllm_config.artifact_config
        r3_enabled = artifact_config.enable_return_routed_experts
        logprobs_enabled = artifact_config.enable_return_logprobs
        scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        hashes_per_kv_block = scheduler_block_size // hash_block_size

        self._capturer: RoutedExpertsCapturer | None = None
        self._store: BackgroundArtifactStore | None = None
        self._buffer: RoutedExpertsArtifactBuffer | None = None
        # Only the executor output rank owns the logprobs data plane; the flag
        # is read through ``getattr`` so test harnesses built with
        # ``object.__new__`` still see it as disabled.
        self._logprobs_enabled = logprobs_enabled
        self._logprobs_active = logprobs_enabled and get_tp_group().is_first_rank
        self._logprobs_mode = vllm_config.model_config.logprobs_mode
        self._logprobs_block_size = hash_block_size
        self._logprobs_stores: dict[tuple[int, str], BackgroundArtifactStore] = {}
        # Explicit cap shared by every per-field store; ``None`` derives each
        # store's capacity from the KV cache (one object per KV block).
        self._explicit_max_bytes = artifact_config.max_bytes
        self._logprobs_max_objects = (
            kv_cache_config.num_blocks * hashes_per_kv_block
        )
        self._logprobs_max_pending = 2 * vllm_config.scheduler_config.max_num_seqs
        self._requests: dict[str, _WorkerRequestState] = {}
        self._generation = 0
        self._step_metadata: ArtifactConnectorMetadata | None = None
        self._pending_output: PendingArtifactOutput | None = None

        # Every TP rank participates in R3 capture collectives, but only the
        # executor output rank owns the artifact data plane.
        if r3_enabled:
            capturer = RoutedExpertsCapturer(
                max_num_batched_tokens=max_num_batched_tokens,
                vllm_config=vllm_config,
            )
            bind_routed_experts_capturer(model, capturer)
            self._capturer = capturer
        if not get_tp_group().is_first_rank:
            return

        if r3_enabled:
            assert self._capturer is not None
            shape_per_token = self._capturer.shape_per_token
            dtype: np.dtype[Any] = np.dtype(self._capturer.output_dtype_name)
            block_nbytes = (
                hash_block_size
                * int(np.prod(shape_per_token))
                * dtype.itemsize
            )
            max_bytes = self._explicit_max_bytes
            if max_bytes is None:
                max_bytes = (
                    kv_cache_config.num_blocks
                    * hashes_per_kv_block
                    * block_nbytes
                )
            self._store = BackgroundArtifactStore(
                InProcessArtifactStore(
                    max_bytes=max_bytes,
                    object_nbytes=block_nbytes,
                ),
                max_pending_batches=2 * vllm_config.scheduler_config.max_num_seqs,
            )
            self._buffer = RoutedExpertsArtifactBuffer(
                dtype,
                shape_per_token,
                hash_block_size,
                vllm_config.scheduler_config.max_num_seqs,
                max_num_batched_tokens,
                vllm_config.max_concurrent_batches,
            )

    def _logprobs_store(self, width: int) -> BackgroundArtifactStore | None:
        """Return (or lazily create) the store for one capture width."""
        if not self._logprobs_enabled:
            return None
        key = (width, self._logprobs_mode)
        store = self._logprobs_stores.get(key)
        if store is not None:
            return store
        object_nbytes = logprob_block_nbytes(self._logprobs_block_size, width)
        max_bytes = self._explicit_max_bytes
        if max_bytes is None:
            # Size every width's store to hold one object per KV hash block,
            # the same capacity contract as the R3 store.
            max_bytes = object_nbytes * self._logprobs_max_objects
        store = BackgroundArtifactStore(
            InProcessArtifactStore(
                max_bytes=max_bytes,
                object_nbytes=object_nbytes,
            ),
            max_pending_batches=self._logprobs_max_pending,
        )
        self._logprobs_stores[key] = store
        return store

    def prepare_output(
        self,
        request_ids: list[str],
        token_starts: np.ndarray,
        query_start_loc: np.ndarray,
    ) -> PendingArtifactOutput | None:
        """Snapshot one step's R3 tensor for asynchronous CPU transfer."""
        buffer = self._buffer
        if self._step_metadata is None:
            return None
        if buffer is None and not getattr(self, "_logprobs_active", False):
            return None
        assert self._pending_output is None

        query_start_loc = query_start_loc[: len(request_ids) + 1]
        num_rows = int(query_start_loc[-1])
        routed_experts = (
            self._capturer.snapshot_routing_data(num_rows)
            if buffer is not None and self._capturer is not None
            else None
        )
        pending_output = PendingArtifactOutput(
            self,
            token_starts,
            query_start_loc,
            routed_experts,
        )
        self._pending_output = pending_output
        return pending_output

    def process_output(
        self,
        request_ids: list[str],
        token_starts: np.ndarray,
        query_start_loc: np.ndarray,
        routed_experts: np.ndarray | None,
        num_sampled: np.ndarray,
        num_rejected: np.ndarray,
        logprobs: LogprobsLists | None = None,
    ) -> dict[str, ArtifactRequestOutput]:
        """Commit one completed step and build request outputs."""
        buffer = self._buffer
        store = self._store
        outputs: dict[str, ArtifactRequestOutput] = {}

        if buffer is not None and store is not None:
            outputs.update(self._process_routed_experts(
                request_ids, token_starts, query_start_loc, routed_experts,
                num_sampled, num_rejected,
            ))
        if logprobs is not None and getattr(self, "_logprobs_active", False):
            self._process_logprobs(
                request_ids, token_starts, query_start_loc, num_sampled, logprobs
            )
        return outputs

    def _process_routed_experts(
        self,
        request_ids: list[str],
        token_starts: np.ndarray,
        query_start_loc: np.ndarray,
        routed_experts: np.ndarray,
        num_sampled: np.ndarray,
        num_rejected: np.ndarray,
    ) -> dict[str, ArtifactRequestOutput]:
        buffer = self._buffer
        store = self._store
        assert buffer is not None and store is not None
        block_size = buffer.block_size

        # Publish the whole batch before materializing any consumer output.
        materialize_outputs: list[tuple[str, int, int]] = []
        block_batches = []
        outputs: dict[str, ArtifactRequestOutput] = {}

        # Use the ModelRunner's actual batch boundaries rather than rebuilding them.
        for request_id, token_start, start, end, sampled, rejected in zip(
            request_ids,
            token_starts,
            query_start_loc[:-1],
            query_start_loc[1:],
            num_sampled,
            num_rejected,
            strict=True,
        ):
            request_num_tokens = end - start
            assert request_num_tokens > 0, (
                "artifact request token count must be positive"
            )
            state = self._requests[request_id]

            # Capture precedes speculative acceptance, so discard the rejected
            # suffix. Batch boundaries still span the full executed range.
            rejected = int(rejected)
            assert 0 <= rejected <= request_num_tokens, (
                "artifact rejected-token count is invalid"
            )
            rows = routed_experts[start : end - rejected]

            capture_start = token_start
            capture_cursor = state.capture_cursor
            if capture_cursor is None:
                capture_cursor = capture_start

            assert capture_start >= capture_cursor, "artifact capture moved backwards"
            if capture_start > capture_cursor:
                # Reattach after an optimistically scheduled suffix was rejected.
                assert capture_cursor < state.scheduled_cursor, (
                    "artifact capture has an unbacked token gap"
                )
                capture_start = capture_cursor

            emit_start = state.emit_cursor
            # Complete blocks without keys remain pending until a hash update.
            completed = buffer.capture(request_id, capture_start, rows)
            state.capture_cursor = capture_start + len(rows)
            state.scheduled_cursor = token_start + request_num_tokens
            block_batches.append((state, completed))

            token_end = capture_start + len(rows)
            if sampled > 0 and emit_start < token_end:
                if emit_start >= capture_start:
                    outputs[request_id] = ArtifactRequestOutput(
                        emit_start,
                        rows[emit_start - capture_start :],
                    )
                    state.emit_cursor = token_end
                else:
                    materialize_outputs.append((request_id, emit_start, token_end))

        # A consumer may reuse a block produced earlier in the same batch.
        self._publish_blocks(block_batches)

        for request_id, emit_start, token_end in materialize_outputs:
            state = self._requests[request_id]
            stored_end = (
                min(token_end // block_size, len(state.artifact_keys)) * block_size
            )
            if emit_start < stored_end:
                first_block = emit_start // block_size
                stored = materialize_routed_experts(
                    store,
                    state.artifact_keys[first_block : stored_end // block_size],
                    shape_per_token=buffer.shape_per_token,
                    dtype=buffer.dtype,
                )
                local_start = emit_start % block_size
                rows = stored[local_start : local_start + stored_end - emit_start]
                if stored_end < token_end:
                    rows = np.concatenate(
                        (rows, buffer.read(request_id, stored_end, token_end))
                    )
            else:
                rows = buffer.read(request_id, emit_start, token_end)
            outputs[request_id] = ArtifactRequestOutput(emit_start, rows)
            state.emit_cursor = token_end
        return outputs

    def _process_logprobs(
        self,
        request_ids: list[str],
        token_starts: np.ndarray,
        query_start_loc: np.ndarray,
        num_sampled: np.ndarray,
        logprobs: LogprobsLists,
    ) -> None:
        """Ingest one step's sampled logprobs into per-request suffixes.

        The flat rows follow generated-token order; the newest row is the
        continuation of this step's last executed token, so its absolute index
        is ``token_start + num_executed`` (e.g. ``token_start + 1`` for plain
        decode, ``token_start + chunk`` for a final prefill chunk). Rows are
        appended at those absolute indices (design §10.3, with the PREDICTED
        vs EXECUTED +1 correction), and completed blocks inside the committed
        range are published or held pending until hashes arrive (§10.5).
        """
        slices = slice_logprobs_per_request(logprobs, len(request_ids))
        batches: list[tuple[_WorkerRequestState, list[int]]] = []
        for i, request_id in enumerate(request_ids):
            state = self._requests.get(request_id)
            suffix = state.logprob_suffix if state is not None else None
            if suffix is None or state.logprob_width is None:
                continue
            if int(num_sampled[i]) == 0:
                continue
            rows = slices[i]
            if rows is None or len(rows[0]) == 0:
                continue
            num_rows = len(rows[0])
            executed = int(query_start_loc[i + 1] - query_start_loc[i])
            newest_index = int(token_starts[i]) + executed
            if newest_index < 1:
                continue
            first_index = newest_index - num_rows + 1
            suffix.append_step(rows, first_index)
            # One past the newest row; only blocks fully inside it can publish.
            committed_end = newest_index + 1
            completed = self._complete_logprob_blocks(
                state, committed_end, self._logprobs_block_size
            )
            if completed:
                state.logprob_done = completed[-1]
                batches.append((state, completed))
        self._publish_logprobs_blocks(batches)

    @staticmethod
    def _complete_logprob_blocks(
        state: _WorkerRequestState, committed_end: int, block_size: int
    ) -> list[int]:
        """Return blocks fully covered inside the committed range."""
        if state.logprob_suffix is None:
            return []
        completed: list[int] = []
        block_index = state.logprob_done + 1
        while (block_index + 1) * block_size <= committed_end:
            if not state.logprob_suffix.is_block_full(block_index, block_size):
                break
            completed.append(block_index)
            block_index += 1
        return completed

    def _publish_logprobs_blocks(
        self,
        batches: list[tuple[_WorkerRequestState, list[int]]] | None = None,
        retain_keys: Mapping[tuple[int, str], Sequence[str]] | None = None,
        release_keys: Mapping[tuple[int, str], Sequence[str]] | None = None,
    ) -> None:
        """Publish or stage logprobs blocks, per (width, mode) store."""
        if not getattr(self, "_logprobs_active", False):
            return
        batches = batches or []
        retain_keys = retain_keys or {}
        release_keys = release_keys or {}
        if not batches and not retain_keys and not release_keys:
            return
        by_store: dict[
            tuple[int, str],
            list[tuple[list[str], list[tuple[int, LogprobsLogicalSuffix]]]],
        ] = {}
        for state, completed in batches:
            if state.logprob_width is None or state.logprob_suffix is None:
                continue
            keyed_end = len(state.logprob_keys)
            ready = [k for k in completed if k < keyed_end]
            state.logprob_pending.extend(k for k in completed if k >= keyed_end)
            if ready:
                store_key = (state.logprob_width, self._logprobs_mode)
                by_store.setdefault(store_key, []).append(
                    (state.logprob_keys, [(k, state.logprob_suffix) for k in ready])
                )
        for store_key in set(by_store) | set(retain_keys) | set(release_keys):
            store = self._logprobs_stores.get(store_key)
            if store is None:
                store = self._logprobs_store(store_key[0])
            if store is None:
                continue
            publish_logprobs(
                store,
                batches=by_store.get(store_key, []),
                block_size=self._logprobs_block_size,
                width=store_key[0],
                retain_keys=retain_keys.get(store_key, ()),
                release_keys=release_keys.get(store_key, ()),
            )

    def capture_prompt_logprobs(
        self,
        prompt_logprobs_dict: dict[str, LogprobsTensors | None],
        prompt_lens: Mapping[str, int],
    ) -> None:
        """Replay cached-prefix prompt logprobs and publish a prompt's blocks.

        Called after the sampler computed a request's final prompt logprobs.
        The sampler only recomputes rows for newly executed tokens; rows for a
        prefix-cached portion have no hidden states, so they are replayed from
        the store (design §3.2). The merge result is written back to the
        ``prompt_logprobs_dict`` in place and published under the request's KV
        block keys for future prefix hits.
        """
        if not getattr(self, "_logprobs_active", False):
            return
        block_size = self._logprobs_block_size
        for req_id, fresh in list(prompt_logprobs_dict.items()):
            state = self._requests.get(req_id)
            suffix = state.logprob_suffix if state is not None else None
            if suffix is None or state.logprob_width is None:
                continue
            width = state.logprob_width
            store = state.logprob_store
            if store is None:
                continue
            prompt_len = prompt_lens[req_id]
            fresh_rows = fresh.logprobs.shape[0] if fresh is not None else 0
            # PREDICTED_TOKEN coverage of a length-L prompt is [1, L): L - 1
            # rows. Fresh rows cover [cached + 1, L), so the prefix gap is
            # [1, cached + 1) with cached = L - 1 - fresh_rows.
            cached = prompt_len - 1 - fresh_rows
            if cached < 0:
                continue
            if cached > 0:
                try:
                    cached_probs, cached_ids, cached_ranks = materialize_logprobs_range(
                        store,
                        state.logprob_keys,
                        block_size,
                        width,
                        1,
                        cached + 1,
                    )
                except Exception as error:
                    # Fail closed: a KV prefix hit must not degrade to missing
                    # prompt logprobs (design §5).
                    raise RuntimeError(
                        f"Prompt logprobs replay failed for {req_id}: KV cache "
                        "hit without a matching logprobs artifact block."
                    ) from error
            else:
                cached_probs = np.empty((0, width), dtype=np.float32)
                cached_ids = np.empty((0, width), dtype=np.int32)
                cached_ranks = np.empty(0, dtype=np.int64)
            if fresh is not None:
                fresh_probs = fresh.logprobs.detach().cpu().numpy()
                fresh_ids = fresh.logprob_token_ids.detach().cpu().numpy()
                fresh_ranks = fresh.selected_token_ranks.detach().cpu().numpy()
            else:
                fresh_probs = np.empty((0, width), dtype=np.float32)
                fresh_ids = np.empty((0, width), dtype=np.int32)
                fresh_ranks = np.empty(0, dtype=np.int64)
            merged_probs = np.concatenate((cached_probs, fresh_probs), axis=0)
            merged_ids = np.concatenate((cached_ids, fresh_ids), axis=0)
            merged_ranks = np.concatenate((cached_ranks, fresh_ranks), axis=0)
            if len(merged_probs) == 0:
                continue
            prompt_logprobs_dict[req_id] = LogprobsTensors(
                logprob_token_ids=torch.from_numpy(merged_ids),
                logprobs=torch.from_numpy(merged_probs),
                selected_token_ranks=torch.from_numpy(merged_ranks),
            )
            for offset in range(len(merged_probs)):
                suffix.set_row(
                    offset + 1,
                    LogprobRow(
                        logprobs=merged_probs[offset].copy(),
                        token_ids=merged_ids[offset].copy(),
                        rank=np.int64(merged_ranks[offset]),
                    ),
                )
            # The whole prompt is now covered; publish every full block inside
            # [1, prompt_len).
            completed = self._complete_logprob_blocks(state, prompt_len, block_size)
            if completed:
                state.logprob_done = completed[-1]
                self._publish_logprobs_blocks([(state, completed)])

    def _publish_blocks(
        self,
        batches: list[tuple[_WorkerRequestState, list[tuple[int, np.ndarray]]]],
        retain_keys: Sequence[str] = (),
        release_keys: Sequence[str] = (),
    ) -> None:
        store = self._store
        buffer = self._buffer
        assert store is not None and buffer is not None
        ready_batches = []
        for state, completed in batches:
            blocks = state.pending_blocks + completed
            keyed_end = len(state.artifact_keys) * buffer.block_size
            ready = [(start, rows) for start, rows in blocks if start < keyed_end]
            state.pending_blocks = [
                (start, buffer.retain_block(rows))
                for start, rows in blocks
                if start >= keyed_end
            ]
            if ready:
                ready_batches.append((state.artifact_keys, ready))
        if ready_batches or retain_keys or release_keys:
            publish_routed_experts(
                store,
                batches=ready_batches,
                block_size=buffer.block_size,
                retain_keys=retain_keys,
                release_keys=release_keys,
            )
        for _, blocks in ready_batches:
            for _, rows in blocks:
                buffer.release_block(rows)

    def begin_step(self, metadata: ArtifactConnectorMetadata | None) -> None:
        """Apply one scheduler step's request and block-hash updates."""
        if pending_output := self._pending_output:
            pending_output.finished.wait()
        self._step_metadata = metadata
        if self._buffer is None and not getattr(self, "_logprobs_active", False):
            return
        if metadata is None:
            return
        assert not metadata.requests.keys() & metadata.finished_requests, (
            "artifact request cannot run and finish in one step"
        )
        assert metadata.generation >= self._generation, (
            "artifact metadata generation moved backwards"
        )
        release_keys: list[str] = []
        logprobs_release: dict[tuple[int, str], list[str]] = {}
        if metadata.generation > self._generation:
            for state in self._requests.values():
                release_keys.extend(reversed(state.artifact_keys))
                if state.logprob_width is not None:
                    logprobs_release.setdefault(
                        (state.logprob_width, self._logprobs_mode), []
                    ).extend(reversed(state.logprob_keys))
            if self._buffer is not None:
                self._buffer.reset()
            self._requests.clear()
            self._generation = metadata.generation
        logprobs_batches: list[tuple[_WorkerRequestState, list[int]]] = []
        logprobs_retain: dict[tuple[int, str], list[str]] = {}
        for request_id, emit_start in metadata.requests.items():
            state = self._requests.setdefault(
                request_id, _WorkerRequestState(emit_cursor=emit_start)
            )
            assert emit_start <= state.emit_cursor, (
                "artifact Scheduler emit cursor moved ahead"
            )
            if metadata.logprobs_widths is not None:
                width = metadata.logprobs_widths.get(request_id)
                if width is not None and state.logprob_width is None:
                    state.logprob_width = width
                    state.logprob_suffix = LogprobsLogicalSuffix()
                    state.logprob_store = self._logprobs_store(width)
        block_batches: list[
            tuple[_WorkerRequestState, list[tuple[int, np.ndarray]]]
        ] = []
        retained_keys: list[str] = []
        for request_id, block_hashes in metadata.block_hashes.items():
            state = self._requests[request_id]
            keys = routed_experts_keys(block_hashes, str(self._generation))
            state.artifact_keys.extend(keys)
            retained_keys.extend(keys)
            block_batches.append((state, []))
            if (
                state.logprob_width is not None
                and state.logprob_suffix is not None
            ):
                width = state.logprob_width
                logprob_keys = logprobs_keys(
                    block_hashes,
                    str(self._generation),
                    width=width,
                    mode=self._logprobs_mode,
                )
                state.logprob_keys.extend(logprob_keys)
                logprobs_retain.setdefault((width, self._logprobs_mode), []).extend(
                    logprob_keys
                )
                pending = [
                    (k, state.logprob_suffix)
                    for k in state.logprob_pending
                    if k < len(state.logprob_keys)
                ]
                state.logprob_pending = [
                    k for k in state.logprob_pending if k >= len(state.logprob_keys)
                ]
                if pending:
                    logprobs_batches.append(
                        (state, [k for k, _ in pending])
                    )
        release_keys.extend(
            key
            for request_id in metadata.finished_requests
            for key in reversed(self._requests[request_id].artifact_keys)
        )
        if self._buffer is not None and self._store is not None:
            # R3-only: logprobs-only workers never have a routed-experts
            # store, and their (empty) R3 batches must not hit its asserts.
            self._publish_blocks(block_batches, retained_keys, release_keys)
        if getattr(self, "_logprobs_active", False):
            for request_id in metadata.finished_requests:
                state = self._requests[request_id]
                if state.logprob_width is not None:
                    logprobs_release.setdefault(
                        (state.logprob_width, self._logprobs_mode), []
                    ).extend(reversed(state.logprob_keys))
            self._publish_logprobs_blocks(
                logprobs_batches,
                retain_keys=logprobs_retain,
                release_keys=logprobs_release,
            )
        for request_id in metadata.finished_requests:
            state = self._requests.pop(request_id)
            if self._buffer is not None:
                for _, rows in state.pending_blocks:
                    self._buffer.release_block(rows)
                self._buffer.discard(request_id)
            state.logprob_suffix = None
            state.logprob_keys.clear()
            state.logprob_pending.clear()

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
        stores = getattr(self, "_logprobs_stores", {})
        for store in stores.values():
            store.close()
        stores.clear()
