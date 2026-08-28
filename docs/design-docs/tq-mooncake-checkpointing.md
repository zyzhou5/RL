# TQ checkpointing with Mooncake

## Status

Experimental NeMo-RL plugin, disabled by default. The implementation uses the
TQ and Mooncake versions pinned by NeMo-RL; it does not require an upstream TQ
change.

## Scope

The plugin makes every payload referenced by a TQ controller checkpoint
durable. It is field-agnostic: tokens, log probabilities, masks, router
indices, non-tensor values, and future TQ fields all follow the same protocol.

This is data-plane checkpointing. Model weights, unfinished token streams,
vLLM KV cache, and arbitrary Gym/sandbox state remain outside its scope.

## Design

Mooncake can create one shared-filesystem DISK replica for every object. The
plugin launches `mooncake_master` with a shared `root_fs_dir` and a fresh
per-launch `cluster_id`. The Mooncake client issuing each PUT/upsert writes that
DISK replica asynchronously, and the master reports it as `COMPLETE` only after
the file write succeeds.

```text
TQ writer -> Mooncake client -> distributed memory replicas
                           \-> async DISK replica -> Lustre/live/<session>/...

TQ controller snapshot -> exact physical-key inventory
                                  |
                         wait for COMPLETE
                                  |
             copy + hash live files into TQ checkpoint
                                  |
                    publish storage_saved=true
```

NeMo-RL installs a Mooncake storage-manager and bootstrap override through TQ's
process-local registries before `tq.init()`. Normal Mooncake transfer paths are
unchanged. The direct DISK mechanism is distinct from TQ's centralized SSD
offload client, which remains disabled.

Producer-side Mooncake clients create the live durable replicas; payloads are
not fetched from distributed memory through the Single Controller. The first
implementation does use the TQ manager process for the Lustre-to-Lustre copies
that make a checkpoint immutable. A copy is required because an upsert reuses a
live key's file path after training resumes.

TQ can recycle a cleared row's global index, and Mooncake writes each DISK
replica asynchronously. For checkpoint-enabled Mooncake, the plugin therefore
does not let TQ release an index until Mooncake has finished and retired every
physical object for that row, including GDR chunk keys. A sequential upsert of
an existing physical key likewise waits for its previous DISK generation before
starting the next one. The plugin owns the raw Mooncake retry loop so every
failed attempt is fenced or retired before another attempt can reuse its live
file path. Concurrent writes to the same physical key remain unsupported.

## Save

1. The recovery coordinator prevents writes and clears that could mutate the
   selected TQ state. Single Controller supplies this barrier in its native
   checkpoint path; synchronous callers must provide equivalent quiescence.
2. TQ writes `controller_state.pkl` to its temporary checkpoint directory.
3. The plugin enumerates every produced row/field pair. GDR chunk metadata is
   expanded into the physical `:c0`, `:c1`, ... keys.
4. It waits for exactly one `COMPLETE` direct DISK replica for every physical
   key.
5. It copies the referenced files into `mooncake_storage/`, records sizes and
   SHA-256 digests, and commits a manifest.
6. TQ writes `metadata.json` with `storage_saved=true` and publishes its
   temporary directory by rename. NeMo-RL rejects TQ's metadata-only
   `storage_saved=false` fallback.

## Restore

1. Initialize a clean TQ/Mooncake system with a new empty live session and all
   memory-providing clients present, before starting any producer or other
   writer.
2. Validate the complete manifest, paths, sizes, and hashes before mutating
   Mooncake; reject any pre-existing target key. Restore requires the exact
   saved `use_gdr` and `gdr_staging_buffer_mb` values, because pinned TQ chooses
   base versus chunk keys and derives chunk offsets from the current layout.
3. PUT the saved bytes into Mooncake without decoding individual fields.
4. Fence every bounded restore batch until its new DISK replicas are
   `COMPLETE`.
5. Let TQ load its controller state only after all payloads are restored.

The final ordering is TQ's existing contract:

```python
if meta.get("storage_saved"):
    client.load_storage_checkpoint(checkpoint_dir)
client.load_controller_checkpoint(controller_path)
```

## Partial-rollout recovery

The plugin only restores TQ. The current Single Controller recovery path uses
that state to reuse completed canonical groups; it does not yet resume a
rollout at a completed-call boundary. A future `rollout_staging` integration
could let Single Controller resume a completed call when external state is
recoverable or redispatch it otherwise. Unfinished generations and missing Gym
state are not made recoverable by this storage plugin.

## Costs and constraints

- Every live Mooncake PUT/upsert also writes to Lustre while the feature is on.
- A sequential repeated-key upsert performs a Mooncake metadata query first;
  clearing a row can wait for its in-flight DISK write before TQ may recycle
  that global index. First-time PUTs do not wait for Lustre completion.
- Mooncake multi-key removal is not atomic. The plugin converges partial and
  transient results idempotently before TQ releases indexes; a retirement
  timeout makes that live session recovery-required rather than safe to keep
  using.
- Every checkpoint copies and hashes every referenced payload. Checkpoint-time
  I/O is therefore comparable in kind to a full SimpleStorage checkpoint, in
  addition to the ongoing live-replica writes.
- The first implementation copies files in one TQ manager process. That phase
  may be parallelized later without changing the manifest.
- Mooncake disk eviction is disabled. Cleared live keys can leave stale files,
  so deployments must budget and manage live-session storage.
- Every node must see the same `storage_root` path.
- A TQ checkpoint destination must not contain the active Mooncake live tree;
  the plugin rejects layouts where TQ's destination replacement would
  delete live replicas.
- Restore is non-transactional after validation; retry from a fresh system.
- Mooncake maps keys to sanitized live filenames non-injectively. The plugin
  rejects any two checkpoint keys that resolve to one live path during save or
  restore instead of accepting an ambiguous durable replica.
- Restore requires global writer quiescence through controller load. The
  empty-key preflight is defensive validation, not a transactional lock.
- GDR is opt-in through `mooncake_cpu.use_gdr`. Every intended producer and
  consumer must initialize CUDA before constructing its Mooncake client; the
  cluster test must verify that the GDR staging path actually activated rather
  than silently falling back to CPU RDMA. Chunked checkpoints require the same
  positive `gdr_staging_buffer_mb` on restore. Homogeneous effective activation
  across clients is a runtime precondition that configuration alone cannot
  prove in this plugin.
- The pinned GDR path also requires `cuda-python` and does not validate its
  buffer-registration return code. Treat the two-node GDR smoke test below as
  a release gate, not optional coverage.
- Any future writer to a new TQ partition, including `rollout_staging`, must
  join the checkpoint barrier before that partition has atomic snapshot
  semantics.
- The plugin stops only the exact `mooncake_master` process it launches. It
  deliberately preserves live-session files for explicit storage management.
- Pinned TQ deletes an existing destination before renaming a new checkpoint
  into place, so replacement of the same checkpoint directory is not atomic.
  Use step-specific directories and retain the previous completed checkpoint.
- This protocol targets Ray/Slurm process and job restarts. Mooncake's direct
  POSIX writer closes live files but does not fsync them, so the proof of
  concept does not claim power-loss or filesystem-metadata-crash atomicity.

## Required cluster validation

The acceptance test is a two-node, sixteen-GPU save/kill/restart/load cycle for
both CPU RDMA and GDR. It must include tensors, non-tensors, log probabilities,
router indices, and a chunked GDR object; compare every restored field and TQ
consumption state; then continue a normal write/read/clear cycle. Missing,
corrupt, and interrupted checkpoints must all fail closed. This distributed
acceptance test is still required; unit coverage alone does not establish GDR
or multi-node correctness.
