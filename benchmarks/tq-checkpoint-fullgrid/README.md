# TransferQueue checkpoint full-grid benchmark

Visible benchmark and launcher source for [NeMo-RL PR #3898](https://github.com/NVIDIA-NeMo/RL/pull/3898),
derived from [Anish's benchmark](https://github.com/NVIDIA-NeMo/RL/commit/36dd799235a4bbd6853fab8376c59f724e6ab12d).
Simple and Mooncake each run once at 25 shapes: rows and context length both
drawn from `{8192, 16384, 32768, 65536, 131072}`. Each allocation uses four nodes,
two storage actors per node (eight total), CPU tensors, Mooncake CPU-RDMA, and
shared Lustre. Each case saves and loads in fresh processes. No final results
are included.

Frozen source: NeMo-RL `428520fccb081381b73cdc1efa556a3fa7a3515b` and TransferQueue
`c51614308b68c8d7a87c9b3ef62d59e14c69bde2`. Python behavior is preserved from the
recorded campaigns; added license headers do not change benchmark timing.

| Source | Purpose |
|---|---|
| [common/](common/) | Benchmark, bootstrap, profiling helper, and node runner |
| [lowrows/](lowrows/) | Rows 8192, 16384, 32768; recorded campaign `20260914T214500Z` |
| [highrows/](highrows/) | Rows 65536, 131072; parallel campaign `20260914T221200Z` |
| [render_fullgrid.py](render_fullgrid.py) | Validate summaries and render tables offline |

## Prepare and submit

The launchers retain historical OCI-HSG bindings. Prepare a fresh project-local
shared-filesystem run directory for each band. From this directory, copy the
common files and the matching band's files into one flat run directory:

```bash
set -euo pipefail
run_root=/path/to/new-low-run
test ! -e "$run_root"
mkdir -p "$run_root/slurm"
cp common/* lowrows/* "$run_root/"
```

Repeat with `highrows/*` and a different fresh run directory. In each copy:

1. Provision clean checkouts at the exact commits above and compatible submodules.
   Reuse the recorded ARM64 container/runtime and Mooncake `0.3.11.post1` wheel
   and overlay, or document a compatible replacement; these assets are external.
2. Rebind `CAMPAIGN_ROOT`, `RL_SOURCE`, `TQ_SOURCE`, `CONTAINER`,
   `MC_BENCH_OVERLAY`, and `MC_BENCH_WHEEL` in `campaign.env`; update the absolute
   `source` paths in both shell scripts, Slurm log paths, and path columns in
   `source.sha256` while preserving its expected source hashes.
3. Review Slurm account, partition, QoS, cluster assertion, and container mounts.
   Recorded resources are four ARM64 nodes, 128 CPUs and 920G RAM per node, four
   reserved GPUs per node (hidden from the benchmark), and a two-hour limit.
   Review RDMA devices, GID/port, `uv`/Python paths, and node-runner hardware checks.
4. For a new low-row reproduction, remove the obsolete `PREDECESSOR_ROOT`
   binding and its two preflight checks in `run.sbatch` (the predecessor's
   `completed.ok` and 11-line summary requirement). Do not fabricate those files.
5. Keep the grid, backend ordering, payloads, timers, verification, eight-actor
   placement, 48 GiB/owner store, 1 GiB buffer, and 3000 GiB free-space guard.
   Keep caches and outputs project-local. Record the binding changes.
6. From each prepared run directory, generate `launcher.sha256` for the actual
   copied files after rebinding; do not reuse a historical launcher file list:

```bash
sha256sum benchmark_bootstrap.py checkpoint_profile.py node_runner.py \
  tq_checkpoint_benchmark.py run_subset.py run.sbatch node_entry.sh campaign.env \
  source.sha256 rl-status.expected tq-status.expected > launcher.sha256
sha256sum --check launcher.sha256
sha256sum --check source.sha256
sbatch --chdir="$PWD" run.sbatch
```

Record both job IDs. The allocations can overlap and contend for Lustre. Before
publishing, check Slurm completion, application markers, source/runtime evidence,
and actor placement. Run `uv run --no-project --no-sync python render_fullgrid.py --help` for the two summary
CSV paths, recorded job statuses, and fresh output-directory arguments. The
renderer checks CSV completeness; it does not independently attest runtime facts.

## Measurement limits

Timers cover only `tq.save_checkpoint(...)` and `tq.load_checkpoint(...)` wall
time, excluding startup, insertion, and verification. Fresh processes do not
imply cold filesystem caches. Verification checks full key/row and byte counts
plus exact tensors/tags for 64 sampled rows. Backend formats and durability
behavior differ. One sample per cell provides no repeatability estimate.
