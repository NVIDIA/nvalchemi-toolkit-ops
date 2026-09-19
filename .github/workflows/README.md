# CI workflows

How continuous integration is put together here, and why. Read the first two
sections before changing anything in this directory; the rest is reference.

## The one thing to know

**A failure in `cache-warm.yml` is a pull-request-speed incident, not just a red
build.** It is the only job that publishes the compiled-kernel cache every other
job restores from. When it stops publishing, pull requests silently get slower
over the following days, and the symptom looks nothing like the cause.

This is not hypothetical. It is exactly what happened before this pipeline was
rewritten: the nightly build failed at a coverage gate, the cache-save steps that
came after it never ran, and pull-request CI drifted from minutes to five hours
while every pull request still reported green.

## Why the tests are split in two

Warp generates and compiles CUDA at runtime. The PME kernels are specialised per
`(spline order, l_max, dtype)` with loops that are unrolled at code generation, so
order 6 emits 216 unrolled blocks and NVRTC takes minutes on it. On a cold cache
the first test to need such a kernel pays that cost — one eight-atom test measured
at eighteen minutes, essentially all of it compilation.

So the cost of this suite is dominated by *compilation*, not by how many tests you
select. That produces two rules the design follows:

1. Keeping the kernel cache warm matters more than trimming tests.
2. Pull requests should run a bounded, predictable subset; everything else runs
   where latency does not block a human.

## Workflows

| File | Kind | Purpose |
|---|---|---|
| `ci.yml` | orchestrator | Triggers, tier selection, coverage gate, status aggregation. No test logic. |
| `reusable-gpu-tests.yml` | `workflow_call` | Runs one test tier on a GPU runner. Called by `ci.yml`. |
| `cache-warm.yml` | producer | Publishes the Warp kernel cache and the testmon database. |
| `cleanup-pr-caches.yml` | janitor | Deletes a pull request's caches when it closes. |
| `docs.yml`, `weekly-examples.yml` | independent | Unrelated to testing. |

Shared *steps* live in `../actions/` as composite actions; shared *jobs* live here
as reusable workflows. Reusable workflows must sit in `.github/workflows/` root —
they cannot be moved into a subdirectory — so the `reusable-` prefix marks the
ones that are not independently runnable.

| Composite action | Purpose |
|---|---|
| `../actions/setup-env` | apt packages, uv, virtualenv, `uv sync`; restores the uv download cache. |
| `../actions/warp-cache` | Restore-key ladder, `WARP_CACHE_PATH` export, and cache-state reporting. |

`cache-warm.yml` is a **pure producer**: it reads nothing but the repository, so
it works on a fresh clone, a new CUDA version, or a runner that has never seen
this project. Nothing in it depends on a test job having run first. Every other
workflow only consumes what it publishes.

## Triggers

| Event | Runs |
|---|---|
| push to `pull-request/<N>` | `test-minimal` (3 shards) + `test-impacted` + coverage gate |
| `merge_group` | same as above; restores caches, writes none |
| push to `main` | `cache-warm.yml`, then `test-full` |
| `schedule` (nightly) | `cache-warm.yml`, `test-full` (4 Python × CUDA 13), `test-cuda12` |
| `pull_request_target: closed` | `cleanup-pr-caches.yml` |
| label `ciflow:all` on a PR | `test-full` instead of the fast tier |
| label `ciflow:skip` on a PR | no GPU tests |

## Test tiers

### `test-minimal` — always runs, in full

The 56 test files that hold the coverage gate for the least time, chosen by
`tools/group_coverage.py` using greedy weighted set cover over measured per-test
coverage. 4,275 of 9,665 tests, 71% coverage against a 70% gate, split across
three shards packed by measured cost.

This tier never uses testmon. It runs the same tests on every pull request, so it
is the floor of confidence and the reason the coverage gate holds no matter what
testmon decides.

### `test-impacted` — what the change actually touched

The full group list under `--testmon --testmon-nocollect`, with the
`test-minimal` files `--ignore`d so work is not duplicated. testmon deselects
everything the change cannot affect. A narrow change finishes in seconds.

The two tiers are a **union**, not a filter of one by the other. The minimal
suite always runs in full, which is what guarantees the coverage floor no matter
what testmon decides; this tier adds whatever else the change touched, from
anywhere in the suite. Coverage is combined from both before the gate is applied.

**This tier skips itself when no testmon database was restored.** Without one,
testmon deselects nothing and the tier would silently become a full-suite run.
Skipping is the safe failure: `test-minimal` still ran, and the run is annotated
with a warning. The database comes from the nightly `cache-warm.yml`.

There is otherwise no upper bound here: a change to a core electrostatics file
selects most of `test/interactions`. It runs in parallel with `test-minimal`, so
it never delays the fast verdict, but the pull request is not green until it
finishes. It carries a `timeout-minutes` for that reason.

### `test-full` — everything

Whole suite, all Python versions, and `--slow` so that slow-marked tests actually
run. Nightly and on `main`.

## Cache lifecycle

```text
                    +==========================================================+
                    |            cache-warm.yml   -- PURE PRODUCER --          |
                    |   on: push->main | schedule | workflow_dispatch          |
                    |   reads only the repo; NOT gated on test success         |
                    +====================+=====================================+
                                         | writes on refs/heads/main
                                         | => globally readable by ALL refs
                    +--------------------+---------------------+
                    v                                          v
       warp-main-py3.12-cuda13-<run>            testmon-main-py3.12-cuda13-<run>
       every merge + nightly, ~30 s warm        nightly only, runs the whole suite
       (cold ~40 min, but only ever once)       built WITHOUT coverage (see Traps)
                    |                                          |
                    v                                          v
   +------------------------------------------------------------------------------+
   |                          C O N S U M E R S                                   |
   +------------------------------------------------------------------------------+

   [A] PR CI          push -> refs/heads/pull-request/<N>
       test-minimal   3 shards, no testmon, ALWAYS runs in full   READ warp
                      shard 1 only ----------------------------> WRITE warp-pull-request-<N>
       test-impacted  --testmon --testmon-nocollect               READ warp, testmon
                      skipped entirely if no testmon db --------> WRITE nothing
       pr-coverage    combines BOTH tiers, applies the threshold once (ubuntu runner)

   [B] merge queue    refs/heads/gh-readonly-queue/main/pr-<N>-<sha>
                      ref is ephemeral ----------------------->  WRITE nothing

   [C] nightly/full   schedule | push->main | label ciflow:all
       test-full      4 python x CUDA 13, --slow                  READ warp
       test-cuda12    4 python x CUDA 12, nightly only            READ warp
                      each lane keeps its OWN cache line ------>  WRITE warp-<ref>-py<V>-cuda<N>
```

The uv download cache is handled inside `../actions/setup-env`, which both
restores and saves it, keyed on `uv.lock`. It is deliberately not published here:
any job can populate it, and a dependency change falls back to the key prefix so
only the delta is downloaded.

### Restore-key ladder

Each context walks its list top-down and stops at the first hit.

```text
  [A] PR, ref = refs/heads/pull-request/132
      1. warp-pull-request-132-py3.12-cuda13-<this run_id>   exact; only on re-run
      2. warp-pull-request-132-py3.12-cuda13-                <-- usual hit: last push
      3. warp-main-py3.12-cuda13-                            <-- first push on branch

  [B] merge queue, ref = .../gh-readonly-queue/main/pr-132-<sha>    NEW REF EVERY ITEM
      1. warp-main-py3.12-cuda13-               <-- the only rung that can hit

  [C] nightly / main
      1. warp-main-py<V>-cuda<N>-
```

A merge-queue item cannot reuse the originating pull request's cache, however
appealing that sounds. The queue ref carries the pull request number, so the key
is easy to construct — but cache reads are scoped to the current ref plus the
default branch, and a queue ref is neither the pull request's ref nor a
descendant of it. Such a rung can never hit, so it is not there.

What actually makes merge-queue items fast is `cache-warm.yml` keeping the
main-scoped baseline fresh after every merge. Queue items also write nothing:
their ref dies with the item, so a cache saved there is unreadable and
unreclaimable.

### Why warming covers only py3.12 / CUDA 13

That is the combination the pull-request tier runs, and the only lane where a
human waits. Warming all eight combinations would cost 3.6 GB of the 10 GB cache
budget; one costs 450 MB. The nightly matrix jobs restore *and* save their own
per-combination cache, so each lane stays warm from its own previous run without
a dedicated producer.

This is also why warming compiles everything registered rather than a curated
list. A measured manifest of only the modules tests use would cut a cold build
from ~40 minutes to ~18 and the cache from 450 MB to 208 MB — but the only honest
source for that list is a completed test run, which would make the producer
depend on a consumer. With one lane to warm, the saving is not worth the cycle.

The nightly matrix lanes do each save their own cache, so main ends up holding
one entry per (Python, CUDA) combination regardless. `test-full` skips saving on
py3.12/CUDA 13 specifically, because that key belongs to `cache-warm.yml` and two
writers would delete each other's entry.

### Why only shard 1 writes

GitHub allows 10 GB of cache per repository, with least-recently-*accessed*
eviction. Writing one cache per shard per pull request exhausted that budget in
practice (five pull requests once held 5.3 GB between them, and `main` held
nothing). So all three shards restore the same pull-request-scoped key and only
shard 1 saves it.

Shard 1 owns `electrostatics_torch`, which holds the expensive PME kernels, so it
is the shard whose compilation is most worth persisting. Shards 2 and 3 restore
`main`'s superset and recompile only kernels the pull request itself changed. If
that becomes the bottleneck for a particular kind of change, move the write to
whichever shard sees the most churn.

### Rotation and cleanup

Cache keys embed `<run_id>` because cache entries are immutable — a fixed key
could never be updated. The cost is that **every push creates a new entry and the
old one lingers for seven days**, so entries must be rotated explicitly:

1. **Before each save**, delete older entries with the same key prefix. Doing this
   before the save rather than after means a failed save leaves the previous cache
   in place rather than nothing.
2. **On `pull_request: closed`**, `cleanup-pr-caches.yml` deletes everything scoped
   to that pull request.
3. GitHub's seven-day inactivity sweep and 10 GB eviction remain as backstops and
   should never be what keeps us inside budget.

Caches here are scoped to `refs/heads/pull-request/<N>` — the copy-pr-bot mirror
branch — and **not** to `refs/pull/<N>/merge`. Cleanup snippets found online
usually target the latter and will silently delete nothing.

## Telling when caches are cold or stale

`../actions/warp-cache` writes the outcome to every GPU job's summary, so a slow
run can be explained without reading logs. It reports which key matched, or that
none did.

A job that falls all the way past `warp-main-` also emits a workflow warning
annotation, because that means no baseline exists and every kernel is being
recompiled — the failure mode that previously went unnoticed for weeks:

```yaml
- name: Warn on missing baseline
  if: steps.warp.outputs.cache-matched-key == ''
  run: echo "::warning::No Warp cache restored. Check cache-warm.yml on main."
```

To check the state of the world directly:

```bash
# Does a main-scoped baseline exist at all? (If not, every PR is cold.)
gh api "repos/NVIDIA/nvalchemi-toolkit-ops/actions/caches?per_page=100" \
  --jq '.actions_caches[] | select(.ref=="refs/heads/main") | .key'

# How much of the 10 GB budget is in use, by ref?
gh api "repos/NVIDIA/nvalchemi-toolkit-ops/actions/caches?per_page=100" \
  --jq '[.actions_caches[] | {ref, mb: (.size_in_bytes/1e6|floor)}]
        | group_by(.ref) | map({ref: .[0].ref, mb: (map(.mb) | add)})
        | sort_by(-.mb)'
```

## Traps

Four things in this repository behave unlike their defaults. Each has cost real
debugging time.

**Never wrap `--testmon` in `coverage run`.** Both install a `sys.settrace` hook
and Python allows one per thread; testmon wins and coverage silently reports near
zero. No error is raised. `make testmon-collect` deliberately runs without
coverage for this reason, and no coverage target passes `--testmon` in collecting
mode.

**`pytest-skip-slow` inverts the usual meaning of the marker.** Tests marked
`slow` are *skipped* unless `--slow` is passed. Before this rewrite nothing passed
it anywhere, so 335 tests ran on no CI path at all. `make test-full` passes it;
`make test-minimal` does not.

**A subdirectory `conftest.py` sees every collected item**, not only those beneath
it. `test/math` and `test/neighbors` add a `slow` marker based on test names, and
without an explicit path check they applied it across the whole tree, overriding
the electrostatics suite's deliberate decision not to treat "stress" tests as
slow. All three hooks now filter on `item.path`.

**Coverage records absolute paths.** Shards run on different runners with
different workspace roots, so `[tool.coverage.paths]` in `pyproject.toml` maps
them back onto one tree. Without it `coverage combine` reports each runner's copy
as a separate, mostly-uncovered file and the gate fails for a reason that looks
nothing like the cause.

## Maintenance

**Regenerating the pull-request suite.** Needed when a module gains behaviour the
selected files do not exercise, or when the reduced suite's coverage drifts from
the full suite's. Do not hand-edit the `ARGS_*_min` lists; their value is that
they are measured.

```bash
# 1. Per-test coverage (hours; test/interactions dominates)
bash tools/measure_test_coverage.sh

# 2. Per-test durations from any full CI job log
gh api repos/NVIDIA/nvalchemi-toolkit-ops/actions/jobs/<job-id>/logs > job.log
uv run python tools/extract_ci_durations.py job.log -o ci_durations.json

# 3. Re-select, then paste the emitted Makefile block
make minimal-suite-report
```

**Rebalancing shards.** The `SHARD_*_GROUPS` lists in the `Makefile` are packed by
measured cost, not spread round-robin, because group costs differ by two orders of
magnitude. Re-pack them whenever the suite is regenerated; the target is equal
wall-clock per shard, and shard 1 currently sets the pace.

**Before enabling `--slow` anywhere it was previously absent**, run those tests
first — they may have rotted while unexecuted:

```bash
bash tools/verify_slow_tests.sh
```
