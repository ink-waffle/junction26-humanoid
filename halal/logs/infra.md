# infra: sim throughput (14.11 night)

Goal: more simulated seconds per wall second from the Isaac sims on the one dev instance (30 vCPU, one RTX PRO 6000),
with process-level settings only. Nothing under `aaltonomous/` or `third_party/` was touched.

## Result

`sims.sh start <tag>` now starts a sim with `OMP_NUM_THREADS=1`, pinned to four cores of its own (a 0-3, b 4-7,
c 8-11), with Kit's task pool at 4 threads and PhysX at 2 worker threads. `env.sh` pins the sourcing shell and its
children to the remaining cores 12-29.

Sim c, sims a and b busy with other people's jobs the whole time:

| | rtf of c, no client | rtf of c, oracle `sort_fixed` (DR train) | CPU of c |
|---|---|---|---|
| before (old `sims.sh`) | 0.29 (a, b old) | not measured | 5.5 cores |
| `OMP_NUM_THREADS=1` only | 0.44 (a, b old) | 0.32 (a, b old), 0.37 (a, b on OMP=1) | 2.4 cores |
| new defaults | 0.58 (a old, b on OMP=1) | 0.43–0.45 (a, b moving to OMP=1) | 1.4 cores |

The load changed during the night, so read the rows with the bracket. Like for like: no client, a and b on old
settings, E3 against E9 below (0.44 → 0.585); under the oracle with a and b on OMP=1, E11 against E10c (0.37 → 0.445).

During the night a and b were restarted with `OMP_NUM_THREADS=1` only (b 01:05, a 01:10 UTC). With c on the new
defaults under the oracle, the three sims then read 0.44 / 0.39 / 0.47 = 1.30 sim-s per wall second (01:13 UTC),
against 2 × 0.27 = 0.54 at the start of the night.

Not measured: all three sims on the new defaults at once.

## Why the sims were slow

`perf` on sim c with the old settings: 81 % of its CPU samples are in libgomp's barrier spin-wait. torch on
`--device cpu` opens an OpenMP pool with one thread per core (the "30 threads at ~30 %" in `top -H`). The tensors are
`num_envs=1`, so the pool does no useful work; after every small torch op 29 workers spin for a while. That is about
8 cores per sim, so two sims already saturated the scheduler (load 35–40) while the GPU sat at 8–16 %.

With the pool at one thread, the next cost is Kit's own pools: 16 carb.tasking threads, 8 PhysX workers, 16 TBB
workers, all for one robot and 18 objects. Fewer threads step faster: less wake-up and hand-over per step.

## Measurements

All on sim c, 10 s windows after 45–60 s of settling, `AALTO_PROFILE=1`. "step" is `controller.step` per 10 ms sim
step; physics is both substeps together; camera is the `camera_image` observation term, which is where Isaac Lab
renders the three camera sensors; aalto is the scene service plus head RGB-D. Load = rtf of a and b at that moment
(both on the old settings up to E9) and the 1-min load average.

| # | sim c configuration | client | rtf c | step ms | physics | render | camera | aalto | CPU c | a / b rtf, load |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | old defaults | none | 0.29 | 33.5 | 8.4 | 7.2 | 7.6 | 2.2 | 555 % | 0.18 / 0.16, 50 |
| E2 | OMP=1 | none | 0.44 | 21.9 | 3.6 | 6.5 | 6.6 | 1.5 | 239 % | 0.25 / 0.26, 46 |
| E2b | OMP=1 | oracle | 0.32 | 30.1 | 4.0 | 7.4 | 7.1 | 7.0 | n/a | 0.25 / 0.23, n/a |
| E3 | OMP=1, cores 24-29 | none | 0.44 | 22.0 | 3.7 | 6.5 | 6.5 | 1.5 | 155 % | 0.24 / 0.21, 39 |
| E4 | E3 + tasking 4, PhysX 4, PXR 4 | none | 0.555 | 17.2 | 1.7 | 3.7 | 6.4 | 1.4 | 151 % | 0.25 / 0.21, 41 |
| E5 | E4 without pinning | none | 0.535 | 17.9 | 2.2 | 3.7 | 6.6 | 1.5 | 249 % | 0.25 / 0.22, 35 |
| E6 | OMP=1, cores 26-29 | none | 0.475 | 20.2 | 1.6 | 6.6 | 6.4 | 1.4 | 128 % | 0.27 / 0.22, 53 |
| E7 | E3 + tasking 2, PhysX 2, PXR 2 | none | 0.585 | 16.4 | 1.4 | 3.6 | 6.0 | 1.3 | 148 % | 0.26 / 0.23, 39 |
| E8 | E3 + tasking 4, PhysX 0, PXR 4 | none | 0.59 | 16.2 | 1.2 | 3.5 | 6.0 | 1.4 | 151 % | 0.25 / 0.21, 39 |
| E9 | E3 + tasking 4, PhysX 2, PXR 4 | none | 0.585 | 16.4 | 1.4 | 3.5 | 6.1 | 1.4 | 152 % | 0.27 / 0.23, 39 |
| E10 | new `sims.sh` defaults (cores 8-11) | none | 0.58 | 16.6 | 1.45 | 3.7 | 6.1 | 1.4 | 137 % | 0.31 / 0.33, 28 |
| E10b | same | oracle | 0.43 | 22.4 | 1.6 | 3.8 | 6.2 | 6.4 | 143 % | 0.30 / 0.32, 25 |
| E10c | same, client pinned by `env.sh` | oracle, 2 trials | 0.445 | 21.4 | 1.5 | 3.7 | 5.8 | 6.1 | n/a | 0.31→0.44 / 0.32→0.39, 26→10 |
| E11 | OMP=1, PhysX 8 passed explicitly, unpinned | oracle | 0.37 | 26.0 | 2.6 | 7.5 | 5.7 | 6.0 | n/a | 0.43 / 0.37, 9 |
| E12 | new defaults but PhysX 8 | none | 0.46 | 21.0 | 3.4 | 7.0 | 5.7 | 1.25 | n/a | 0.43 / 0.36, 10 |
| E12b | same | oracle | 0.38 | 26.2 | 3.5 | 7.2 | 5.5 | 5.8 | n/a | 0.44 / 0.39, 12 |

"tasking" = `--/plugins/carb.tasking.plugin/threadCount`, "PhysX" = `--/persistent/physics/numThreads`, both through
`--kit_args`; "PXR" = `PXR_WORK_THREAD_LIMIT`. From E10 on b runs with `OMP_NUM_THREADS=1`, from the second half of
E10c a does too, so E10 and later are under lighter load than E1–E9. E6 most likely ran with 4 PhysX threads, not
the default 8: it started right after E5 exited and inherited E5's value through the user config (see "Things to
know"); its physics time says so, the value was not read back.

## Oracle episodes (`sort_fixed`, DR train, `--no-cameras`)

Every episode: 0 safety clips, command rate 29.99–30.0 Hz, sim gate pass.

| run | sim c configuration | seed | correct | ticks | sim s | wall s | rtf | policy rt Hz | blue grasp |
|---|---|---|---|---|---|---|---|---|---|
| E2b | OMP=1 only, a and b on old settings | 1000 | 3/3 | 1228 | 40.9 | 127.5 | 0.32 | 92.6 | 1 retry |
| E10b | new defaults | 1000 | 3/3 | 955 | 31.8 | 73.7 | 0.43 | 121.1 | first try |
| E10c | new defaults, client pinned | 1000 | 3/3 | 955 | 31.8 | 72.7 | 0.44 | 121.9 | first try |
| E10c | same | 1001 | 4/5 | 2654 | 88.5 | 196.0 | 0.45 | 135.5 | n/a |
| E11 | OMP=1 only, light load, oracle connected at sim t 0.3 s | 1000 | 3/3 | 1507 | 50.2 | 137.2 | 0.37 | 107.7 | 2+ retries |
| E11b | same sim, second episode | 1000 | 3/3 | 943 | 31.4 | 84.7 | 0.37 | 105.2 | first try |
| E12b | new defaults but PhysX 8 | 1000 | 3/3 | 949 | 31.6 | 84.0 | 0.38 | 108.3 | first try |

The sim is not deterministic: the same seed on the same configuration (OMP=1 only) gave 1228, 1507 and 943 ticks,
because the first grasp of the blue object sometimes fails and the oracle retries. On the new defaults it held on
the first try both times. Seven episodes do not separate a configuration effect from this run-to-run variation.

## Verdicts

- `OMP_NUM_THREADS=1`: kept. The largest single step (0.29 → 0.44) and it frees 3 cores per sim.
- PhysX 2 workers: kept. It is the second large step: same configuration with 8 workers (E12) against 2 (E10):
  physics 3.4 → 1.45 ms, render 7.0 → 3.7 ms, rtf 0.46 → 0.58 without a client and 0.38 → 0.44 under the oracle.
  0, 2 and 4 workers are within 5 % of each other (E8, E9, E4).
- Kit task pool 4: kept, on weaker evidence. Render is 6.6 ms with the default pool (E6) and 3.7 ms with 4 (E4) at
  4 PhysX threads, if E6's inherited PhysX count is what I think it is. Pool 2 and 4 are equal (E7, E9).
- Pinning to own cores: kept for isolation, not for speed. Pinning alone changed nothing (E2 vs E3); with small
  pools the pinned run was 4 % ahead of the unpinned one (E4 vs E5) and used 1 core less. Four cores are enough
  (E10: 1.4 cores used).
- `PXR_WORK_THREAD_LIMIT`: dropped. It did not cap the TBB pool (15 workers in E5); pinning does (the pool follows
  the affinity mask), and E10 without it matches E9 with it.
- Client pinning in `env.sh`: kept by design, not measured. The evaluator uses 0.35 cores and shows no OpenMP or
  BLAS spinning in `perf`, so clients need no thread settings.

## Things to know

- `--/persistent/physics/numThreads` is a persistent Kit setting: every Kit process writes its persistent settings to
  the shared `user.config.json` on exit. A sim started by another path (`aalto sim`, an old `sims.sh`) therefore gets
  2 or 8 PhysX threads depending on which sim exited last. Sims started by the new `sims.sh` always pass the value.
- The first start of a third concurrent sim took 335 s: Kit keeps one shader cache per concurrent instance slot and
  slot 2 was empty. It is filled now (16–17 s per start since). A fourth concurrent sim would pay this once more.
- A sim that had a client call `randomize` keeps the DR camera corruption running after the client leaves: its rtf
  without a client stays at the with-client value until restart.

## Costs in organiser code (not patched)

- DR camera corruption of the head RGB-D (`hooks._publish_rgbd` → `dr.corrupt`): the `aalto` term goes from 1.4 to
  6.4 ms per step once `randomize` has been called, 5 ms of a 22.4 ms step (22 %). `perf` shows numpy's
  `random_standard_normal` and zlib in the main thread.
- A Python profile hook is active in the sim's main thread (`sys_profile_call_or_return`, `call_profile_func`:
  about 3 % of the process's samples). Its origin was not identified.

## Dead ends and not done

- Four or more sims: not tried (a and b were in use, a fourth instance compiles shaders for minutes first).
- Process priorities: not tried; after the change the CPU is no longer saturated by the sims.
- Whether the pool sizes shift grasp outcomes: open, see the oracle table. A grasp experiment that compares numbers
  across the restart should repeat its reference on the new settings.
