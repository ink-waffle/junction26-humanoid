# Live sim leaderboard (S7)

Reads the `runs/eval/<team>_<task>_<variant>_<stamp>/eval.json` dirs that `aaltonomous/sim/evaluate.py` writes and
shows a ranked board for the venue screen. Stdlib only (PyYAML for the config if installed). No sim or GPU needed.

## At the venue

```bash
# on the machine that holds the eval results (run evals with --team <name>!)
python3 -m aaltonomous.leaderboard.serve --runs runs/eval --port 8090
#  -> http://<host>:8090/            big dark page, auto-refresh every 20 s (venue screen, full-screen browser)
#  -> http://<host>:8090/leaderboard.json
#  -> runs/leaderboard.html          static copy, rewritten on every scan: post it to the team chat

python3 -m aaltonomous.leaderboard.serve --runs runs/eval --once     # just write the static page + print the table
python3 -m aaltonomous.leaderboard.ingest --runs runs/eval [--json]  # table / JSON in the terminal
```

Several result dirs: `--runs runs/eval /mnt/verda/runs/eval`. Other rules: `--config my.yaml`.
Tests: `python3 -m unittest aaltonomous/leaderboard/test_leaderboard.py`.

## Ranking (edit `leaderboard.yaml`)

- Only runs on `variant: eval_heldout` with at least `min_trials` finished trials count.
- Per team and task, the **best** (or `select: latest`) **gate-passing** run is used. Team score =
  sum of `weight × score_mean` (mean episode score, rules §3–4). Default weights: rung 1 = rung 2 = 1, rung 3 shown but
  unranked; rung 2 already pays +2 per object.
- Gate badge: PASS (every weighted task the team tried has a passing run), PARTIAL, FAIL. Only PASS teams are ranked
  (`finalist_gate: any` also admits PARTIAL). The gate is re-checked against `gate_max_clips`/`gate_min_hz`, so an eval
  run with looser `--gate-*` flags doesn't slip through.
- Top `finalists` (5) ranked teams are highlighted. Ties: the team that reached the score first ranks higher.
- `reference_teams` (oracle, policy, baseline) are greyed out and never ranked. `aliases` merges misspelt team names.
- Team name = `summary.team` (evaluate `--team`), else the dir-name prefix.

## Robustness

A dir without `eval.json` is an eval in progress: shown under "Evaluating now" with the trials finished so far (from
the `episode_*.jsonl` summaries), never ranked. Unparseable or empty `eval.json` is listed in `errors` in the JSON
(fresh files < 30 s are treated as still being written). Missing summary fields are recomputed from `trials`.
