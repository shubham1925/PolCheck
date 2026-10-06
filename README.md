# polcheck

Regression testing for robots that run learned policies. Compares a new policy version's simulated runs against the previous version's and answers: is it meaningfully worse, where, and how sure are we?

Early development. The build spec is [.agents/BUILD_PLAN.md](.agents/BUILD_PLAN.md); design decisions are in [docs/DECISIONS.md](docs/DECISIONS.md).

## Development

```sh
uv sync
uv run pytest
uv run ruff check
uv run mypy src
uv run pre-commit install   # optional: run the checks on every commit
```

## Recording your own runs

polcheck never runs your simulator. You run your evaluation as usual and hand polcheck the results, in one of two ways.

### Option A: record time-series with the `Recorder`

Use this when you can read the robot's state each step. It enables every measure, including smoothness, contact force and clearance.

**1. Describe your scenarios in a suite file.** `config` is free-form; only your own code interprets it.

```toml
[suite]
name = "shelf-pick"
version = 1

[[scenario]]
family = "top-shelf"
config = { shelf_height = 1.4, clutter = 3 }
seeds = { start = 0, count = 50 }
```

**2. Wrap your evaluation loop.** You write the part that reads your simulator's state (an "adapter"); polcheck only sees plain arrays. `demo/envs.py`'s `SignalLogger` is a worked example for MuJoCo.

```python
from polcheck.recorder import Contact, Recorder
from polcheck.schema import SimulatorInfo

rec = Recorder(
    ".polcheck",  # store directory, created if missing
    "shelf-v14",  # the policy version being evaluated
    "suites/shelf.toml",  # suite file; afterwards "shelf-pick@1" also works
    simulator=SimulatorInfo(name="isaac-sim", version="4.5"),
    physics_hash="...",  # any string that changes when physics settings change
)

for scenario in rec.suite.scenarios:
    for env_seed in scenario.seeds.seeds():
        with rec.run(scenario=scenario, env_seed=env_seed) as run:
            ...  # reset your simulator from scenario.config and env_seed
            while not done:
                ...  # step your policy and simulator
                run.log(
                    t=t,
                    joint_pos=q,
                    joint_vel=dq,
                    ee_pos=p,
                    ee_quat=quat,
                    objects={"box": (box_pos, box_quat)},
                    clearance=min_dist,
                )
                run.log_contacts(t=t, contacts=[Contact("finger_l", "box", 3.2, True)])
            run.finish(
                success=ok, failure_reason=None if ok else "timeout", metrics={"task_score": score}
            )  # optional extra numbers you already have

batch_id = rec.close()
```

Signals you can log (all optional; leave out what you can't provide, and measures that need it are reported as skipped):

| Argument | Shape | Meaning |
|---|---|---|
| `t` | scalar | time in seconds; must increase every step |
| `joint_pos`, `joint_vel`, `joint_torque` | n joints | rad, rad/s, N·m |
| `ee_pos` | 3 | end-effector position, metres, world frame |
| `ee_quat` | 4 | end-effector orientation, **w, x, y, z** |
| `objects` | name → (pos 3, quat 4) | tracked objects, same conventions |
| `clearance` | scalar | minimum distance from the robot to its surroundings, metres |
| `qpos` | n | full MuJoCo state, only for replay video |

Contacts are `Contact(geom_a, geom_b, force_norm, intended)`. Mark contacts the task requires (e.g. fingers on the object) as `intended=True`; the rest count as unintended contact.

Every step of a run must log the same signals. If your code raises inside a run, that run is stored as failed and the batch continues. To let polcheck compare seed-by-seed (more sensitive), re-run a few seeds within the batch so it can confirm your setup is deterministic (see `--repeat-check` in `demo/run_suite.py`).

### Option B: import per-episode results from a file

If your evaluation tool already writes one row per episode, import it directly. Map your column names in `polcheck.toml` (see `polcheck.toml.example`):

```sh
uv run polcheck ingest results.csv --reader tabular --policy shelf-v14 --suite suites/shelf.toml
```

CSV and JSONL work. Without time-series, only success rate and your own metric columns can be compared.

### Then: compare

Comparing a candidate batch against a baseline (`polcheck compare`) is not built yet; it arrives in milestones M4 (statistics) and M7 (CLI and reports).

## Demo (MuJoCo)

A scripted controller picks a cube and places it at a goal in Gymnasium-Robotics' Fetch pick-and-place. The default suite, `demo/suites/pick2.toml` (`pick@2`), has four families by object start region (`near-left`, `near-right`, `far-left`, `far-right`), 50 seeds each. The policy sees the object through 12 mm of simulated perception noise and far starting positions are near the arm's reach, so the scripted controller succeeds on about 95% of seeds, not all of them. `pick.toml` (`pick@1`) is the easier original, kept for reference.

```sh
uv sync --extra demo
uv run --extra demo python -m demo.run_suite --policy-version scripted-v1 --seeds-per-family 25
uv run --extra demo pytest tests/test_demo.py   # skipped unless the demo extra is installed
```

Plain `uv run` syncs to the default environment and uninstalls MuJoCo, so keep `--extra demo` on every demo command (or re-run `uv sync --extra demo`).

### Watching an episode

`demo/watch.py` plays one episode, either live in a MuJoCo window or saved as a video. Nothing is written to the polcheck store; recorded runs come from `demo.run_suite`.

```sh
# Live window (needs a display; on WSL2 this works through WSLg)
uv run --extra demo python -m demo.watch --family far-left --seed 3

# Slower playback: 1 = real time (default 0.5); a full episode is under a second
uv run --extra demo python -m demo.watch --family far-left --seed 3 --speed 0.2

# Save an MP4 instead (renders off-screen, so no display needed)
uv run --extra demo python -m demo.watch --family near-right --seed 7 --video ep.mp4
```

`--grasp-height` (metres, default 0) raises the scripted grasp point to make a weaker policy: 0.033 fails about 10% of episodes and 0.04 nearly all of them. Pass the same `ScriptedPick(grasp_height=...)` to `demo.run_suite.run_suite` to record that variant.

The episode is simulated in full before anything is drawn, then replayed. Rendering mid-episode perturbs the physics enough to flip borderline outcomes, so this is what keeps the viewer in agreement with recorded runs.

`--family` is one of the four families above and `--seed` is 0–49. The script prints the outcome, steps taken and final distance to the goal. The cube is the black box and the goal is the red dot.

The live window uses MuJoCo's GLFW backend and video uses EGL. Set `MUJOCO_GL` to override the backend. Replaying runs already recorded in the store (`polcheck render RUN_ID`) comes later, in M8.
