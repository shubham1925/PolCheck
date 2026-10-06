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

## Demo (MuJoCo)

A scripted controller picks a cube and places it at a goal in Gymnasium-Robotics' Fetch pick-and-place. The suite in `demo/suites/pick.toml` has four families by object start region (`near-left`, `near-right`, `far-left`, `far-right`), 50 seeds each.

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
