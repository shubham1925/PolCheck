"""Watch one scripted episode: live in a MuJoCo window, or saved as a video.

    uv run --extra demo python -m demo.watch --family far-left --seed 3
    uv run --extra demo python -m demo.watch --family near-right --seed 7 --video ep.mp4
    uv run --extra demo python -m demo.watch --family far-left --seed 3 --grasp-height 0.02

Nothing is recorded to the polcheck store; use demo.run_suite for that.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# Not imported from demo.run_suite: that would import MuJoCo before MUJOCO_GL is set.
DEFAULT_SUITE = Path(__file__).parent / "suites" / "pick.toml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--family", default="far-left")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--video", type=Path, help="save an MP4 here instead of opening a window")
    parser.add_argument(
        "--speed", type=float, default=0.5, help="live playback speed (1 = real time)"
    )
    parser.add_argument(
        "--grasp-height",
        type=float,
        default=0.0,
        help="grasp point above the object's centre in metres; e.g. 0.02 for the weak variant",
    )
    args = parser.parse_args()

    # Pick the GL backend before MuJoCo is imported: EGL renders off-screen for
    # video, GLFW opens a window (needs a display, e.g. WSLg).
    os.environ.setdefault("MUJOCO_GL", "egl" if args.video else "glfw")

    import gymnasium as gym
    import mujoco
    import numpy as np

    from demo.envs import ENV_ID, MAX_STEPS, reset_to_scenario
    from demo.scripted import ScriptedPick
    from polcheck.suite import load_suite

    suite = load_suite(args.suite)
    scenarios = [sc for sc in suite.scenarios if sc.family == args.family]
    if not scenarios:
        parser.error(f"no family {args.family!r} in {args.suite}")

    env = gym.make(
        ENV_ID,
        max_episode_steps=MAX_STEPS,
        render_mode="rgb_array" if args.video else "human",
    )
    policy = ScriptedPick(grasp_height=args.grasp_height)
    obs = reset_to_scenario(env, scenarios[0], args.seed)
    policy.reset()
    sim = env.unwrapped
    dt = sim.dt  # type: ignore[attr-defined]
    model, data = sim.model, sim.data  # type: ignore[attr-defined]

    # Simulate the whole episode before rendering anything: Fetch's render
    # callback calls mj_forward, which perturbs the physics enough to flip
    # borderline episodes. Saving each step's state and replaying it keeps the
    # outcome identical to an unrendered demo.run_suite run.
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, spec))

    def snapshot() -> np.ndarray:
        mujoco.mj_getState(model, data, state, spec)
        return state.copy()

    states = [snapshot()]
    success, steps = False, 0
    while steps < MAX_STEPS and not success:
        obs, _, _, _, info = env.step(policy.act(obs))
        steps += 1
        states.append(snapshot())
        success = bool(info["is_success"])

    frames = []
    for s in states:
        mujoco.mj_setState(model, data, s, spec)
        mujoco.mj_forward(model, data)
        if args.video:
            frames.append(env.render())
        else:
            env.render()
            time.sleep(dt / args.speed)

    distance = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
    outcome = "success" if success else "timeout"
    print(
        f"{args.family} seed {args.seed}: {outcome} after {steps} steps "
        f"({steps * dt:.2f} s sim), final goal distance {distance * 100:.1f} cm"
    )

    if args.video:
        import imageio.v2 as imageio

        hold = frames[-1:] * int(1 / dt)  # hold the last frame for a second
        imageio.mimsave(args.video, frames + hold, fps=round(1 / dt))
        print(f"saved {args.video}")
    else:
        time.sleep(2.0)  # leave the final pose on screen briefly
    env.close()


if __name__ == "__main__":
    main()
