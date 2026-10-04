"""Run a policy over a suite in Fetch pick-and-place and record it with polcheck.

uv run python -m demo.run_suite --policy-version scripted-v1 --seeds-per-family 25
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

import gymnasium as gym
import numpy as np

from demo.envs import MAX_STEPS, SignalLogger, make_env, physics_hash, reset_to_scenario
from demo.envs import simulator_info as sim_info
from demo.scripted import ScriptedPick
from polcheck.recorder import Recorder, RunContext
from polcheck.schema import Scenario
from polcheck.store import Store

DEFAULT_SUITE = Path(__file__).parent / "suites" / "pick.toml"


class Policy(Protocol):
    def reset(self) -> None: ...
    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray: ...


def record_episode(
    env: gym.Env[Any, Any],
    logger: SignalLogger,
    policy: Policy,
    scenario: Scenario,
    env_seed: int,
    run: RunContext,
    max_steps: int = MAX_STEPS,
) -> None:
    obs = reset_to_scenario(env, scenario, env_seed)
    policy.reset()
    logger.log(run)
    success = False
    for _ in range(max_steps):
        obs, _, _, _, info = env.step(policy.act(obs))
        logger.log(run)
        success = bool(info["is_success"])
        if success:
            break
    goal_distance = float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]))
    run.finish(
        success=success,
        failure_reason=None if success else "timeout",
        metrics={"final_goal_distance": goal_distance},
    )


def run_suite(
    policy: Policy,
    policy_version: str,
    *,
    store: Store | str | Path = ".polcheck",
    suite: str | Path = DEFAULT_SUITE,
    seeds_per_family: int | None = None,
    families: Iterable[str] | None = None,
    max_steps: int = MAX_STEPS,
) -> UUID:
    env = make_env(max_steps)
    logger = SignalLogger(env)
    wanted = set(families) if families is not None else None
    with Recorder(
        store, policy_version, suite, simulator=sim_info(), physics_hash=physics_hash(env)
    ) as rec:
        for scenario in rec.suite.scenarios:
            if wanted is not None and scenario.family not in wanted:
                continue
            for env_seed in list(scenario.seeds.seeds())[:seeds_per_family]:
                with rec.run(scenario=scenario, env_seed=env_seed) as run:
                    record_episode(env, logger, policy, scenario, env_seed, run, max_steps)
    env.close()
    return rec.batch_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--policy-version", required=True)
    parser.add_argument("--store", default=".polcheck")
    parser.add_argument("--suite", default=str(DEFAULT_SUITE))
    parser.add_argument("--seeds-per-family", type=int, default=None)
    parser.add_argument("--families", nargs="*", default=None)
    args = parser.parse_args()

    batch_id = run_suite(
        ScriptedPick(),
        args.policy_version,
        store=args.store,
        suite=args.suite,
        seeds_per_family=args.seeds_per_family,
        families=args.families,
    )
    runs = Store(args.store).read_batch(batch_id).runs
    by_family: dict[str, list[bool]] = {}
    for r in runs:
        by_family.setdefault(r.family, []).append(r.success)
    print(f"batch {batch_id}: {len(runs)} runs")
    for family, results in by_family.items():
        print(f"  {family:<12} {sum(results)}/{len(results)} succeeded")


if __name__ == "__main__":
    main()
