"""Scripted pick-and-place controller for Fetch.

Used for M1's recorded runs and as the fallback demo policy (BUILD_PLAN 9).
Deterministic: the same observation sequence always gives the same actions.
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class Phase(Enum):
    APPROACH = "approach"
    DESCEND = "descend"
    GRASP = "grasp"
    CARRY = "carry"


class ScriptedPick:
    """approach -> descend -> grasp -> carry to goal, re-approaching if the object drops.

    Actions are `[dx, dy, dz, gripper]` in [-1, 1]; the env scales dx..dz by 5 cm
    per step, and gripper +1 opens, -1 closes.
    """

    def __init__(
        self,
        *,
        gain: float = 10.0,
        hover: float = 0.05,
        grasp_height: float = 0.0,
        grasp_steps: int = 5,
        tolerance: float = 0.01,
    ) -> None:
        self.gain = gain
        self.hover = hover
        self.grasp_height = grasp_height
        """Grasp point above the object's centre; a weakened variant raises it."""
        self.grasp_steps = grasp_steps
        self.tolerance = tolerance
        self.reset()

    def reset(self) -> None:
        self.phase = Phase.APPROACH
        self._grasp_count = 0

    def act(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        o = obs["observation"]
        grip, obj, goal = o[0:3], o[3:6], obs["desired_goal"]
        holding_lost = np.linalg.norm(grip - obj) > 0.05
        if self.phase is Phase.CARRY and holding_lost:
            self.reset()

        gripper = 1.0
        if self.phase is Phase.APPROACH:
            target = obj + np.array([0.0, 0.0, self.hover])
            if np.linalg.norm(grip[:2] - obj[:2]) < self.tolerance:
                self.phase = Phase.DESCEND
        if self.phase is Phase.DESCEND:
            target = obj + np.array([0.0, 0.0, self.grasp_height])
            if np.linalg.norm(grip - target) < self.tolerance:
                self.phase = Phase.GRASP
        if self.phase is Phase.GRASP:
            target, gripper = grip, -1.0
            self._grasp_count += 1
            if self._grasp_count >= self.grasp_steps:
                self.phase = Phase.CARRY
        if self.phase is Phase.CARRY:
            target, gripper = goal, -1.0

        delta = np.clip(self.gain * (target - grip), -1.0, 1.0)
        return np.append(delta, gripper).astype(np.float32)
