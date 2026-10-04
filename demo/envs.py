"""Fetch pick-and-place wrapper: scenario families and polcheck signal extraction.

Scenario config keys (interpreted only here):

- `object_x`, `object_y`: `[low, high]` ranges for the object's start position,
  as offsets in metres from the gripper's initial position. The robot faces +x,
  so negative x is near the robot and positive y is to its left.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from typing import Any

import gymnasium as gym
import gymnasium_robotics
import mujoco
import numpy as np

from polcheck.recorder import Contact, RunContext
from polcheck.schema import Scenario, SimulatorInfo

ENV_ID = "FetchPickAndPlace-v4"
MAX_STEPS = 100  # the env default of 50 (2 s) is too tight for the scripted controller

ARM_JOINTS = [
    "robot0:shoulder_pan_joint",
    "robot0:shoulder_lift_joint",
    "robot0:upperarm_roll_joint",
    "robot0:elbow_flex_joint",
    "robot0:forearm_roll_joint",
    "robot0:wrist_flex_joint",
    "robot0:wrist_roll_joint",
]
ARM_ROOT_BODY = "robot0:shoulder_pan_link"  # the arm is this body and its descendants
FINGER_BODIES = ("robot0:r_gripper_finger_link", "robot0:l_gripper_finger_link")
OBJECT_BODY = "object0"
TABLE_BODY = "table0"

gym.register_envs(gymnasium_robotics)


def make_env(max_steps: int = MAX_STEPS) -> gym.Env[Any, Any]:
    return gym.make(ENV_ID, max_episode_steps=max_steps)


def simulator_info() -> SimulatorInfo:
    return SimulatorInfo(name="mujoco", version=mujoco.__version__)


def physics_hash(env: gym.Env[Any, Any]) -> str:
    """Hash of the settings that change physics results for the same actions."""
    sim = env.unwrapped
    opt = sim.model.opt  # type: ignore[attr-defined]
    settings = {
        "env": ENV_ID,
        "n_substeps": int(sim.n_substeps),  # type: ignore[attr-defined]
        "timestep": float(opt.timestep),
        "gravity": [float(g) for g in opt.gravity],
        "integrator": int(opt.integrator),
        "solver": int(opt.solver),
        "cone": int(opt.cone),
        "jacobian": int(opt.jacobian),
        "iterations": int(opt.iterations),
        "tolerance": float(opt.tolerance),
        "noslip_iterations": int(opt.noslip_iterations),
        "impratio": float(opt.impratio),
        "density": float(opt.density),
        "viscosity": float(opt.viscosity),
    }
    payload = json.dumps(settings, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def reset_to_scenario(
    env: gym.Env[Any, Any], scenario: Scenario, env_seed: int
) -> dict[str, np.ndarray]:
    """Reset with `env_seed` (which also fixes the goal), then place the object
    in the scenario's start region using a generator seeded by `env_seed`."""
    env.reset(seed=env_seed)
    sim = env.unwrapped
    model, data = sim.model, sim.data  # type: ignore[attr-defined]
    rng = np.random.default_rng(env_seed)
    offset = np.array(
        [rng.uniform(*_range(scenario, "object_x")), rng.uniform(*_range(scenario, "object_y"))]
    )
    qpos = sim._utils.get_joint_qpos(model, data, "object0:joint").copy()  # type: ignore[attr-defined]
    qpos[:2] = sim.initial_gripper_xpos[:2] + offset  # type: ignore[attr-defined]
    sim._utils.set_joint_qpos(model, data, "object0:joint", qpos)  # type: ignore[attr-defined]
    mujoco.mj_forward(model, data)
    obs: dict[str, np.ndarray] = sim._get_obs()  # type: ignore[attr-defined]
    return obs


class SignalLogger:
    """Extracts section 6.4 signals and contacts from the env's MuJoCo state."""

    def __init__(self, env: gym.Env[Any, Any]) -> None:
        sim = env.unwrapped
        self.model: mujoco.MjModel = sim.model  # type: ignore[attr-defined]
        self.data: mujoco.MjData = sim.data  # type: ignore[attr-defined]
        m = self.model
        joint_ids = [m.joint(name).id for name in ARM_JOINTS]
        self._qpos_adr = np.array([m.jnt_qposadr[j] for j in joint_ids])
        self._dof_adr = np.array([m.jnt_dofadr[j] for j in joint_ids])
        self._grip_site = m.site("robot0:grip").id
        self._object_body = m.body(OBJECT_BODY).id
        body_names = [m.body(m.geom_bodyid[g]).name for g in range(m.ngeom)]
        self._geom_names = [m.geom(g).name or body_names[g] for g in range(m.ngeom)]
        # Only the arm counts as "robot": the base stands on the floor right next
        # to the table, which would pin clearance and flood the contact log.
        arm_root = m.body(ARM_ROOT_BODY).id
        self._arm_geoms = [
            g for g in range(m.ngeom) if _descends_from(m, m.geom_bodyid[g], arm_root)
        ]
        self._arm_set = set(self._arm_geoms)
        self._arm_vertices = [_local_vertices(m, g) for g in self._arm_geoms]
        self._finger_geoms = {g for g in range(m.ngeom) if body_names[g] in FINGER_BODIES}
        self._object_geoms = {g for g in range(m.ngeom) if body_names[g] == OBJECT_BODY}
        (self._table_geom,) = [g for g in range(m.ngeom) if body_names[g] == TABLE_BODY]

    def log(self, run: RunContext) -> None:
        d = self.data
        ee_quat = np.zeros(4)
        mujoco.mju_mat2Quat(ee_quat, d.site_xmat[self._grip_site])
        run.log(
            t=float(d.time),
            joint_pos=d.qpos[self._qpos_adr],
            joint_vel=d.qvel[self._dof_adr],
            ee_pos=d.site_xpos[self._grip_site],
            ee_quat=ee_quat,
            objects={"box": (d.xpos[self._object_body], d.xquat[self._object_body])},
            clearance=self.clearance(),
            qpos=d.qpos,
        )
        run.log_contacts(t=float(d.time), contacts=list(self.contacts()))

    def clearance(self) -> float:
        """Minimum distance from the arm to the table.

        `mj_geomDistance` returns spurious zeros for box-box pairs in MuJoCo 3.11,
        so this is computed directly: the table is an axis-aligned box, and the
        point of a convex geom closest to it is a vertex in all but edge-on-edge
        cases near the table's rim. Penetration reports 0.
        """
        d = self.data
        g = self._table_geom
        if not np.allclose(d.geom_xmat[g], np.eye(3).ravel()):
            raise RuntimeError("table is not axis-aligned; clearance needs updating")
        lo = d.geom_xpos[g] - self.model.geom_size[g]
        hi = d.geom_xpos[g] + self.model.geom_size[g]
        best = np.inf
        for geom, local in zip(self._arm_geoms, self._arm_vertices, strict=True):
            world = d.geom_xpos[geom] + local @ d.geom_xmat[geom].reshape(3, 3).T
            gap = np.maximum(np.maximum(lo - world, world - hi), 0.0)
            best = min(best, float(np.sqrt((gap**2).sum(axis=1)).min()))
        return best

    def contacts(self) -> Iterator[Contact]:
        """Contacts involving the arm. Finger-object contacts are intended."""
        force = np.zeros(6)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 not in self._arm_set and g2 not in self._arm_set:
                continue
            mujoco.mj_contactForce(self.model, self.data, i, force)
            intended = (g1 in self._finger_geoms and g2 in self._object_geoms) or (
                g2 in self._finger_geoms and g1 in self._object_geoms
            )
            yield Contact(
                self._geom_names[g1],
                self._geom_names[g2],
                float(np.linalg.norm(force[:3])),
                intended,
            )


def _range(scenario: Scenario, key: str) -> tuple[float, float]:
    value = scenario.config.get(key)
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(v, int | float) for v in value)
    ):
        raise ValueError(f"scenario {scenario.family!r}: {key} must be [low, high], got {value!r}")
    low, high = float(value[0]), float(value[1])  # type: ignore[arg-type]
    return low, high


def _descends_from(model: mujoco.MjModel, body: int, ancestor: int) -> bool:
    while body != 0:
        if body == ancestor:
            return True
        body = model.body_parentid[body]
    return False


def _local_vertices(model: mujoco.MjModel, geom: int) -> np.ndarray:
    """A box or mesh geom's vertices in its own frame."""
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_BOX:
        corners = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
        return np.asarray(corners * model.geom_size[geom], dtype=np.float64)
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        return np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)
    raise NotImplementedError(f"clearance for geom type {model.geom_type[geom]}")
