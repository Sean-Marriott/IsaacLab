# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Seat a physical branch in the cutting mouth of the Mk3 chainsaw, carried by a DJI M350.

The alignment task of :mod:`.chainsaw_align_env_cfg`, with the target pose made physical: a rigid, fixed
cylinder lies across the approach at the target, and the mouth has to take it by feeding forward onto it. The
branch pushes back on the saw, so seating it loads the payload and, through the tube hinge, the airframe. The
policy observes the same terms as in the base task, with no contact sensing, as on the real vehicle: it has to
feel the branch through the hinge encoders and the airframe's response.

The mouth pocket is a forward-opening V between the saw bar and the feed-arm hook. The 10-50 mm branches used
here all reach about the same depth in it; thicker ones would seat further forward. Each branch is placed so
that, seated, it puts the mouth on the target pose; the
base task's success check (on the target, slow and square, having arrived along the approach axis) then means
seated. On every reset the diameter is drawn from :data:`BRANCH_SEATS`, the branch is turned to any heading with
the target turned to match, and it slopes across the approach by up to :data:`MAX_BRANCH_TILT` (vertical), ramped in
by the curriculum. The saw head is rolled to the slope at spawn and held there, as the vehicle's own saw-roll
controller would, and the airframe-rest observation and the spawn allow for how the roll moves the hanging mouth.
A steep branch ends :data:`MAX_BRANCH_RISE` above the cut, so it never reaches the rotors.

Contact on the saw above a force threshold is penalized, so the policy learns to seat the branch rather than
ram it, and any contact between the branch and the airframe, rotors, Jetson or tube ends the episode.
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObjectCollectionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import isaaclab_tasks.contrib.drone_arl.chainsaw_align.mdp as mdp

from .chainsaw_align_env_cfg import (
    COMMAND_NAME,
    ChainsawAlignSceneCfg,
    EventCfg,
    MatriceChainsawAlignEnvCfg,
    RewardsCfg,
    TerminationsCfg,
)

BRANCH_LENGTH = 3.0
"""Length of the branch [m]. Long enough that the drone has to lift the saw over it, rather than fly round its
ends, to reach the approach side from the far side."""

BRANCH_SEATS = {
    "branch_10mm": (0.025, -0.011, 0.01),
    "branch_20mm": (0.025, -0.011, 0.02),
    "branch_30mm": (0.025, -0.011, 0.03),
    "branch_40mm": (0.027, -0.013, 0.04),
    "branch_50mm": (0.025, -0.008, 0.05),
}
"""The branches, each with its seat ``(x, z, diameter)`` [m]: where its axis sits relative to the mouth when it is
seated, along the mouth frame's +x (forward) and +z (up) axes.

Measured in PhysX by flying a scripted controller onto each branch and correcting a first estimate by where the
mouth came to rest. The first estimate, from the Mk3 collision hulls (the ``roll_link`` saw and the
``feed_arm_link`` hook) projected onto the plane across the branch, put the thinner branches deeper in the
forward-opening V of the pocket: in simulation, 1-5 cm branches all stop ~2.5 cm ahead of the mouth, likely
because PhysX's simplified hulls fill the apex of the V. Thicker branches seat further forward (6-8 cm at 3.3-5.1 cm
ahead); the pocket tips are ~6 cm ahead of the mouth, so up to ~8 cm seat inside it. Re-measure if the USD's
colliders change.
"""

BRANCH_COLLECTION_NAME = "branch"
BRANCH_EVENT_NAME = "place_branch"

MAX_BRANCH_TILT = math.radians(90.0)
"""Largest slope of the branch across the approach, which the saw head is rolled to match [rad]: up to vertical,
the saw-roll joint's full range. The curriculum ramps it in from level branches."""

MAX_BRANCH_RISE = 0.8
"""Largest height of the branch's upper end above the cut [m]. A steep branch is slid down its axis to end there.

A branch that carried on up past 60-85 deg of slope would cut through the front-left prop disc: the airframe hangs
1.25 m above the mouth and 0.37 m behind it, with the front props reaching to within ~5 cm of the branch's plane.
Ending it 0.8 m above the cut keeps it at least 28 cm (axis) from every part of the vehicle at every slope, below
the landing gear, with the payload hanging at rest. Branches up to 32 deg keep their full length. Measured from the
Mk3 colliders at the simulated rest pose.
"""

SAW_ROLL_HANG_SHIFT = (-0.0038, 0.0382, 0.0315)
"""Shift of the hanging mouth with the saw roll [m]: the ``roll_hang_shift`` of
:func:`~isaaclab_tasks.contrib.drone_arl.chainsaw_align.mdp.rolled_hang_offset`. Rolling the head turns the mouth
about the roll axis, ~4 cm away, so at 90 deg it moves 3.8 cm sideways and 3.2 cm up. Fitted to the rest pose
measured in PhysX at 0-90 deg in 15 deg steps, to within ~1 mm."""

SAW_ROLL_JOINT_NAME = "roll_axis_center"
"""The Mk3's saw-roll joint, which rolls the saw head about the mouth's forward axis (within ~6 deg), positive with
the branch slope as :class:`~isaaclab_tasks.contrib.drone_arl.chainsaw_align.mdp.RandomBranch` defines it. It is
set by the vehicle's own controller, not the policy, so here it is held at the slope from the spawn on."""

SAW_BODY_NAMES = ["roll_link", "feed_arm_link"]
"""Bodies that may touch the branch: the saw and the feed-arm hook."""

AIRFRAME_BODY_NAMES = ["body", "rotor0", "rotor1", "rotor2", "rotor3", "jetson", "top_hinge", "main_tube"]
"""Bodies whose contact with the branch ends the episode: a rotor or airframe strike, or the tube fouling it."""

SPAWN_CLEARANCE = 0.55
"""Clearance of the spawned mouth from the branch, in the target frame [m]: the saw reaches ~0.45 m back from the
mouth and the tube ~0.4 m horizontally."""


def _branch_collection() -> RigidObjectCollectionCfg:
    """One kinematic cylinder per entry of :data:`BRANCH_SEATS`, lying along the target's y axis, parked until
    :class:`~isaaclab_tasks.contrib.drone_arl.chainsaw_align.mdp.RandomBranch` places one."""
    rigid_objects = {}
    for name, (_, _, diameter) in BRANCH_SEATS.items():
        rigid_objects[name] = RigidObjectCfg(
            prim_path=f"{{ENV_REGEX_NS}}/Branch/{name}",
            spawn=sim_utils.CylinderCfg(
                radius=0.5 * diameter,
                height=BRANCH_LENGTH,
                axis="Y",
                rigid_props=sim_utils.UsdPhysicsRigidBodyCfg(kinematic_enabled=True),
                collision_props=sim_utils.UsdPhysicsCollisionCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.36, 0.25, 0.15), roughness=0.9),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -50.0)),
        )
    return RigidObjectCollectionCfg(rigid_objects=rigid_objects)


##
# Scene definition
##


@configclass
class ChainsawAlignBranchSceneCfg(ChainsawAlignSceneCfg):
    """The base scene with the branches, and contact sensors on the saw and the rest of the vehicle."""

    branch: RigidObjectCollectionCfg = _branch_collection()

    # A policy step of history (decimation 10 at 100 Hz), so the reward and termination see every physics step.
    saw_contact = ContactSensorCfg(prim_path=f"{{ENV_REGEX_NS}}/Robot/({'|'.join(SAW_BODY_NAMES)})", history_length=10)
    airframe_contact = ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/({'|'.join(AIRFRAME_BODY_NAMES)})", history_length=10
    )


##
# MDP settings
##


_BASE_EVENTS = EventCfg()


@configclass
class BranchEventCfg:
    """The base events, with a branch placed at the target on every reset.

    The base terms are listed again rather than inherited, because reset terms run in the order they are declared
    and the branch has to come first: it turns the target, which the spawn reads.
    """

    # -- startup
    lock_saw_head = _BASE_EVENTS.lock_saw_head
    randomize_hinge_damping = _BASE_EVENTS.randomize_hinge_damping
    randomize_saw_mass = _BASE_EVENTS.randomize_saw_mass

    # -- reset
    # A branch of random diameter, at any heading, sloping by up to max_tilt, with the target turned to it and the
    # saw head rolled to the slope. max_tilt is the curriculum's starting value (see
    # MatriceChainsawAlignBranchEnvCfg).
    place_branch = EventTerm(
        func=mdp.RandomBranch,  # type: ignore[arg-type]
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(BRANCH_COLLECTION_NAME),
            "command_name": COMMAND_NAME,
            "seats": BRANCH_SEATS,
            "yaw_range": (-math.pi, math.pi),
            "max_tilt": 0.0,
            "roll_joint_name": SAW_ROLL_JOINT_NAME,
            "branch_length": BRANCH_LENGTH,
            "max_rise": MAX_BRANCH_RISE,
        },
    )
    reset_base = _BASE_EVENTS.reset_base
    reset_payload = _BASE_EVENTS.reset_payload

    # -- interval
    wind = _BASE_EVENTS.wind


@configclass
class BranchRewardsCfg(RewardsCfg):
    """The base rewards, with a penalty on hard contact between the saw and the branch."""

    # Seating the branch takes a light push. Over 15 N (about a sixth of the vehicle's weight) the excess costs
    # 0.02 per newton on the hardest step of the policy step: ramming it at the ~90 N the Lee controller can put
    # out at full tilt costs ~1.5 per step, about a third of the most a step can earn.
    saw_contact_force = RewTerm(
        func=mdp.contact_forces, weight=-0.02, params={"threshold": 15.0, "sensor_cfg": SceneEntityCfg("saw_contact")}
    )


@configclass
class BranchTerminationsCfg(TerminationsCfg):
    """The base terminations, ending the episode on contact between the branch and anything but the saw."""

    airframe_contact = DoneTerm(
        func=mdp.illegal_contact, params={"threshold": 1.0, "sensor_cfg": SceneEntityCfg("airframe_contact")}
    )


##
# Environment configuration
##


@configclass
class MatriceChainsawAlignBranchEnvCfg(MatriceChainsawAlignEnvCfg):
    """Seat a fixed branch in the Mk3 chainsaw's cutting mouth on the DJI M350, arriving along a set direction."""

    scene: ChainsawAlignBranchSceneCfg = ChainsawAlignBranchSceneCfg(num_envs=4096, env_spacing=8.0)
    events: BranchEventCfg = BranchEventCfg()
    rewards: BranchRewardsCfg = BranchRewardsCfg()
    terminations: BranchTerminationsCfg = BranchTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Keep the spawned payload clear of the branch: no mouth within SPAWN_CLEARANCE of it, along its length
        # and either side. The branch lies along the target's y axis at the target height.
        self.events.reset_base.params["keep_out_half_extents"] = (
            SPAWN_CLEARANCE,
            0.5 * BRANCH_LENGTH + SPAWN_CLEARANCE,
            math.inf,
        )
        # The hanging mouth moves as the saw head rolls: the airframe-rest observation and the spawn follow the roll.
        roll_params = {
            "roll_cfg": SceneEntityCfg("robot", joint_names=[SAW_ROLL_JOINT_NAME]),
            "roll_hang_shift": SAW_ROLL_HANG_SHIFT,
        }
        self.observations.policy.airframe_rest_error.params.update(roll_params)
        self.events.reset_base.params.update(roll_params)
        # Slope the branch in with the spawn widening, over the first half of the difficulty level. play_mode puts
        # it at its end value with the other ramps.
        self.curriculum.difficulty.params["ramps"][f"{BRANCH_EVENT_NAME}.max_tilt"] = (0.0, MAX_BRANCH_TILT, 0.0, 0.5)
