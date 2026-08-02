"""Shared analytical model for the academic 6-DOF SCARA demonstration.

The canonical joint order is::

    [theta1, l1, l2, theta2, theta3, theta4]

Angles are supplied in degrees and distances in millimetres.  The model follows
the final-report position equations, with the corrected convention that ``l2``
translates along local +Y.

The inverse solver in this module is intentionally a *position* solver, not a
general six-degree-of-freedom pose solver.  The caller supplies a wrist posture
(``theta2`` and ``theta3``); the solver calculates ``theta1``, ``l1`` and ``l2``.
``theta4`` is free for a position task and is retained only for illustration.
"""

from dataclasses import dataclass

import numpy as np


BASE_HEIGHT_MM = 800.0
WRIST_OFFSET_MM = 150.0

JOINT_LIMITS = {
    "theta1": (-180.0, 180.0),
    "l1": (0.0, 500.0),
    "l2": (0.0, 500.0),
    "theta2": (-180.0, 180.0),
    "theta3": (-135.0, 135.0),
    "theta4": (-180.0, 180.0),
}


@dataclass(frozen=True)
class ReferenceCase:
    """A corrected forward-kinematics reference case."""

    number: int
    joints: tuple[float, float, float, float, float, float]
    expected_position: tuple[float, float, float]


REFERENCE_CASES = (
    ReferenceCase(1, (0.0, 250.0, 0.0, 0.0, 90.0, 0.0), (-150.0, 150.0, 1050.0)),
    ReferenceCase(2, (0.0, 250.0, 250.0, 90.0, 90.0, 0.0), (0.0, 400.0, 1200.0)),
    ReferenceCase(3, (180.0, 500.0, 500.0, 0.0, 0.0, 0.0), (0.0, -800.0, 1300.0)),
    ReferenceCase(4, (180.0, 500.0, 500.0, 90.0, 90.0, 180.0), (0.0, -650.0, 1450.0)),
    ReferenceCase(5, (180.0, 0.0, 0.0, -90.0, 0.0, 180.0), (0.0, -300.0, 800.0)),
)


def forward_kinematics(theta1, l1, l2, theta2, theta3, theta4):
    """Return the analytical end-effector position ``[x, y, z]`` in mm.

    ``theta4`` does not appear in the position equation because it is the final
    tool rotation.  It is accepted to preserve the canonical six-joint order.
    """

    del theta4  # Position is invariant to the final tool rotation.
    th1, th2, th3 = np.deg2rad([theta1, theta2, theta3])
    c1, s1 = np.cos(th1), np.sin(th1)
    c2, s2 = np.cos(th2), np.sin(th2)
    c3, s3 = np.cos(th3), np.sin(th3)

    x = (
        -WRIST_OFFSET_MM * s1
        - WRIST_OFFSET_MM * c3 * s1
        - l2 * s1
        - WRIST_OFFSET_MM * c1 * c2 * s3
    )
    y = (
        WRIST_OFFSET_MM * c1
        + WRIST_OFFSET_MM * c1 * c3
        + l2 * c1
        - WRIST_OFFSET_MM * s1 * c2 * s3
    )
    z = BASE_HEIGHT_MM + l1 + WRIST_OFFSET_MM * s2 * s3
    return np.array([x, y, z], dtype=float)


def _rotation_y(angle_radians):
    cosine, sine = np.cos(angle_radians), np.sin(angle_radians)
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=float,
    )


def _rotation_z(angle_radians):
    cosine, sine = np.cos(angle_radians), np.sin(angle_radians)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )


def forward_orientation(theta1, l1, l2, theta2, theta3, theta4):
    """Return the analytical ``tool0`` orientation as a 3x3 rotation matrix.

    The prismatic coordinates are accepted to preserve canonical joint ordering
    but do not affect orientation.
    """

    del l1, l2
    th1, th2, th3, th4 = np.deg2rad([theta1, theta2, theta3, theta4])
    return _rotation_z(th1) @ _rotation_y(th2) @ _rotation_z(th3) @ _rotation_y(th4)


def forward_transform(theta1, l1, l2, theta2, theta3, theta4):
    """Return the full analytical world-to-``tool0`` homogeneous transform.

    Translation is expressed in millimetres to match :func:`forward_kinematics`.
    """

    joints = (theta1, l1, l2, theta2, theta3, theta4)
    transform = np.eye(4)
    transform[:3, :3] = forward_orientation(*joints)
    transform[:3, 3] = forward_kinematics(*joints)
    return transform


def _validate_joint_limits(joints, tolerance=1e-9):
    names = tuple(JOINT_LIMITS)
    for name, value in zip(names, joints):
        lower, upper = JOINT_LIMITS[name]
        if value < lower - tolerance or value > upper + tolerance:
            raise ValueError(
                f"Unreachable posture: {name}={value:.6g} is outside "
                f"[{lower:.6g}, {upper:.6g}]."
            )


def inverse_position(
    target_position,
    theta2=0.0,
    theta3=90.0,
    theta4=0.0,
    tolerance=1e-9,
):
    """Solve position IK for a caller-selected wrist posture.

    The existing demonstration uses ``theta2=0`` and ``theta3=90``.  That
    posture reaches the complete example line while keeping the position solve
    analytical and exact.  No requested orientation matrix is solved here.

    Raises:
        ValueError: if the posture cannot reach the target within joint limits.
    """

    x, y, z = np.asarray(target_position, dtype=float)
    th2, th3 = np.deg2rad([theta2, theta3])
    rho_squared = float(x * x + y * y)
    b_term = WRIST_OFFSET_MM * np.cos(th2) * np.sin(th3)
    radicand = rho_squared - b_term * b_term

    if radicand < -tolerance:
        raise ValueError(
            "Unreachable position for the selected wrist posture: "
            "radial distance is smaller than the wrist contribution."
        )
    a_term = float(np.sqrt(max(0.0, radicand)))
    l2 = a_term - WRIST_OFFSET_MM * (1.0 + np.cos(th3))
    l1 = z - BASE_HEIGHT_MM - WRIST_OFFSET_MM * np.sin(th2) * np.sin(th3)

    if rho_squared <= tolerance:
        raise ValueError("The world Z-axis is unreachable for the selected posture.")

    sin_theta1 = (-a_term * x - b_term * y) / rho_squared
    cos_theta1 = (a_term * y - b_term * x) / rho_squared
    theta1 = float(np.rad2deg(np.arctan2(sin_theta1, cos_theta1)))

    # Snap only floating-point noise at an exact boundary; do not clamp an
    # actually unreachable target.
    if abs(l1) <= tolerance:
        l1 = 0.0
    if abs(l1 - 500.0) <= tolerance:
        l1 = 500.0
    if abs(l2) <= tolerance:
        l2 = 0.0
    if abs(l2 - 500.0) <= tolerance:
        l2 = 500.0

    joints = np.array([theta1, l1, l2, theta2, theta3, theta4], dtype=float)
    _validate_joint_limits(joints, tolerance=tolerance)

    achieved = forward_kinematics(*joints)
    residual = float(np.linalg.norm(achieved - np.asarray(target_position, dtype=float)))
    if residual > 1e-6:
        raise RuntimeError(f"Position IK residual is unexpectedly large: {residual:.6g} mm")
    return joints


def plan_cartesian_trajectory(
    start_position,
    end_position,
    motion_time=30.0,
    acceleration_time=10.0,
    deceleration_start_time=20.0,
    time_step=0.04,
    theta2=0.0,
    theta3=90.0,
    theta4_start=0.0,
    theta4_end=180.0,
):
    """Plan a straight Cartesian path with a trapezoidal speed profile.

    The returned dictionary contains both commanded positions and positions
    achieved by applying forward kinematics to every generated joint vector.
    A zero-distance request is represented by a stationary trajectory.
    """

    if not (0.0 < acceleration_time <= deceleration_start_time < motion_time):
        raise ValueError("Require 0 < acceleration_time <= deceleration_start_time < motion_time.")
    if time_step <= 0.0:
        raise ValueError("time_step must be positive.")

    start = np.asarray(start_position, dtype=float)
    end = np.asarray(end_position, dtype=float)
    if start.shape != (3,) or end.shape != (3,):
        raise ValueError("start_position and end_position must each contain three values.")

    time = np.arange(0.0, motion_time + 0.5 * time_step, time_step)
    total_distance = float(np.linalg.norm(end - start))
    progress = np.zeros_like(time)
    speed = np.zeros_like(time)

    if total_distance > 1e-12:
        deceleration_time = motion_time - deceleration_start_time
        profile_area = (
            0.5 * acceleration_time
            + (deceleration_start_time - acceleration_time)
            + 0.5 * deceleration_time
        )
        maximum_speed = total_distance / profile_area
        acceleration = maximum_speed / acceleration_time
        deceleration = maximum_speed / deceleration_time

        for index, current_time in enumerate(time):
            if current_time <= acceleration_time:
                travelled = 0.5 * acceleration * current_time**2
                current_speed = acceleration * current_time
            elif current_time <= deceleration_start_time:
                acceleration_distance = 0.5 * maximum_speed * acceleration_time
                travelled = acceleration_distance + maximum_speed * (
                    current_time - acceleration_time
                )
                current_speed = maximum_speed
            else:
                time_in_deceleration = current_time - deceleration_start_time
                acceleration_distance = 0.5 * maximum_speed * acceleration_time
                constant_distance = maximum_speed * (
                    deceleration_start_time - acceleration_time
                )
                travelled = (
                    acceleration_distance
                    + constant_distance
                    + maximum_speed * time_in_deceleration
                    - 0.5 * deceleration * time_in_deceleration**2
                )
                current_speed = maximum_speed - deceleration * time_in_deceleration

            progress[index] = np.clip(travelled / total_distance, 0.0, 1.0)
            speed[index] = max(0.0, current_speed)

        progress[-1] = 1.0
        speed[-1] = 0.0

    commanded = start[:, None] + (end - start)[:, None] * progress[None, :]
    theta4_values = theta4_start + (theta4_end - theta4_start) * progress
    joint_configurations = np.array(
        [
            inverse_position(
                commanded[:, index],
                theta2=theta2,
                theta3=theta3,
                theta4=theta4_values[index],
            )
            for index in range(time.size)
        ]
    )
    achieved = np.array(
        [forward_kinematics(*joints) for joints in joint_configurations]
    ).T
    position_error = np.linalg.norm(achieved - commanded, axis=0)
    wrist_joints = np.vstack(
        [
            np.full_like(time, theta2),
            np.full_like(time, theta3),
            theta4_values,
        ]
    )

    return {
        "time": time,
        "progress": progress,
        "commanded_positions": commanded,
        "achieved_positions": achieved,
        "position_error": position_error,
        "joint_configs": joint_configurations,
        "velocities": speed,
        "wrist_joints": wrist_joints,
        # Compatibility aliases for the original academic plotting scripts.
        "positions": commanded,
        "orientations": wrist_joints,
        "start_pos": start,
        "end_pos": end,
    }


def sample_workspace(
    theta1_samples=25,
    l1_samples=5,
    l2_samples=5,
    theta2_samples=7,
    theta3_samples=7,
):
    """Sample the declared joint ranges and return reachable XYZ points."""

    axes = (
        np.linspace(*JOINT_LIMITS["theta1"], theta1_samples),
        np.linspace(*JOINT_LIMITS["l1"], l1_samples),
        np.linspace(*JOINT_LIMITS["l2"], l2_samples),
        np.linspace(*JOINT_LIMITS["theta2"], theta2_samples),
        np.linspace(*JOINT_LIMITS["theta3"], theta3_samples),
    )
    grid = np.meshgrid(*axes, indexing="ij")
    values = [item.ravel() for item in grid]
    th1, l1, l2, th2, th3 = values
    th1_rad, th2_rad, th3_rad = np.deg2rad([th1, th2, th3])
    c1, s1 = np.cos(th1_rad), np.sin(th1_rad)
    c2, s2 = np.cos(th2_rad), np.sin(th2_rad)
    c3, s3 = np.cos(th3_rad), np.sin(th3_rad)

    x = -WRIST_OFFSET_MM * s1 - WRIST_OFFSET_MM * c3 * s1 - l2 * s1 - WRIST_OFFSET_MM * c1 * c2 * s3
    y = WRIST_OFFSET_MM * c1 + WRIST_OFFSET_MM * c1 * c3 + l2 * c1 - WRIST_OFFSET_MM * s1 * c2 * s3
    z = BASE_HEIGHT_MM + l1 + WRIST_OFFSET_MM * s2 * s3
    return np.column_stack([x, y, z])


def static_payload_reactions(joint_configurations, mass_kg=1.0):
    """Return the report's simplified static reactions for a payload."""

    joints = np.asarray(joint_configurations, dtype=float)
    theta2 = np.deg2rad(joints[:, 3])
    theta3 = np.deg2rad(joints[:, 4])
    gravity = 9.81
    tool_length_m = WRIST_OFFSET_MM / 1000.0

    reactions = np.zeros((joints.shape[0], 6), dtype=float)
    reactions[:, 1] = mass_kg * gravity
    reactions[:, 3] = mass_kg * gravity * tool_length_m * np.cos(theta2) * np.sin(theta3)
    reactions[:, 4] = mass_kg * gravity * tool_length_m * np.sin(theta2) * np.cos(theta3)
    return reactions
