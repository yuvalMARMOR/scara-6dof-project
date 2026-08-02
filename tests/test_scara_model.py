"""Focused numerical checks for the corrected academic demonstration."""

import unittest

import numpy as np

from scara_model import (
    REFERENCE_CASES,
    forward_kinematics,
    plan_cartesian_trajectory,
)
class SCARAModelTests(unittest.TestCase):
    def test_five_corrected_reference_cases(self):
        for case in REFERENCE_CASES:
            with self.subTest(case=case.number):
                calculated = forward_kinematics(*case.joints)
                np.testing.assert_allclose(
                    calculated,
                    case.expected_position,
                    atol=1e-9,
                    rtol=0.0,
                )

    def test_demonstrated_trajectory_is_consistent_under_fk(self):
        trajectory = plan_cartesian_trajectory(
            np.array([-150.0, 150.0, 1050.0]),
            np.array([0.0, 400.0, 1200.0]),
        )
        self.assertLess(float(np.max(trajectory["position_error"])), 1e-6)
        np.testing.assert_allclose(
            trajectory["commanded_positions"][:, 0],
            [-150.0, 150.0, 1050.0],
            atol=1e-9,
        )
        np.testing.assert_allclose(
            trajectory["commanded_positions"][:, -1],
            [0.0, 400.0, 1200.0],
            atol=1e-9,
        )

    def test_zero_distance_trajectory_is_stationary(self):
        point = np.array([-150.0, 150.0, 1050.0])
        trajectory = plan_cartesian_trajectory(point, point)
        expected = np.repeat(point[:, None], trajectory["time"].size, axis=1)
        np.testing.assert_allclose(trajectory["commanded_positions"], expected, atol=1e-9)
        np.testing.assert_allclose(trajectory["achieved_positions"], expected, atol=1e-9)
        np.testing.assert_allclose(trajectory["velocities"], 0.0, atol=1e-12)
        np.testing.assert_allclose(trajectory["position_error"], 0.0, atol=1e-9)

if __name__ == "__main__":
    unittest.main()
