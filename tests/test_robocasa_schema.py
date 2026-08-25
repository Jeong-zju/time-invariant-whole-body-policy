import numpy as np

from robocasa_act_navigate.eval_server import base_only_to_native
from robocasa_lp_act.data import quaternion_xyzw_to_yaw


def test_xyzw_quaternion_yaw() -> None:
    quaternion = np.asarray([[0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]])
    np.testing.assert_allclose(quaternion_xyzw_to_yaw(quaternion), np.pi / 2.0, atol=1e-7)


def test_base_only_native_hold_contract() -> None:
    base = np.asarray([[[0.2, -0.1, 0.3]]], dtype=np.float32)
    native = base_only_to_native(base)
    np.testing.assert_allclose(native["action.base_motion"], [[[0.2, -0.1, 0.3, 0.0]]])
    np.testing.assert_allclose(native["action.control_mode"], 1.0)
    np.testing.assert_allclose(native["action.end_effector_position"], 0.0)
    np.testing.assert_allclose(native["action.end_effector_rotation"], 0.0)
    np.testing.assert_allclose(native["action.gripper_close"], -1.0)
