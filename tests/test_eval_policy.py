import numpy as np
import pytest
import torch

from lp_act.eval_policy import build_act_observation_batch, build_lp_execution_origin
from lp_act.stats import FeatureStats


def synthetic_observation():
    return {
        "robot_r1::proprio": np.arange(61, dtype=np.float32),
        "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": np.zeros((224, 224, 4), dtype=np.uint8),
        "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": np.full(
            (224, 224, 3), 255, dtype=np.uint8
        ),
        "robot_r1::robot_r1:zed_link:Camera:0::rgb": np.full((3, 112, 112), 0.5, dtype=np.float32),
        "robot_r1::cam_rel_poses": np.zeros(21, dtype=np.float32),
        "task_id": np.asarray([0], dtype=np.int64),
    }


def test_build_act_observation_batch_matches_training_schema():
    stats = FeatureStats(mean=np.ones(61, dtype=np.float32), std=np.full(61, 2.0, dtype=np.float32))
    batch = build_act_observation_batch(synthetic_observation(), stats, device=torch.device("cpu"))

    assert batch["observation.state"].shape == (1, 61)
    np.testing.assert_allclose(batch["observation.state"][0].numpy(), (np.arange(61) - 1) / 2)
    for key in (
        "observation.rgb.left_realsense_link_camera_0",
        "observation.rgb.right_realsense_link_camera_0",
        "observation.rgb.zed_link_camera_0",
    ):
        assert batch[key].shape == (1, 3, 224, 224)
        assert torch.isfinite(batch[key]).all()


def test_build_act_observation_batch_rejects_bad_proprio_shape():
    obs = synthetic_observation()
    obs["robot_r1::proprio"] = np.zeros(60, dtype=np.float32)
    stats = FeatureStats(mean=np.zeros(61), std=np.ones(61))
    with pytest.raises(ValueError, match="61 proprio"):
        build_act_observation_batch(obs, stats, device=torch.device("cpu"))


def test_build_lp_execution_origin_uses_measured_joints_and_predicted_grippers():
    state = np.arange(61, dtype=np.float32)
    chunk = np.zeros((32, 24), dtype=np.float32)
    chunk[0, 14] = -0.75
    chunk[0, 22] = 0.5
    origin = build_lp_execution_origin(state, chunk)
    expected = np.concatenate([state[53:57], state[3:10], [-0.75], state[28:35], [0.5]])
    np.testing.assert_array_equal(origin, expected)
