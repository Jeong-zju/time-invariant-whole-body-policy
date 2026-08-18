"""RoboCasa whole-body config with a matched 32-step output horizon."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig


ROBOCASA_LPWB_CONFIG = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["robot0_eye_in_hand", "robot0_agentview_left", "robot0_agentview_right"],
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "end_effector_position_relative",
            "end_effector_rotation_relative",
            "gripper_qpos",
            "base_position",
            "base_rotation",
        ],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(32)),
        modality_keys=[
            "end_effector_position",
            "end_effector_rotation",
            "gripper_close",
            "base_motion",
            "control_mode",
        ],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}

register_modality_config(ROBOCASA_LPWB_CONFIG, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
