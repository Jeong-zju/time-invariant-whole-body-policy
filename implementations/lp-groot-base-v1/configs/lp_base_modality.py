from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig


register_modality_config(
    {
        "video": ModalityConfig(delta_indices=[0], modality_keys=["res256_image_side_0", "res256_image_side_1", "res256_image_wrist_0"]),
        "state": ModalityConfig(delta_indices=[0], modality_keys=["base_position", "base_rotation"]),
        "action": ModalityConfig(
            delta_indices=list(range(32)), modality_keys=["lp_base_path"],
            action_configs=[ActionConfig(rep=ActionRepresentation.ABSOLUTE, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)],
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
    },
    EmbodimentTag.NEW_EMBODIMENT,
)
