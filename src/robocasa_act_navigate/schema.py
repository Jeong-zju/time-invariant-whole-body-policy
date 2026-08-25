"""Verified RoboCasa365 NavigateKitchen schema."""

FPS = 20
CHUNK_SIZE = 32
EXECUTION_STEPS = 8
ROBOT_STATE_DIM = 16
NUM_TASKS = 13
STATE_DIM = ROBOT_STATE_DIM + NUM_TASKS
ACTION_DIM = 12
IMAGE_SIZE = (256, 256)
CAMERA_KEYS = (
    "observation.images.robot0_eye_in_hand",
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
)
