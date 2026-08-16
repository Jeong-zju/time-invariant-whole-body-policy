"""Core primitives for the time-aware whole-body policy experiments."""

from .phase1 import (
    ACTION_DIM,
    BASE_HEIGHT_INDEX,
    BASE_SE2_SLICE,
    UPPER_BODY_DIM,
    MultiHorizonTargets,
    Phase1ExecutionAdapter,
    WholeBodyPlan,
    WholeBodyReference,
    build_multi_horizon_targets,
    compare_base_proxy_to_odometry,
    integrate_body_twist,
    load_target_batch,
    save_target_batch,
)
from .arena_m1 import (
    ArenaM1PlanExecutor,
    PlanActivationDiagnostics,
    decode_m1_policy_output_to_simulator_chunk,
    ordered_upper_sim_indices,
)
from .se2 import (
    compose,
    exp,
    interpolate,
    inverse,
    log,
    relative,
    wrap_angle,
)

__all__ = [
    "ACTION_DIM",
    "BASE_HEIGHT_INDEX",
    "BASE_SE2_SLICE",
    "UPPER_BODY_DIM",
    "MultiHorizonTargets",
    "Phase1ExecutionAdapter",
    "WholeBodyPlan",
    "WholeBodyReference",
    "ArenaM1PlanExecutor",
    "PlanActivationDiagnostics",
    "decode_m1_policy_output_to_simulator_chunk",
    "ordered_upper_sim_indices",
    "build_multi_horizon_targets",
    "compare_base_proxy_to_odometry",
    "compose",
    "exp",
    "integrate_body_twist",
    "interpolate",
    "inverse",
    "load_target_batch",
    "log",
    "relative",
    "save_target_batch",
    "wrap_angle",
]
