"""Portable policy identity and raw observation contracts (no Torch dependency)."""
import math
from pathlib import Path

from robot.robot_spec import CHECKPOINT_PREFIX, MODEL_NAME


RUNTIME_TARGETS = ("rpi5", "pizero2w", "esp32", "v2")
TARGET_MODELS = {target: MODEL_NAME for target in RUNTIME_TARGETS}
CHECKPOINT_PREFIXES = {target: CHECKPOINT_PREFIX for target in RUNTIME_TARGETS}


def resolve_target_checkpoints(root, targets, explicit_checkpoint=None):
    resolved = {}
    for target in targets:
        if target not in CHECKPOINT_PREFIXES:
            raise ValueError(f"Unknown runtime target: {target}")
        prefix = CHECKPOINT_PREFIXES[target]
        candidates = [Path(root) / "models/checkpoints" / f"{prefix}_{kind}.pt"
                      for kind in ("best", "ppo", "latest")]
        chosen = Path(explicit_checkpoint) if explicit_checkpoint else next((p for p in candidates if p.is_file()), None)
        if chosen is None or not chosen.is_file():
            raise FileNotFoundError(f"Checkpoint for {target} not found: {chosen or candidates[0]}")
        resolved[target] = chosen
    return resolved


def validate_model_identity(checkpoint, model_path):
    expected = Path(model_path).name
    actual = checkpoint.get("model_name") or Path(checkpoint.get("model_path", "")).name
    if not actual or actual != expected:
        raise ValueError(f"Checkpoint model identity {actual!r} does not match {expected!r}; use explicit transfer to new output paths")
    if checkpoint.get("model_path") and Path(checkpoint["model_path"]).name != actual:
        raise ValueError("Checkpoint model identity metadata conflicts")


def observation_schema(obs_dim, embed_phase_clock, control_dt, period=0.8):
    return validate_observation_schema(dict(version=1, obs_dim=int(obs_dim),
        embed_phase_clock=bool(embed_phase_clock), period=float(period),
        control_dt=float(control_dt), reset="episode_steps_zero", advance="after_action",
        zero_command="advance"))


def validate_observation_schema(schema):
    if not isinstance(schema, dict):
        raise ValueError("observation_schema must be a mapping")
    if (type(schema.get("version")) is not int or schema.get("version") != 1
            or type(schema.get("obs_dim")) is not int or schema.get("obs_dim") not in (37, 42)
            or type(schema.get("embed_phase_clock")) is not bool
            or schema.get("reset") != "episode_steps_zero"
            or schema.get("advance") != "after_action" or schema.get("zero_command") != "advance"):
        raise ValueError("Unsupported observation_schema layout or phase clock rules")
    for key in ("period", "control_dt"):
        value = schema.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"observation_schema {key} must be finite and positive")
    return dict(schema)


def phase_clock_for_schema(schema, episode_steps):
    """Return [sin, cos] gait phase clock per validated observation_schema, or None if disabled."""
    import numpy as np
    from sim.env.env_config import compute_gait_phase

    validated = validate_observation_schema(schema)
    if not validated["embed_phase_clock"]:
        return None
    _, sine, cosine = compute_gait_phase(
        episode_steps, dt_control=validated["control_dt"], period=validated["period"]
    )
    return np.array([float(sine), float(cosine)], dtype=np.float32)
