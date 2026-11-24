from __future__ import annotations

from pathlib import Path
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_gitattributes_enforces_lf_on_step_and_python():
    attrs = (REPO_ROOT / ".gitattributes").read_text(encoding="utf-8")
    for rule in (
        "*.py text eol=lf",
        "*.step text eol=lf",
        "*.xml text eol=lf",
        "*.urdf text eol=lf",
        "*.h text eol=lf",
        "*.ino text eol=lf",
    ):
        assert rule in attrs, f"Missing LF rule in .gitattributes: {rule}"

    for step_file in sorted((REPO_ROOT / "src/cad/purchased/step").glob("*.step")):
        assert b"\r\n" not in step_file.read_bytes(), f"CRLF found in {step_file.name}"


def test_default_checkpoint_slots_do_not_hold_untagged_legacy_weights():
    ckpt_dir = REPO_ROOT / "models/checkpoints"
    for name in (
        "cyberduck_v2_best.pt",
        "cyberduck_v2_latest.pt",
        "cyberduck_v2_ppo.pt",
    ):
        path = ckpt_dir / name
        if not path.exists():
            continue
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        assert isinstance(payload, dict), f"Unexpected payload type in {name}"
        assert payload.get("model_name") == "cyberduck_v2.xml", (
            f"Legacy/untagged checkpoint left in active slot {name}: "
            f"model_name={payload.get('model_name')!r}"
        )
