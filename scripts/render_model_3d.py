"""Compatibility entry point for rendering the CyberDuck V2 model."""

from pathlib import Path
import runpy


V2_RENDERER = Path(__file__).resolve().parents[1] / "duck_v2" / "scripts" / "render_cyberduck_v2.py"


def render_sculpted_duck() -> None:
    runpy.run_path(str(V2_RENDERER), run_name="__main__")


if __name__ == "__main__":
    render_sculpted_duck()
