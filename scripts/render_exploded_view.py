"""Render CyberDuck V2 views, including its exploded visual assembly."""

from pathlib import Path
import runpy


def render_exploded() -> None:
    renderer = Path(__file__).resolve().parents[1] / "duck_v2/scripts/render_cyberduck_v2.py"
    runpy.run_path(str(renderer), run_name="__main__")


if __name__ == "__main__":
    render_exploded()
