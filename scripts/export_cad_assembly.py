"""Compatibility entry point for exporting CyberDuck V2 visual CAD."""

from pathlib import Path
import runpy


V2_EXPORTER = Path(__file__).resolve().parents[1] / "duck_v2" / "scripts" / "export_cyberduck_v2_cad.py"


def export_assembly():
    namespace = runpy.run_path(str(V2_EXPORTER))
    return namespace["export_v2_cad"]()


if __name__ == "__main__":
    export_assembly()
