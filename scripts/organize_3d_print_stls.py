"""Package the V2 visual STLs without mixing in retired robot parts."""

from pathlib import Path
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = REPO_ROOT / "duck_v2/meshes"
TARGET_DIR = REPO_ROOT / "duck_v2/cad"


def organize() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    output = TARGET_DIR / "cyberduck_v2_stl_parts.zip"
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for part in sorted(SOURCE_DIR.glob("*.stl")):
            archive.write(part, arcname=part.name)
    print(f"V2 visual STL package: {output}")
    print("Các mesh này chưa chứng nhận gá MG996R hoặc dung sai in/lắp ráp.")


if __name__ == "__main__":
    organize()
