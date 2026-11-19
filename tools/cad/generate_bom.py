"""Report the V2 visual-part inventory and its selected actuator target."""

from pathlib import Path
import struct


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    print("# CyberDuck V2 — danh mục asset mô phỏng")
    print("Servo mục tiêu: 8 × MG996R; gá cơ khí và mass lắp ráp chưa được nghiệm thu.")
    print("Danh mục dưới đây là mesh giáp ngoài, không phải BOM chế tạo hoàn chỉnh.")
    print("\n| Asset | Tam giác |")
    print("| :--- | ---: |")
    for part in sorted((REPO_ROOT / "robot/meshes").glob("*.stl")):
        with part.open("rb") as source:
            source.seek(80)
            triangles = struct.unpack("<I", source.read(4))[0]
        print(f"| {part.name} | {triangles} |")


if __name__ == "__main__":
    main()
