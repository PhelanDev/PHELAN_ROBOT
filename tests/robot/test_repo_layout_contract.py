"""Contract tests enforcing the src/ + clean-root CyberDuck V2 repository layout."""

import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from robot.robot_spec import MODEL_PATH, REPO_ROOT, ROBOT_ONLY_MJCF_PATH, SRC_ROOT, URDF_PATH

FORBIDDEN_TOP_LEVEL_DIRS = (
    "duck_v2",
    "training",
    "pipelines",
    "scripts",
    "robot",
    "cad",
    "sim",
    "hardware",
    "tools",
    "archive",
    "plans",
)
FORBIDDEN_TOP_LEVEL_FILES = ("viewer_3d.html", "his.md", "tailieubandau.txt")
SRC_PILLAR_DIRS = ("robot", "cad", "sim", "hardware", "tools")


def test_clean_root_and_src_pillars_exist():
    for name in FORBIDDEN_TOP_LEVEL_DIRS:
        assert not (REPO_ROOT / name).exists(), f"Top-level directory should not exist at repo root: {name}/"
    for name in FORBIDDEN_TOP_LEVEL_FILES:
        assert not (REPO_ROOT / name).exists(), f"Stray file should not exist at repo root: {name}"

    assert SRC_ROOT == REPO_ROOT / "src"
    for pillar in SRC_PILLAR_DIRS:
        assert (SRC_ROOT / pillar).is_dir(), f"Missing src pillar: src/{pillar}/"

    assert (REPO_ROOT / "docs" / "his.md").is_file()
    assert (REPO_ROOT / "docs" / "tailieubandau.txt").is_file()
    assert (REPO_ROOT / "docs" / "plans").is_dir()
    assert (REPO_ROOT / "docs" / "archive" / "v1").is_dir()
    assert not (REPO_ROOT / "docs" / "superpowers").exists()


def test_no_legacy_or_prefixed_src_imports():
    py_files = (
        [REPO_ROOT / "duck.py"]
        + [p for pillar in SRC_PILLAR_DIRS for p in (SRC_ROOT / pillar).rglob("*.py") if "__pycache__" not in p.parts]
        + [p for p in (REPO_ROOT / "tests").rglob("*.py") if "__pycache__" not in p.parts]
    )
    bad_import_re = re.compile(r"^\s*(?:from|import)\s+(?:duck_v2|training|pipelines|scripts|src)\b", re.MULTILINE)
    offenders = []
    for path in py_files:
        text = path.read_text(encoding="utf-8")
        for match in bad_import_re.finditer(text):
            offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()}: {match.group(0).strip()}")
    assert not offenders, "Legacy or 'src.'-prefixed imports remain:\n" + "\n".join(offenders)


def test_tests_root_has_no_stray_test_files():
    stray = sorted(p.name for p in (REPO_ROOT / "tests").glob("test_*.py"))
    assert not stray, f"Stray flat test files at tests/ root: {stray}"
    for domain in ("cad", "robot", "hardware", "sim"):
        domain_dir = REPO_ROOT / "tests" / domain
        assert domain_dir.is_dir(), f"Missing test domain directory: tests/{domain}/"
        assert list(domain_dir.glob("test_*.py")), f"Empty test domain directory: tests/{domain}/"


def test_robot_xml_urdf_meshes_colocated():
    meshes_dir = SRC_ROOT / "robot" / "meshes"
    assert meshes_dir.is_dir()
    for xml_path in (MODEL_PATH, ROBOT_ONLY_MJCF_PATH):
        assert xml_path.parent == SRC_ROOT / "robot"
        root = ET.parse(xml_path).getroot()
        for mesh in root.iter("mesh"):
            mesh_file = mesh.attrib.get("file")
            if mesh_file:
                assert (meshes_dir / mesh_file).is_file(), f"Missing MJCF mesh {mesh_file} referenced by {xml_path.name}"

    assert URDF_PATH.parent == SRC_ROOT / "robot"
    urdf_root = ET.parse(URDF_PATH).getroot()
    for mesh in urdf_root.iter("mesh"):
        filename = mesh.attrib.get("filename")
        if filename:
            assert (SRC_ROOT / "robot" / filename).is_file(), f"Missing URDF mesh {filename}"


def test_viewer_html_media_paths_exist():
    viewer_path = SRC_ROOT / "tools" / "viewer_3d.html"
    assert viewer_path.is_file(), "viewer_3d.html must live at src/tools/viewer_3d.html"
    html = viewer_path.read_text(encoding="utf-8")
    assert "duck_v2/" not in html, "viewer_3d.html still references duck_v2/"
    media_refs = sorted(set(re.findall(r"\.\./robot/media/[A-Za-z0-9_.-]+\.png", html)))
    assert media_refs, "src/tools/viewer_3d.html should reference ../robot/media/*.png images"
    for rel in media_refs:
        resolved = (viewer_path.parent / rel).resolve()
        assert resolved.is_file(), f"Missing media asset referenced by viewer_3d.html: {rel}"


def test_conftest_isolates_src_hardware_and_models():
    conftest_text = (REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert '"src" / "hardware"' in conftest_text or '"src/hardware"' in conftest_text
    assert '"models"' in conftest_text
    assert '"pipelines"' not in conftest_text


def test_gitignore_ignores_generated_and_doc_paths():
    gitignore_lines = [line.strip() for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()]
    for expected in ("artifacts/", "docs/his.md", "docs/plans/", "docs/tailieubandau.txt"):
        assert expected in gitignore_lines, f"Missing {expected} in .gitignore"
    assert "viewer_3d.html" not in gitignore_lines
    tracked = subprocess.check_output(["git", "ls-files", "artifacts"], cwd=REPO_ROOT, text=True).strip()
    assert tracked == "", f"Generated files under artifacts/ must not be tracked by Git: {tracked}"
