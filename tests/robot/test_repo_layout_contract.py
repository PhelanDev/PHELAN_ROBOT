"""Contract tests enforcing the 5-pillar CyberDuck V2 repository layout."""

import re
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from robot.robot_spec import MODEL_PATH, REPO_ROOT, ROBOT_ONLY_MJCF_PATH, URDF_PATH

LEGACY_TOP_LEVEL_DIRS = ("duck_v2", "training", "pipelines", "scripts")
ACTIVE_CODE_DIRS = ("robot", "cad", "sim", "hardware", "tools", "tests")


def test_no_legacy_top_level_directories_or_imports():
    for name in LEGACY_TOP_LEVEL_DIRS:
        assert not (REPO_ROOT / name).exists(), f"Legacy top-level directory still exists: {name}/"

    py_files = [REPO_ROOT / "duck.py"] + [
        p
        for folder in ACTIVE_CODE_DIRS
        for p in (REPO_ROOT / folder).rglob("*.py")
        if "__pycache__" not in p.parts
    ]
    legacy_import_re = re.compile(r"^\s*(?:from|import)\s+(?:duck_v2|training|pipelines|scripts)\b", re.MULTILINE)
    offenders = []
    for path in py_files:
        text = path.read_text(encoding="utf-8")
        for match in legacy_import_re.finditer(text):
            offenders.append(f"{path.relative_to(REPO_ROOT).as_posix()}: {match.group(0).strip()}")
    assert not offenders, "Legacy imports remain:\n" + "\n".join(offenders)


def test_tests_root_has_no_stray_test_files():
    stray = sorted(p.name for p in (REPO_ROOT / "tests").glob("test_*.py"))
    assert not stray, f"Stray flat test files at tests/ root: {stray}"
    for domain in ("cad", "robot", "hardware", "sim"):
        domain_dir = REPO_ROOT / "tests" / domain
        assert domain_dir.is_dir(), f"Missing test domain directory: tests/{domain}/"
        assert list(domain_dir.glob("test_*.py")), f"Empty test domain directory: tests/{domain}/"


def test_robot_xml_urdf_meshes_colocated():
    meshes_dir = REPO_ROOT / "robot" / "meshes"
    assert meshes_dir.is_dir()
    for xml_path in (MODEL_PATH, ROBOT_ONLY_MJCF_PATH):
        assert xml_path.parent == REPO_ROOT / "robot"
        root = ET.parse(xml_path).getroot()
        for mesh in root.iter("mesh"):
            mesh_file = mesh.attrib.get("file")
            if mesh_file:
                assert (meshes_dir / mesh_file).is_file(), f"Missing MJCF mesh {mesh_file} referenced by {xml_path.name}"

    assert URDF_PATH.parent == REPO_ROOT / "robot"
    urdf_root = ET.parse(URDF_PATH).getroot()
    for mesh in urdf_root.iter("mesh"):
        filename = mesh.attrib.get("filename")
        if filename:
            assert (REPO_ROOT / "robot" / filename).is_file(), f"Missing URDF mesh {filename}"


def test_viewer_html_media_paths_exist():
    html = (REPO_ROOT / "viewer_3d.html").read_text(encoding="utf-8")
    assert "duck_v2/" not in html, "viewer_3d.html still references duck_v2/"
    media_refs = sorted(set(re.findall(r"robot/media/[A-Za-z0-9_.-]+\.png", html)))
    assert media_refs, "viewer_3d.html should reference robot/media/*.png images"
    for rel in media_refs:
        assert (REPO_ROOT / rel).is_file(), f"Missing media asset referenced by viewer_3d.html: {rel}"


def test_conftest_isolates_hardware_and_models():
    conftest_text = (REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")
    assert '"hardware"' in conftest_text
    assert '"models"' in conftest_text
    assert '"pipelines"' not in conftest_text


def test_gitignore_ignores_artifacts_dir():
    gitignore_lines = [line.strip() for line in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()]
    assert "artifacts/" in gitignore_lines
    tracked = subprocess.check_output(["git", "ls-files", "artifacts"], cwd=REPO_ROOT, text=True).strip()
    assert tracked == "", f"Generated files under artifacts/ must not be tracked by Git: {tracked}"
