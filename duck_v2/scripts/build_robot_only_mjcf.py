"""Build the standalone robot MJCF from the canonical V2 simulation scene."""

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from duck_v2.robot_spec import MODEL_PATH, MJCF_MODEL_NAME, ROBOT_ONLY_MJCF_PATH


WORLD_OBJECTS = {
    "floor",
    "pillar_l1_left",
    "pillar_l1_right",
    "pillar_l2_center",
    "pillar_l3_left",
    "pillar_l3_right",
    "pillar_back",
    "stair_step_1",
    "stair_step_2",
    "stair_step_3",
    "stair_platform",
    "stair_dn_1",
    "stair_dn_2",
    "floor_exit",
}


def build_robot_only_mjcf(source: Path = MODEL_PATH, output: Path = ROBOT_ONLY_MJCF_PATH) -> Path:
    tree = ET.parse(source)
    root = tree.getroot()
    root.set("model", MJCF_MODEL_NAME + "_robot")

    visual = root.find("visual")
    if visual is not None:
        root.remove(visual)

    asset = root.find("asset")
    if asset is not None:
        for element in list(asset):
            if element.tag == "texture" or (element.tag == "material" and element.get("name") == "floor_mat"):
                asset.remove(element)

    worldbody = root.find("worldbody")
    if worldbody is not None:
        for element in list(worldbody):
            if element.tag == "light":
                worldbody.remove(element)
            elif element.tag == "geom" and element.get("name") in WORLD_OBJECTS:
                worldbody.remove(element)

    output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return output


if __name__ == "__main__":
    print(build_robot_only_mjcf())
