"""
Unified class taxonomy for combining IDD + DAWN + KITTI + Waymo (+ GTSRB
for sign *type*, handled separately as a classifier, not merged here).

Every converter script maps its own dataset's native categories onto
this single id space, so a merged dataloader can train one detector
across all sources. Missing classes in a given source (e.g. KITTI has
no traffic-light boxes) just produce no labels for that class — normal
and expected, not an error.

IMPORTANT: IDD's traffic taxonomy includes classes with no BDD100K/
KITTI/Waymo equivalent (autorickshaw, animal) — these matter for the
Indian-conditions goal, so they get their own ids rather than being
awkwardly folded into "car" or dropped.
"""

UNIFIED_CLASSES = {
    0: "car",
    1: "truck",
    2: "bus",
    3: "motorcycle",
    4: "bicycle",
    5: "rider",           # person actively riding a 2/3-wheeler
    6: "person",
    7: "traffic_light",
    8: "traffic_sign",
    9: "train",
    10: "autorickshaw",   # IDD-specific, no equivalent in the others
    11: "animal",         # IDD-specific (cattle etc. on the road)
}
NUM_UNIFIED_CLASSES = len(UNIFIED_CLASSES)
NAME_TO_ID = {v: k for k, v in UNIFIED_CLASSES.items()}

# --- per-source mappings -----------------------------------------------

IDD_TO_UNIFIED = {
    "car": "car", "truck": "truck", "bus": "bus", "motorcycle": "motorcycle",
    "bicycle": "bicycle", "rider": "rider", "person": "person",
    "traffic light": "traffic_light", "traffic sign": "traffic_sign",
    "train": "train", "vehicle fallback": "car",   # IDD's catch-all -> nearest equivalent
    "autorickshaw": "autorickshaw", "animal": "animal",
    "caravan": "truck", "trailer": "truck",
}

KITTI_TO_UNIFIED = {
    "Car": "car", "Van": "truck", "Truck": "truck", "Tram": "train",
    "Pedestrian": "person", "Person_sitting": "person",
    "Cyclist": "rider", "Misc": None, "DontCare": None,
}

WAYMO_TO_UNIFIED = {
    # Waymo's TYPE_* enum from label.proto
    "TYPE_VEHICLE": "car", "TYPE_PEDESTRIAN": "person",
    "TYPE_CYCLIST": "rider", "TYPE_SIGN": "traffic_sign",
}

# DAWN is weather-condition imagery, typically annotated in PASCAL VOC
# with mostly vehicle/person boxes (coverage varies by release) — map
# whatever category strings appear, unknowns are dropped with a warning
DAWN_TO_UNIFIED = {
    "car": "car", "truck": "truck", "bus": "bus", "motorcycle": "motorcycle",
    "bicycle": "bicycle", "person": "person",
}


def map_class(source_name: str, mapping: dict):
    """Returns unified class id, or None if this source class has no
    unified equivalent (e.g. KITTI's DontCare) — caller should skip it."""
    unified_name = mapping.get(source_name)
    if unified_name is None:
        return None
    return NAME_TO_ID[unified_name]
