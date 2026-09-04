"""
Phase 1 class taxonomy — deliberately small: object detection (car,
pedestrian, cyclist, etc from KITTI) + generic sign LOCATION (from
GTSDB). Sign TYPE classification (GTSRB, stop vs yield vs speed-limit)
is Phase 2, same as your other stretch features.
"""

PHASE1_CLASSES = {
    0: "car",
    1: "van_truck",     # KITTI's Van + Truck collapsed together for Phase 1 simplicity
    2: "tram",
    3: "pedestrian",
    4: "cyclist",
    5: "traffic_sign",  # from GTSDB — location only, not yet classified by type
}
NUM_PHASE1_CLASSES = len(PHASE1_CLASSES)
NAME_TO_ID = {v: k for k, v in PHASE1_CLASSES.items()}

KITTI_TO_PHASE1 = {
    "Car": "car",
    "Van": "van_truck", "Truck": "van_truck",
    "Tram": "tram",
    "Pedestrian": "pedestrian", "Person_sitting": "pedestrian",
    "Cyclist": "cyclist",
    "Misc": None, "DontCare": None,
}


def map_kitti_class(name):
    unified = KITTI_TO_PHASE1.get(name)
    return NAME_TO_ID[unified] if unified else None


# GTSDB's 43 numeric sign classes all collapse to one "traffic_sign" id
# for Phase 1 — Phase 2 (GTSRB) is what tells you *which* sign it is
GTSDB_SIGN_ID = NAME_TO_ID["traffic_sign"]
