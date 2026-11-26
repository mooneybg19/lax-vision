"""Men's lacrosse field world model, in yards, for PnLCalib-style camera calibration.

Replaces the football-grid assumption in field_geom with the actual lacrosse markings, so the learned
landmark detector and the calibrator generalise to lacrosse-lined fields (not only shared football turf).

World frame (ground plane, z = 0):
  x: 0..110 along the length of the field
  y: 0..60 across the width
  goals on the centre line of the width (y = 30), 15 yd in from each end line -> x = 15 and x = 95.

Dimensions: NFHS/NCAA men's field 110 x 60 yd; goals 15 yd from the end line (80 yd apart); crease radius
9 ft = 3 yd; goal mouth 6 ft = 2 yd; restraining lines 20 yd from the centre line (x = 35, 75); wing lines
10 yd long, 10 yd from the centre of the field (y = 20 and 40), centred on midfield (x = 50..60); centre
line at x = 55. Sources: NFHS Boys Lacrosse and NCAA Men's Lacrosse rulebooks (field diagram).
"""
import numpy as np

LENGTH = 110.0
WIDTH = 60.0
GOAL_Y = WIDTH / 2          # 30
GOALS_X = (15.0, 95.0)
CREASE_R = 3.0
GOAL_HALF = 1.0             # half of the 2-yd mouth
RESTRAIN_X = (35.0, 75.0)
CENTER_X = 55.0
WING_Y = (20.0, 40.0)
WING_X = (50.0, 60.0)

# ---- named keypoints (world x,y in yards). Order is the detector's channel order; keep it stable. ----
KEYPOINTS = {
    "corner_0_0": (0.0, 0.0), "corner_0_60": (0.0, 60.0),
    "corner_110_0": (110.0, 0.0), "corner_110_60": (110.0, 60.0),
    "restrain_L_near": (35.0, 0.0), "restrain_L_far": (35.0, 60.0),
    "restrain_R_near": (75.0, 0.0), "restrain_R_far": (75.0, 60.0),
    "center_near": (55.0, 0.0), "center_far": (55.0, 60.0),
    "wing_L_near": (50.0, 20.0), "wing_L_far": (50.0, 40.0),
    "wing_R_near": (60.0, 20.0), "wing_R_far": (60.0, 40.0),
    # crease cardinal points + goal posts, both goals
    "creaseL_front": (18.0, 30.0), "creaseL_back": (12.0, 30.0),
    "creaseL_near": (15.0, 27.0), "creaseL_far": (15.0, 33.0),
    "goalL_near": (15.0, 29.0), "goalL_far": (15.0, 31.0),
    "creaseR_front": (92.0, 30.0), "creaseR_back": (98.0, 30.0),
    "creaseR_near": (95.0, 27.0), "creaseR_far": (95.0, 33.0),
    "goalR_near": (95.0, 29.0), "goalR_far": (95.0, 31.0),
}
KP_NAMES = list(KEYPOINTS.keys())
KP_WORLD = np.array([KEYPOINTS[n] for n in KP_NAMES], float)   # (N,2)


def _circle(cx, cy, r, n=48):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [[float(cx + r * np.cos(t)), float(cy + r * np.sin(t))] for t in a]


# ---- straight line segments and crease arcs (polylines) for the line head and for drawing ----
LINES = {
    "sideline_near": [[0.0, 0.0], [110.0, 0.0]],
    "sideline_far": [[0.0, 60.0], [110.0, 60.0]],
    "endline_L": [[0.0, 0.0], [0.0, 60.0]],
    "endline_R": [[110.0, 0.0], [110.0, 60.0]],
    "center": [[55.0, 0.0], [55.0, 60.0]],
    "restrain_L": [[35.0, 0.0], [35.0, 60.0]],
    "restrain_R": [[75.0, 0.0], [75.0, 60.0]],
    "wing_L": [[50.0, 20.0], [50.0, 40.0]],
    "wing_R": [[60.0, 20.0], [60.0, 40.0]],
    "goalmouth_L": [[15.0, 29.0], [15.0, 31.0]],
    "goalmouth_R": [[95.0, 29.0], [95.0, 31.0]],
}
ARCS = {"creaseL": _circle(15.0, 30.0, CREASE_R), "creaseR": _circle(95.0, 30.0, CREASE_R)}


def field_json():
    """Drawing spec in yards for the hub's radar/field and heatmap (length x width, lines + crease circles)."""
    return {
        "sport": "lacrosse-m", "length": LENGTH, "width": WIDTH, "units": "yd",
        "lines": LINES, "crease": [{"cx": 15.0, "cy": 30.0, "r": CREASE_R}, {"cx": 95.0, "cy": 30.0, "r": CREASE_R}],
        "goals": [{"x": 15.0, "y": 30.0}, {"x": 95.0, "y": 30.0}],
        "restraining_x": list(RESTRAIN_X), "center_x": CENTER_X,
    }


def draw(canvas, H, color=(255, 255, 255), thick=1):
    """Draw the field onto `canvas` given a world(yd)->image 3x3 homography H (for synthetic data + overlays)."""
    import cv2

    def proj(pts):
        P = (H @ np.array([[p[0], p[1], 1.0] for p in pts]).T).T
        return (P[:, :2] / P[:, 2:3]).astype(np.int32)

    for seg in LINES.values():
        q = proj(seg)
        cv2.line(canvas, tuple(q[0]), tuple(q[1]), color, thick, cv2.LINE_AA)
    for arc in ARCS.values():
        q = proj(arc)
        cv2.polylines(canvas, [q.reshape(-1, 1, 2)], True, color, thick, cv2.LINE_AA)
    return canvas
