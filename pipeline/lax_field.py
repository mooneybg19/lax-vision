"""Men's lacrosse field world model, in yards, for PnLCalib-style camera calibration.

Replaces the football-grid assumption in field_geom with the actual lacrosse markings, so the learned
landmark detector and the calibrator generalise to lacrosse-lined fields (not only shared football turf).

World frame (ground plane, z = 0):
  x: 0..110 along the length of the field
  y: 0..60 across the width
  goals on the centre line of the width (y = 30), 15 yd in from each end line -> x = 15 and x = 95.

Dimensions (USA Lacrosse "How to line a men's field", 2026; NFHS/NCAA): 110 x 60 yd (width may be reduced to
53 1/3 on football fields, so sideline points depend on WIDTH while centre-relative markings do not); goal line
15 yd from each end line; crease radius 9 ft = 3 yd centred on the goal line; goal mouth 6 ft = 2 yd; restraining
line 20 yd from each goal line (x = 35, 75), sideline to sideline; goal-area box sides 20 yd either side of the
field centre (y = c -/+ 20) from the restraining line to the end line; wing lines 20 yd either side of the field
centre, 10 yd each side of the centre line (x = 45..65); centre line x = 55 with the faceoff X at its middle.
"""
import numpy as np

LENGTH = 110.0
WIDTH = 60.0                # regulation; 53.33 on many football-lined fields (see keypoints(width))
GOAL_Y = WIDTH / 2          # 30
GOALS_X = (15.0, 95.0)
CREASE_R = 3.0
GOAL_HALF = 1.0             # half of the 2-yd mouth
RESTRAIN_X = (35.0, 75.0)
CENTER_X = 55.0
WING_OFF = 20.0            # wing lines and box sides: 20 yd either side of the field centre
WING_X = (45.0, 65.0)

# ---- named keypoints. Order is the detector's channel order; keep it stable. ----
def keypoints(width=WIDTH):
    """Named landmarks (x, y) in yards for a field of the given width (y = 0 is the near/camera sideline)."""
    c = width / 2.0; lo, hi = c - WING_OFF, c + WING_OFF
    return {
        "corner_0_0": (0.0, 0.0), "corner_0_60": (0.0, width),
        "corner_110_0": (110.0, 0.0), "corner_110_60": (110.0, width),
        "restrain_L_near": (35.0, 0.0), "restrain_L_far": (35.0, width),
        "restrain_R_near": (75.0, 0.0), "restrain_R_far": (75.0, width),
        "center_near": (55.0, 0.0), "center_far": (55.0, width), "faceoff_X": (55.0, c),
        # goal-area box: corners on the restraining line and on the end line
        "boxL_near": (35.0, lo), "boxL_far": (35.0, hi), "boxL_end_near": (0.0, lo), "boxL_end_far": (0.0, hi),
        "boxR_near": (75.0, lo), "boxR_far": (75.0, hi), "boxR_end_near": (110.0, lo), "boxR_end_far": (110.0, hi),
        # wing line end points
        "wing_near_L": (45.0, lo), "wing_near_R": (65.0, lo), "wing_far_L": (45.0, hi), "wing_far_R": (65.0, hi),
        # crease cardinal points + goal posts, both goals
        "creaseL_front": (18.0, c), "creaseL_back": (12.0, c), "creaseL_near": (15.0, c - 3), "creaseL_far": (15.0, c + 3),
        "goalL_near": (15.0, c - 1), "goalL_far": (15.0, c + 1),
        "creaseR_front": (92.0, c), "creaseR_back": (98.0, c), "creaseR_near": (95.0, c - 3), "creaseR_far": (95.0, c + 3),
        "goalR_near": (95.0, c - 1), "goalR_far": (95.0, c + 1),
    }


KEYPOINTS = keypoints()
KP_NAMES = list(KEYPOINTS.keys())
KP_WORLD = np.array([KEYPOINTS[n] for n in KP_NAMES], float)   # (N,2)


def _circle(cx, cy, r, n=48):
    a = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return [[float(cx + r * np.cos(t)), float(cy + r * np.sin(t))] for t in a]


# ---- straight line segments and crease arcs (polylines) for the line head and for drawing ----
def lines(width=WIDTH):
    c = width / 2.0; lo, hi = c - WING_OFF, c + WING_OFF
    return {
        "sideline_near": [[0.0, 0.0], [110.0, 0.0]], "sideline_far": [[0.0, width], [110.0, width]],
        "endline_L": [[0.0, 0.0], [0.0, width]], "endline_R": [[110.0, 0.0], [110.0, width]],
        "center": [[55.0, 0.0], [55.0, width]],
        "restrain_L": [[35.0, 0.0], [35.0, width]], "restrain_R": [[75.0, 0.0], [75.0, width]],
        "boxL_near": [[0.0, lo], [35.0, lo]], "boxL_far": [[0.0, hi], [35.0, hi]],
        "boxR_near": [[75.0, lo], [110.0, lo]], "boxR_far": [[75.0, hi], [110.0, hi]],
        "wing_near": [[45.0, lo], [65.0, lo]], "wing_far": [[45.0, hi], [65.0, hi]],
        "goalmouth_L": [[15.0, c - 1], [15.0, c + 1]], "goalmouth_R": [[95.0, c - 1], [95.0, c + 1]],
    }


def arcs(width=WIDTH):
    c = width / 2.0
    return {"creaseL": _circle(15.0, c, CREASE_R), "creaseR": _circle(95.0, c, CREASE_R)}


LINES = lines()
ARCS = arcs()


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
