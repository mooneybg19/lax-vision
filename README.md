# lax-vision

Men's lacrosse film: find players, split the teams, and read the offensive set during settled possessions. Private research. Built on the soccer/hockey approach (detect, split teams by appearance per game, map to the field from a landmark, match the shape to a set library, smooth over time), not on the football pipeline.

Film is never stored here. The repo keeps detections, a few still frames, and a note of where each game came from (`games/`).

## Games

| file | role | source |
|---|---|---|
| `games/scituate-hanover-2024.json` | **dev** (used to build and tune) | archive.org, Scituate vs Hanover, 5/25/2024 (MA), 1080p |
| `games/anne-arundel-carroll-2021.json` | held-out | archive.org, Anne Arundel CC vs Carroll CC, 4/20/2021 (MD), 720p |
| `games/jesuit-westview.json` | held-out | archive.org, Jesuit HS vs Westview (OR), 1080p |
| `games/stillwater-bsm-2025.json` | held-out | archive.org, Stillwater vs Benilde-St. Margaret's, 5/15/2025 (MN), 1080p; most of the 1st half missing |

All four are community-TV style sideline broadcasts. No coach press-box film was found yet.

## Pipeline

1. `pipeline/detect.py` (GitHub Actions, `detect.yml`, the game split into parallel pieces): 2 frames a second, YOLO11 person boxes, torso color signature, a stripes score, whether the feet are on the turf (field color learned per frame), the goal found by its orange frame, and the direction of the painted cross-field lines near the goal.
2. `pipeline/analyze.py` (runs anywhere, seconds): team split per game with 10-minute refits for lighting; benches and sideline crowds dropped; settled offense = goal in view, at least 4 of each team in the zone, almost nobody in a third color, held 4+ seconds; offense = the team standing farther from the goal; set = offense counted in four bands (top, middle, front of the net, behind) and matched to the library.
3. `pipeline/frames.py` (Actions, `frames.yml`): clean frames for labeling (`label/`, with `items.json`) and annotated frames (`annotated/`).

### Set library and naming

Calls are stored **top-first** (Crown Point convention, from the restraining line toward/behind the goal): 1-4-1, 2-2-2, 2-3-1, 1-3-2, 3-3, 3-1-2 (umbrella), Open (wheel/circle, incl. the rotated 2-top/2-behind look), Other. `analyze.py --naming goal-first` (or `SET_NAMING=goal-first`) adds the Lacrosse Unlimited reading as `set_display`; every moment also carries `set_goal_first`. 2-3-1 and 1-3-2 swap, 3-1-2 becomes 2-1-3, and symmetric and named sets don't change.

## Licenses of the parts

- Ultralytics YOLO11: AGPL-3.0. Fine for personal use.
- Roboflow100-VL `lacrosse-object-detection` (Goalie, Longpole, Referee, Shortstick): the dataset README states **License: MIT**. It's pulled from the Hugging Face mirror `gatilin/rf100-vl`, so no Roboflow key is needed. Used by `pipeline/train_lax.py` / `train-lax.yml`.
- Tactix (soccer tactical engine, GPL-3.0): used as the design model only. No code copied.
- scikit-learn, OpenCV, NumPy: BSD/Apache.
