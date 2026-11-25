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

## v1 (tracking + spot memory) — status on the dev game

`python pipeline/analyze_v1.py <game> [--matcher bands|2d]` reads one set per settled possession from the v1 tracks,
holding six offensive *spots* (not people) through off-screen gaps and subs, and calls `Skip` when the offense
isn't settled. `python pipeline/score.py <moments.json> <answers.json>` scores it against Brian's frame key or his
50-clip review. On Scituate–Hanover: possession sets 9/44 (v0 10/44; always guessing 2-3-1 would get 13/44),
Skip 5/6 (v0 0/6), offense team 16/20 (same). Spot memory changes the score by about one possession. The blocker is
geometry: from one broadcast angle the across-field position of each player isn't measured well enough to tell
wings from crease players, so 1-4-1 / 2-3-1 / Open can't be separated by rules. Next step is a model trained on
labeled clips rather than more rules.

## Engine (whole game) — v2

The hub direction: upload a game, it is processed after the fact, and everything below is available with the video.

1. `pipeline/engine_extract.py` (Actions, `engine.yml`): the whole game in time-ordered chunks (default 20 in parallel,
   3 fps, 1280 wide, chunks overlap 4 s). Per frame: YOLO11 player boxes + BoT-SORT track ids (ReID), the lacrosse
   detector's role (Goalie/Longpole/Referee/Shortstick), torso color, stripes, camera motion, the goal; once a second the
   painted line segments and near sideline. After the chunk, up to 3 crops per track go through SigLIP. Raw output:
   `results/<game>-engine-raw/chunk_NNN/{dets.npz,frames.json}`.
2. `pipeline/engine_analyze.py <game> [--colors white,blue]` (any machine, minutes): field mapping, stitching, teams,
   radar, events, player data, sets. Output in `results/<game>-engine/`:
   - `field.json` — men's field in yards for drawing (x from the left end line 0..110, y from the near sideline;
     goals at x=15 and x=95, y=26.67).
   - `radar/chunk_NNN.json` — per frame `[player, x, y, team, role, seen]`; `seen=0` is a remembered off-screen spot.
   - `possessions.json` / `sets.json` — settled possessions: field end, offense team, set (top-first, plus
     `set_goal_first`), smoothed set timeline, man-up flag. `sets.json` is in the shape `pipeline/score.py` reads.
   - `events.json` — faceoffs (and which team came out with the ball), goals (a faceoff that is not a period start),
     man-up periods (6 attackers seen, never more than 5 defenders), clears/rides and re-settles between possessions.
     Every event carries a confidence word.
   - `players.json` — per stitched track: team, role, seconds mapped, distance, sprints (>6 yd/s), top speed,
     10-yd heatmap; team heatmaps flipped so each team attacks right.
   - `quality.json`, `summary.md` — coverage, error checks, failure reasons, timing.

Field mapping (`pipeline/field_geom.py`, from `fieldh_v2.py`): on shared turf the football grid (a line every 5 yards),
the thick near sideline and the goal give one image→field homography per keyframe (direct linear fit, no camera
model). Each frame borrows the nearest good keyframe in the same camera shot (≤10 s) through the frame-to-frame camera
motion. Checks: the goal mouth must map to 0.7–4.5 yd (true 2 yd) and 80% of players must land on the field; the
report gives the median mapped goal width and how much fits 1 s apart disagree. Fields without the football grid get
no field map yet (fallback to lacrosse lines/creases is not built).

Team split follows the roboflow/sports recipe (SigLIP embeddings → UMAP → KMeans), with torso-color fallback.
Set reading is template matching (Tactix-style) on the radar with 2 s smoothing; Tactix code is not copied.
No ball tracking yet: shots, saves, ground balls, assists are out of scope for this round.
