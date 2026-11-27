# Field-registration detector: training pipeline

Goal: from one broadcast frame, the mapping between image pixels and field yards (men's lacrosse field, possibly
lined for football), accurate to a fraction of a yard. Trained on the hand labels from `tools/labeler/`, starting
from PnLCalib's soccer detectors. Run it with `field_hrnet_training.ipynb` on a Colab T4.

## Conventions

- World: yards, `x` 0..110 left to right as seen from the camera, `y = 0` at the near (camera) sideline, `y = W` far
  (W = 60, or 53.33 on football-lined fields). Football lines: yard line N at `x = 5 + N` (left) / `105 - N` (right),
  hash rows `hash` yards in from each sideline (17.78 high school, 20 college). Landmark names and positions are a
  bit-exact port of the labeler (`pipeline/labels_io.py`, tested against the JavaScript in node).
- Image: 1280x720, **index coordinates** (pixel `i` has its centre at `i`; OpenCV). The labeler stores clicks in canvas
  coordinates (centre at `i + 0.5`); `game_solve` subtracts 0.5. The labeler's paint-snapped `H` is in index
  coordinates (its paint distance field is indexed that way).
- Lens: one radial coefficient per game, `p_d = c + (p_u - c)(1 + k1 r^2)`, `r = |p_u - c| / 1280`,
  `c = (639.5, 359.5)` (`synth_cam.distort`). `H_u` maps world -> undistorted pixels.

## Steps (files)

1. **Labels** `labels_io.py`: loads the export (list of documents, or a db dump), rejects a label if a world coordinate
   disagrees with the landmark definition, `H` is singular / puts a click behind the camera, the stated `click_off` /
   `rms` does not match the reprojection under `H`, or `H_clicks` is not the DLT of the clicks. `paint.py` is the
   labeler's paint detector, ported bit-exactly (tested on real frames).
2. **Per-game fit** `game_solve.py`: all frames of one game share the lens `k1` and (football) the hash distance; each
   frame has its own homography. Minimises click reprojection error plus the distance of the projected painted lines
   to the paint ridges in the frame (the labeler's own evidence, truncated at 6 px), robust soft-L1, sparse Jacobian.
   Start: grid over `k1` (and hash 17.78 / 20 / labelled) with per-frame refits. Outlier clicks: standardised by
   leverage, `|e| / (sigma sqrt(1 - h)) > 3.4` and `> 3 px`, removed and re-solved until stable. The hash estimate is
   snapped to 17.78 or 20 when within `max(3 se, 0.4 yd)` (0.4 yd ~ the length of a tick) and the game re-solved;
   otherwise kept with a warning. Frames flagged `low_paint_match` (< 60% after the fit) or `high_rms` are dropped
   from training; other flags are informational. Report: `RUN/game_solve.json`.
3. **Targets** `field_targets.py`: one fixed channel list = 35 lacrosse + 78 football landmarks (+ background), and
   13 lacrosse + 18 yard lines (5..105 step 5 minus 35/55/75, which are lacrosse lines) + 2 hash rows (+ all-lines).
   Keypoints: Gaussian, sigma 2 output px, only where the projected point is inside the image. Lines (NBJW layout):
   Gaussians at the two visible extremities of each line, last channel Gaussians along every visible line.
   Channels that do not exist on the frame's field are **masked** (no loss, excluded from the softmax). On football
   games, lacrosse-only marks (box, wings, x = 0/110 end lines, creases, goals, faceoff X) are trained only if some
   frame of that game has a click on one of them (`lax_painted`); the restraining/centre/sideline points (= football
   lines) always.
4. **Model** `field_model.py`: two HRNetV2-w48 nets as in NBJW/PnLCalib (keypoints: masked softmax over channels;
   lines: sigmoid), each from its soccer checkpoint (`SV_kp`, `SV_lines`) with only the final 1x1 conv re-initialised.
   Input 960x540 RGB in [0,1], output 480x270. Loss: MSE weighted `1 + 10 * target` on existing channels.
   `train_field.py`: AdamW (body 1e-4, new head 1e-3), 1 warm-up epoch then cosine, stages 1-2 frozen for 8 epochs,
   AMP, batch 2 x 2 accumulation. Augmentation: flip (left/right channels permuted, near/far unchanged, permutation
   derived from world coordinates and tested), colour jitter, blur, noise, JPEG.
5. **Calibration** `field_calib.py`: RANSAC homography on detected keypoints (8 px), robust refinement with the line
   extremities (point-on-line) and the all-lines map (truncated distance field), sanity checks (>= 5 inliers, or 4 + 2
   agreeing lines; inliers not collinear; near sideline below far; image centre maps near the field).
6. **Engine** `field_infer.py`: `keyframe_h(frame, kp_best.pt)` keeps its old interface (image -> engine field H).

## Splits

`field_data.split`, deterministic per frame id: **test** = every frame of `jesuit-westview` + ~15% of
`scituate-hanover-2024` (dev game); **val** = ~10% of the rest (only to pick the best epoch); **train** = the rest.

## The metric (`field_calib.yard_errors`, `eval_field.py`)

For each test frame: image grid points every 16 px; keep those whose position under the **label** (game_solve `H_u`
+ the game's `k1`) is on the field, `0 <= x <= 110`, `0 <= y <= W`; map the same pixels with the **predicted**
calibration; error = distance in yards. Reported:

- `success_rate`: frames where calibration returned an H that passed the sanity checks;
- `points_median_yd`, `points_p90_yd`: over all grid points of the calibrated frames;
- `frame_median_of_medians_yd`, `frame_p90_of_medians_yd`: per-frame median error, then median / p90 over frames;
- `all_frames_*_fail_inf`: the same over all frames with failures counted as infinite error.

Configs: **A** keypoints only, field type known, k1 = 0 (what the engine can do for a known field);
**B** + lines; **C** B with the field type chosen automatically; **D** B with the game's k1 known; **old_unet** the
earlier synthetic-data detector + `field_cal` (60-yd world). The current engine's `field_geom.keyframe_h` needs the
engine's per-second line segments, which exist only for the dev game, so it is not in this table.

Note: the grid metric weighs far-field pixels (few pixels per yard) heavily; p90 is dominated by the far side.

## What has been verified (CPU, no real labels yet)

- Landmark/line/fitH port == labeler JS (node); paint port == labeler JS on real frames, bit-exact.
- game_solve on synthetic games with known truth: recovers k1 and the hash, flags an injected bad click, field error
  drops (football, clicks only: median 1.78 -> 0.09 yd, p90 6.1 -> 0.37 yd when the hash was mislabelled).
- Targets: render -> decode error 0.03 px median; flipped targets == targets of the mirrored camera; GPU targets ==
  CPU targets; masked channels get exactly zero gradient.
- Perfect heatmaps -> decode -> calibrate -> metric: ~0.01 yd median (pins every coordinate convention).
- Whole path (prepare, train kp with resume, train line, eval, field_infer) runs on fake labels
  (`python pipeline/smoke_field.py --weights DIR`). Fake labels say nothing about accuracy.
