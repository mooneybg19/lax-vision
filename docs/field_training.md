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
   frame has its own homography. Minimises click reprojection error plus the distance of the projected painted-line
   samples to **sub-pixel paint-ridge centres** found in the frame (`paint.ridge_points`: the labeler's ridge strength,
   smoothed and parabola-refined; the labeler's whole-pixel ridges sit ~0.85 px off the centre of a 5-px line, which
   biased the lens estimate), truncated at 6 px, robust soft-L1, sparse Jacobian. Start: grid over `k1` (0.01 steps;
   hash 17.78 / 20 / labelled) with per-frame refits. Frames whose own observations cannot pin their H down (predicted
   1-sigma position of the visible field, p90, > 15 px at the start, e.g. clicks nearly on one line) are left out;
   after the fit, > 6 px is flagged `poorly_constrained`. Outlier clicks: standardised by leverage,
   `|e| / (sigma sqrt(1 - h)) > 3.4` and `> 3 px`, removed and re-solved until stable. The hash estimate is used as
   estimated when its standard error is <= 0.3 yd (the painted ticks are 2 ft long; the image decides), otherwise it
   is fixed at the nearest standard and re-solved. Frames flagged `low_paint_match` (< 60% after the fit),
   `high_rms` or `poorly_constrained` are dropped from training; other flags are informational. A game-level warning
   is printed when the fit moves the clicks a lot (paint and clicks disagree). Report: `RUN/game_solve.json`.
   All homographies are kept with a canonical sign (`labels_io.canon`: w > 0 in front of the camera); dividing by
   `H[2,2]` alone flips it whenever the near-left corner is behind the camera plane (a camera panned to the right end).
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
   Input 960x540 RGB in [0,1], output 480x270. Losses: keypoints = soft cross-entropy between the masked softmax and the
   per-pixel target distribution, weighted `1 + 10 * peak target`; lines = weighted BCE. (NBJW's MSE on the softmax
   saturated at "background everywhere" with a re-initialised head in a CPU overfit check: loss flat for 60 steps,
   no detections; with CE the same check detected all 35 landmarks of the frame, flipped and unflipped, with the
   correct channel names, within 120 steps.)
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
earlier synthetic-data detector + `field_cal` (60-yd world); **engine** the current engine's field fit
(`field_geom.keyframe_h` + shot chaining, read from `results/<game>-engine/fieldfit`), only on the dev game's test
frames because the engine was only run on that game. Frames the engine did not map count as failures; the engine
does not store which goal it registered to, so the better of the two goals is scored (this favours the engine).

Note: the grid metric weighs far-field pixels (few pixels per yard) heavily; p90 is dominated by the far side.

## What has been verified (CPU, no real labels yet)

- Landmark/line/fitH port == labeler JS (node); paint port == labeler JS on real frames, bit-exact.
- game_solve, synthetic games with known truth (`tests/test_game_solve.py`):
  football, clicks only, hash mislabelled 17.78 (truth 20), k1 = -0.06: k1 -0.058 +- 0.007, hash 20.00, the injected
  15-px click flagged; field error of the kept frames median 1.60 -> 0.07 yd, p90 4.48 -> 0.23 yd (19/30 kept, the
  rest poorly constrained by 5-8 noisy clicks).
  lacrosse with painted frames, k1 = 0.05: k1 0.045, field error median 0.23 -> 0.015 yd, p90 1.9 -> 0.065 yd.
  Real frames with synthetic lines painted over the real field (8 frames per game): k1 within 0.004 of the truth in
  all four games, hash 17.79 (truth 17.78).
- Targets: render -> decode error 0.03 px median; flipped targets == targets of the mirrored camera; GPU targets ==
  CPU targets; masked channels get exactly zero gradient. Real-frame overlay: `results/field_training_checks/`.
- Perfect heatmaps -> decode -> calibrate -> metric: ~0.01 yd median (pins every coordinate convention).
- Cameras with the world origin behind the camera plane (`tests/test_sign.py`).
- One-frame overfit on CPU (480x272): `results/field_training_checks/cpu_overfit_one_frame.log`.
- Whole path (prepare, train kp with resume, train line, eval incl. old-UNet and engine baselines, field_infer) runs on
  fake labels (`python pipeline/smoke_field.py --weights DIR`), and the notebook itself was executed end to end on CPU
  with Colab stubs and fake labels. Fake labels say nothing about accuracy.
