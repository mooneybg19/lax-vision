# Field landmark labeler

Published as a Claude artifact ("Field landmark labeling"). Labels are stored in the artifact's database,
collection `labels`, one document per frame.

Each saved label holds the clicked landmarks (pixels and field yards), the paint-snapped homography `H`
(field yards -> image px, row-major 3x3, near/camera sideline at y = 0), the click-only homography, the
field type (lacrosse / football lines), field width, hash spacing, paint match and its shifted-field
background, and a quality tier: `paint` (verified against the painted lines), `points` (verified by the
clicks checking each other), or `paint_points_disagree` (lines match but a click is off; resolved by the
per-game joint fit during training).

Frames come from `results/label_frames/` (pipeline/label_frames.py), packed per game into
`bundles/*.jpg` (concatenated JPEGs; offsets in manifest.json).
