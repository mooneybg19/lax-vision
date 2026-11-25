# Engine run: scituate-hanover-2024

- Film covered: 104 minutes in 20 time-ordered chunks at 3 frames a second.
- Compute: 400 CPU-minutes for extraction (3.9 per game minute) + 70 s analysis.
- Field mapping: 46% of frames have a field fit (goal mouth maps to 1.4 yd, true 2 yd; neighbouring fits 1 s apart disagree by 2.57 yd). 56% of player sightings are on the radar.
- Teams: white vs blue (SigLIP + UMAP + KMeans).
- Players: 4782 stitched tracks with physical data (one athlete can be several tracks until jersey numbers are attached).
- Possessions: 227 settled possessions; sets: Skip 148, Other 79.
- Events: clear 22, re-settle 54, man-up 15, faceoff 12, goal 12.

Confidence notes are in events.json; field-mapping failure reasons are in quality.json.

## Checks against what we know about this game (dev game; final score 19-4)

- Sets vs Brian's 50-clip review: 1/44 right (v0 10/44, v1 9/44, always-2-3-1 13/44); Skip 6/6.
  Vs his 22-frame key: 0/20 sets, offense team 12/17, Skip 2/2. Most settled possessions come out Skip or Other,
  because the mapped positions are too far off for the set templates (median template error 16 yd).
- Faceoffs found: 12 (the game had about 27: 23 goals + 4 period starts). Goals inferred: 12 of 23; the team
  credited is unreliable (11 of 12 given to white, but the 19-goal team wore the dark tops).
- Field fit: 46% of frames; spot checks show fits off by ~10 yd even where the line fit looks clean (goalie mapped
  9 yd off the goal line). This is the blocker for sets, man-up and physical numbers.
- Players: 4,782 stitched tracks (median 4.7 s each): identities break often, so per-player totals are partial.
