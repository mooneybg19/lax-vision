"""Per-game joint refinement of the hand labels.

One game = one camera, so constants that a single frame cannot pin down are shared by all its frames:
  k1    one radial lens coefficient (synth_cam.distort: p_d = c + (p_u - c)(1 + k1 r^2), r = |p_u - c| / 1280)
  hash  football fields: the hash-row distance from each sideline (yd), estimated, then snapped to 17.78 (HS) or
        20 (college) when the estimate is consistent with one of them
  width optional (fit_width=True): the field width for lacrosse-only fields (reported, not snapped)
Per frame: an undistorted homography H_f (8 dof; world yd -> undistorted px).

Observations (all in image pixels):
  clicks  D_k1(H_f X(name; width, hash)) - click
  paint   (if the frame jpg is available) the distance from each projected painted-line sample to the nearest painted
          ridge in the frame (paint.py, the labeler's own detector), truncated at 6 px exactly like the labeler's
          snap -- so the same evidence the labeler used, but now with the lens and the per-game constants free.
Robust (soft-L1, 3 px) least squares over all frames at once with a sparse Jacobian. Clicks whose residual exceeds
3 sigma (and 3 px) are dropped and the game is re-solved (up to 3 rounds). Frames are flagged when a click is an
outlier, when the paint match after the solve is poor, when their H moved a lot, or when they have no redundancy.
"""
import os
import numpy as np
import cv2
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

import labels_io as LI
import paint as PT
from synth_cam import distort, undistort, IMG_W, IMG_H

T_IMG = np.array([[1 / 640, 0, -1.0], [0, 1 / 640, -360 / 640], [0, 0, 1.0]])
T_INV = np.linalg.inv(T_IMG)
CLICK_W, PAINT_W, F_SCALE = 1.0, 1.0, 3.0
K1_PRIOR_SD = 0.5


def _H_from(H0, d):
    D = np.eye(3) + np.append(d, 0.0).reshape(3, 3)
    return T_INV @ D @ T_IMG @ H0


def _lin(fn, W0, h0):
    """Evaluate fn(W, h) -> (N,2) at base, W+1, h+1: exact linear coefficients of y (and x) in W and hash."""
    b = np.asarray(fn(W0, h0), float)
    return b, np.asarray(fn(W0 + 1, h0), float) - b, np.asarray(fn(W0, h0 + 1), float) - b


class Frame:
    def __init__(self, doc, W0, h0, fb, paintfield=None):
        self.doc = doc; self.id = doc["frame"]
        self.names = list(doc["pts"].keys())
        # labeler clicks are canvas coords (pixel centre at i+0.5); everything here is index coords (centre at i),
        # the convention of the paint distance field the labeler snapped H to
        self.clicks = np.array([doc["pts"][n] for n in self.names], float) - 0.5
        self.keep = np.ones(len(self.names), bool)
        self.cb, self.cW, self.ch = _lin(lambda W, h: [LI.world_of(n, W, h, fb) for n in self.names], W0, h0)
        self.H0 = LI.H_of(doc); self.H0 = self.H0 / self.H0[2, 2]      # the label's H (reporting)
        self.Hb = self.H0.copy()                                          # base of the per-frame parametrisation
        self.P = paintfield
        self.sb = None
        if paintfield is not None:
            # line samples visible under the label H (fixed set, like the labeler's snap), linear in W and hash
            S0, _ = PT.paint_samples(W0, fb, h0)
            vis = PT.visible(self.H0, S0, paintfield)
            if vis.sum() >= 40:
                def samp(W, h):
                    S, _ = PT.paint_samples(W, fb, h)
                    if len(S) != len(S0):        # vertical-line sample counts depend on W: re-sample at base t
                        S = _resample_like(W0, W, h, fb)
                    return S[vis, :2]
                self.sb, self.sW, self.sh = _lin(samp, W0, h0)


def _resample_like(W0, W, h, fb):
    """Paint samples at width W but with the sample count / spacing parameter of width W0 (keeps rows aligned)."""
    S0, _ = PT.paint_samples(W0, fb, h)
    out = []
    lines = LI.yard_lines(W) if fb else LI.field_lines(W)
    lines0 = LI.yard_lines(W0) if fb else LI.field_lines(W0)
    for (a, b), (a0, b0) in zip(lines, lines0):
        L0 = np.hypot(b0[0] - a0[0], b0[1] - a0[1]); n = max(2, int(np.ceil(L0 / .5)))
        L = np.hypot(b[0] - a[0], b[1] - a[1]); nx = -(b[1] - a[1]) / L; ny = (b[0] - a[0]) / L
        for i in range(n + 1):
            out.append([a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n, nx, ny])
    if fb:
        for X in range(5, 106):
            out.append([X, h, 0, 1]); out.append([X, W - h, 0, 1])
    else:
        for gx in (15, 95):
            for i in range(72):
                t = i / 72 * 6.2832
                out.append([gx + 3 * np.cos(t), W / 2 + 3 * np.sin(t), np.cos(t), np.sin(t)])
    out = np.array(out, float)
    assert len(out) == len(S0)
    return out


class GameSolver:
    def __init__(self, docs, frames_dir=None, fit_hash=True, fit_width=False, use_paint=True, verbose=True):
        self.game = docs[0]["game"]
        W0, h0, fb = LI.doc_geometry(docs[0])
        for d in docs:
            g = LI.doc_geometry(d)
            if g[2] != fb or abs(g[0] - W0) > 1e-6 or (fb and abs(g[1] - h0) > 1e-6):
                raise ValueError(f"{self.game}: frames disagree on field type/width/hash ({d['frame']}: {g} vs {(W0, h0, fb)}); "
                                 "fix the labels so the whole game uses one setting")
        self.W0, self.h0, self.fb, self.verbose = W0, h0, fb, verbose
        self.frames = []
        for d in docs:
            P = None
            if use_paint and frames_dir:
                p = os.path.join(frames_dir, d["frame"] + ".jpg")
                im = cv2.imread(p)
                if im is not None and im.shape[1] == IMG_W and im.shape[0] == IMG_H:
                    P = PT.paint_field(im)
            self.frames.append(Frame(d, W0, h0, fb, P))
        # only estimate constants that something observes
        hash_obs = any(np.any(f.ch[:, 1] != 0) for f in self.frames) or any(f.sb is not None and np.any(f.sh[:, 1] != 0) for f in self.frames)
        width_obs = any(np.any(f.cW[:, 1] != 0) for f in self.frames) or any(f.sb is not None and np.any(f.sW[:, 1] != 0) for f in self.frames)
        self.fit_hash = bool(fb and fit_hash and hash_obs)
        self.fit_width = bool(fit_width and width_obs)

    # --- parameter layout: [k1, (hash), (width), frame0 d(8), frame1 d(8), ...]
    def _shared(self, x):
        i = 1; h = self.h0; W = self.W0
        if self.fit_hash:
            h = x[i]; i += 1
        if self.fit_width:
            W = x[i]; i += 1
        return x[0], h, W, i

    def _frame_res(self, f, d, k1, h, W):
        H = _H_from(f.Hb, d)
        X = f.cb + f.cW * (W - self.W0) + f.ch * (h - self.h0)
        q, w = LI.proj(H, X)
        q = distort(q, k1)
        rc = (q - f.clicks) * CLICK_W
        rc[w <= 0] = 50.0
        rc[~f.keep] = 0.0
        out = [rc.ravel()]
        if f.sb is not None:
            S = f.sb + f.sW * (W - self.W0) + f.sh * (h - self.h0)
            qs, ws = LI.proj(H, S)
            qs = distort(qs, k1)
            dd = PT.dt_at(f.P, qs)
            dd = np.where(np.isfinite(dd) & (ws > 0), np.minimum(dd, PT.TRUNC), PT.TRUNC)
            out.append(dd * PAINT_W)
        return np.concatenate(out)

    def residuals(self, x):
        k1, h, W, i0 = self._shared(x)
        r = [np.array([k1 / K1_PRIOR_SD])]
        for j, f in enumerate(self.frames):
            r.append(self._frame_res(f, x[i0 + 8 * j:i0 + 8 * j + 8], k1, h, W))
        return np.concatenate(r)

    def _sparsity(self, x):
        _, _, _, i0 = self._shared(x)
        sizes = [len(self._frame_res(f, np.zeros(8), 0.0, self.h0, self.W0)) for f in self.frames]
        A = lil_matrix((1 + sum(sizes), len(x)), dtype=int)
        A[0, 0] = 1
        row = 1
        for j, s in enumerate(sizes):
            A[row:row + s, :i0] = 1
            A[row:row + s, i0 + 8 * j:i0 + 8 * j + 8] = 1
            row += s
        return A

    def _x0(self, k1=0.0, h=None, W=None):
        x = [k1]
        if self.fit_hash:
            x.append(self.h0 if h is None else h)
        if self.fit_width:
            x.append(self.W0 if W is None else W)
        return np.concatenate([np.array(x, float), np.zeros(8 * len(self.frames))])

    def _bounds(self, n):
        lo = np.full(n, -np.inf); hi = np.full(n, np.inf)
        lo[0], hi[0] = -0.9, 0.9
        i = 1
        if self.fit_hash:
            lo[i], hi[i] = 10.0, 25.0; i += 1
        if self.fit_width:
            lo[i], hi[i] = 45.0, 65.0
        return lo, hi

    def _run(self, x0):
        res = least_squares(self.residuals, x0, jac_sparsity=self._sparsity(x0), bounds=self._bounds(len(x0)),
                            method="trf", loss="soft_l1", f_scale=getattr(self, "f_scale", F_SCALE), x_scale="jac", max_nfev=200,
                            ftol=1e-10, xtol=1e-10, gtol=1e-10)
        return res

    def _shared_se(self, res):
        J = res.jac.toarray() if hasattr(res.jac, "toarray") else np.asarray(res.jac)
        m, n = J.shape
        s2 = 2 * res.cost / max(1, m - n)
        i0 = self._shared(res.x)[3]
        A = J.T @ J
        try:
            Saa = A[:i0, :i0] - A[:i0, i0:] @ np.linalg.pinv(A[i0:, i0:]) @ A[i0:, :i0]
            cov = np.linalg.pinv(Saa) * s2
            return np.sqrt(np.maximum(np.diag(cov), 0))
        except np.linalg.LinAlgError:
            return np.full(i0, np.nan)

    def click_errors(self, x):
        k1, h, W, i0 = self._shared(x)
        out = []
        for j, f in enumerate(self.frames):
            H = _H_from(f.Hb, x[i0 + 8 * j:i0 + 8 * j + 8])
            X = f.cb + f.cW * (W - self.W0) + f.ch * (h - self.h0)
            out.append(np.hypot(*(distort(LI.proj(H, X)[0], k1) - f.clicks).T))
        return out

    def _frame_cost(self, f, H, k1, h, W):
        r = self._frame_res_H(f, H, k1, h, W)
        return float(np.sum(2 * F_SCALE ** 2 * (np.sqrt(1 + (r / F_SCALE) ** 2) - 1)))

    def _frame_res_H(self, f, H, k1, h, W):
        X = f.cb + f.cW * (W - self.W0) + f.ch * (h - self.h0)
        q, w = LI.proj(H, X)
        rc = (distort(q, k1) - f.clicks) * CLICK_W
        rc[w <= 0] = 50.0
        out = [rc.ravel()]
        if f.sb is not None:
            S = f.sb + f.sW * (W - self.W0) + f.sh * (h - self.h0)
            qs, ws = LI.proj(H, S)
            dd = PT.dt_at(f.P, distort(qs, k1))
            out.append(np.where(np.isfinite(dd) & (ws > 0), np.minimum(dd, PT.TRUNC), PT.TRUNC) * PAINT_W)
        return np.concatenate(out)

    def _refit_frame(self, f, k1, h, W):
        """Best per-frame H for given shared constants: the label H or a DLT (all clicks, and leave-one-out)."""
        X = f.cb + f.cW * (W - self.W0) + f.ch * (h - self.h0)
        U = undistort(f.clicks, k1)
        cands = [f.H0]
        n = len(f.names)
        subsets = [np.arange(n)] + ([np.delete(np.arange(n), i) for i in range(n)] if n >= 6 else [])
        for idx in subsets:
            try:
                Hc = LI.fit_h(X[idx], U[idx])
                if np.all(np.isfinite(Hc)):
                    cands.append(Hc / Hc[2, 2])
            except Exception:
                pass
        costs = [self._frame_cost(f, Hc, k1, h, W) for Hc in cands]
        j = int(np.argmin(costs))
        return cands[j], costs[j]

    def initialise(self):
        """Grid over the shared constants with per-frame refits; sets each frame's base H. Returns (k1, h, W)."""
        k1s = np.linspace(-0.3, 0.3, 13)
        hs = sorted({self.h0, LI.HASH_HS, LI.HASH_COLLEGE}) if self.fit_hash else [self.h0]
        Ws = sorted({self.W0, 60.0, LI.FB_WIDTH}) if self.fit_width else [self.W0]
        best = None
        for k1 in k1s:
            for h in hs:
                for W in Ws:
                    c = sum(self._refit_frame(f, k1, h, W)[1] for f in self.frames) + (k1 / K1_PRIOR_SD) ** 2
                    if best is None or c < best[0]:
                        best = (c, k1, h, W)
        _, k1, h, W = best
        for f in self.frames:
            f.Hb = self._refit_frame(f, k1, h, W)[0]
        if self.verbose:
            print(f"  [{self.game}] init grid: k1 {k1:+.3f}" + (f" hash {h}" if self.fit_hash else "") + (f" width {W}" if self.fit_width else ""))
        return k1, h, W

    def _leverage(self, x):
        """Per-click leverage (mean of the two diagonal hat-matrix entries) of each frame's 8-dof fit, shared
        constants held fixed. A click's residual has variance 2 sigma^2 (1 - h) if it is not an outlier."""
        k1, h, W, i0 = self._shared(x)
        out = []
        for j, f in enumerate(self.frames):
            d0 = x[i0 + 8 * j:i0 + 8 * j + 8]
            keep_save = f.keep; f.keep = np.ones(len(f.names), bool)
            r0 = self._frame_res(f, d0, k1, h, W)
            J = np.empty((len(r0), 8))
            for k in range(8):
                dd = d0.copy(); dd[k] += 1e-6
                J[:, k] = (self._frame_res(f, dd, k1, h, W) - r0) / 1e-6
            f.keep = keep_save
            Wt = np.ones(len(r0)); Wt[:2 * len(f.names)] = np.repeat(f.keep, 2)    # only kept clicks fit the frame
            Jw = J * Wt[:, None]
            try:
                M = np.linalg.pinv(Jw.T @ Jw)
            except np.linalg.LinAlgError:
                M = np.zeros((8, 8))
            Jc = J[:2 * len(f.names)]
            hd = np.einsum("ij,jk,ik->i", Jc, M, Jc)           # diag of J M J^T for the click rows
            out.append(np.clip(hd.reshape(-1, 2).mean(1), 0, 1))
        return out

    def _reject(self, x, update=True):
        """Outlier clicks: |e| / (sigma sqrt(1-h)) > 3.4 (chi^2_2 at 99.7%) and |e| > 3 px. A click not used in the
        fit (already rejected) has 1+h instead of 1-h. Returns (sigma, number of keep flags changed)."""
        errs = self.click_errors(x); levs = self._leverage(x)
        z = []
        for e, hl, f in zip(errs, levs, self.frames):
            m = f.keep & (hl < 0.98)
            z.append(e[m] / np.sqrt(1 - hl[m]))
        z = np.concatenate(z) if z else np.array([])
        sigma = max(float(np.median(z)) / 1.1774, 0.3) if len(z) else 1.0
        changed = 0
        if update:
            for e, hl, f in zip(errs, levs, self.frames):
                var = np.where(f.keep, np.clip(1 - hl, 1e-3, None), 1 + hl)
                std = e / (sigma * np.sqrt(var))
                keep = ~((std > 3.4) & (e > 3.0))
                min_keep = 4 if f.sb is None else 0
                if keep.sum() < min_keep:
                    keep = np.zeros(len(e), bool); keep[np.argsort(std)[:min_keep]] = True
                changed += int((keep != f.keep).sum()); f.keep = keep
        self.levs = levs
        return sigma, changed

    def solve(self, max_rounds=4):
        rep = {"game": self.game, "field": "football" if self.fb else "lacrosse", "width_labeled": self.W0,
               "hash_labeled": self.h0 if self.fb else None, "n_frames": len(self.frames),
               "n_paint_frames": sum(f.sb is not None for f in self.frames)}
        x0 = self._x0()
        e0_list = self.click_errors(x0)              # under the label H, no lens, labeled hash/width
        e0 = np.concatenate(e0_list)
        rep["before"] = {"click_rms_px": float(np.sqrt(np.mean(e0 ** 2))), "click_median_px": float(np.median(e0)),
                         "click_p90_px": float(np.percentile(e0, 90))}
        k1i, hi, Wi = self.initialise()
        xi = self._x0(k1=k1i, h=hi, W=Wi)
        res = None
        # continuation: a wide robust scale first (so a wrong hash / lens is pulled in rather than explained away
        # as outliers), then the working scale with outlier rejection recomputed from scratch each round
        self.f_scale = 30.0
        res = self._run(xi)
        self.f_scale = F_SCALE
        for rnd in range(max_rounds + 1):
            res = self._run(res.x)
            sigma, changed = self._reject(res.x)
            if self.verbose:
                print(f"  [{self.game}] round {rnd}: cost {res.cost:.1f}  sigma {sigma:.2f}px  "
                      f"outlier clicks {sum(int((~f.keep).sum()) for f in self.frames)} (changed {changed})")
            if changed == 0:
                break
        if changed:
            res = self._run(res.x)
            sigma, _ = self._reject(res.x, update=False)
        self.sigma = sigma
        se = self._shared_se(res)
        k1, h, W, i0 = self._shared(res.x)
        rep["k1"] = float(k1); rep["k1_se"] = float(se[0])
        if self.fit_hash:
            hse = float(se[1])
            std = min((LI.HASH_HS, LI.HASH_COLLEGE), key=lambda s: abs(s - h))
            rep["hash_est"] = float(h); rep["hash_se"] = hse
            if abs(h - std) <= max(3 * hse, 0.4):
                rep["hash"] = std; rep["hash_snapped"] = True
                # re-solve with the hash fixed at the standard value so every frame is consistent with it
                self.fit_hash = False; self.h0_true = std
                hold = self.h0; self.h0 = std
                for f in self.frames:
                    f.cb = f.cb + f.ch * (std - hold)
                    if f.sb is not None:
                        f.sb = f.sb + f.sh * (std - hold)
                xs = self._x0(k1=k1, W=W if self.fit_width else None)
                xs[self._shared(xs)[3]:] = res.x[i0:]
                res = self._run(xs)
                se = self._shared_se(res)
                k1, h, W, i0 = self._shared(res.x)
                rep["k1"] = float(k1); rep["k1_se"] = float(se[0])
            else:
                rep["hash"] = float(h); rep["hash_snapped"] = False
                rep.setdefault("warnings", []).append(
                    f"hash estimate {h:.2f}+-{hse:.2f} yd matches neither 17.78 nor 20; using the estimate")
        else:
            rep["hash"] = self.h0 if self.fb else None
        if self.fit_width:
            rep["width_est"] = float(W); rep["width_se"] = float(se[-1])
        rep["width"] = float(W)
        errs = self.click_errors(res.x)
        allk = np.concatenate([e[f.keep] for e, f in zip(errs, self.frames)])
        rep["after"] = {"click_rms_px": float(np.sqrt(np.mean(allk ** 2))), "click_median_px": float(np.median(allk)),
                        "click_p90_px": float(np.percentile(allk, 90)), "sigma_px": float(self.sigma)}
        rep["outlier_clicks"] = {f.id: [n for n, k in zip(f.names, f.keep) if not k] for f in self.frames if not f.keep.all()}
        frames = {}
        g = np.array([[x, y] for x in np.linspace(0, 110, 23) for y in np.linspace(0, W, 12)])
        for j, f in enumerate(self.frames):
            H = _H_from(f.Hb, res.x[i0 + 8 * j:i0 + 8 * j + 8])
            fr = {"H": (H / H[2, 2]).ravel().tolist(), "n_clicks": len(f.names), "n_kept": int(f.keep.sum()),
                  "click_rms_before": float(np.sqrt(np.mean(e0_list[j] ** 2))),
                  "click_rms_after": float(np.sqrt(np.mean(errs[j][f.keep] ** 2))), "flags": []}
            # how far the corrected mapping moved inside the image (px), on the visible field
            q0, w0 = LI.proj(f.H0, g); q1 = distort(LI.proj(H, g)[0], k1)
            vis = (w0 > 0) & (q0[:, 0] >= 0) & (q0[:, 0] < IMG_W) & (q0[:, 1] >= 0) & (q0[:, 1] < IMG_H)
            fr["moved_px_max"] = float(np.max(np.hypot(*(q1 - q0)[vis].T))) if vis.any() else None
            if f.sb is not None:
                pm0 = PT.paint_match(f.H0, f.P, self.W0, self.fb, self._lab_hash())
                pm1 = PT.paint_match(H, f.P, W, self.fb, rep["hash"] if self.fb else LI.HASH_HS,
                                     dist=lambda q, k=k1: distort(q, k))
                fr["paint_before"] = pm0 and round(pm0["frac"], 3); fr["paint_after"] = pm1 and round(pm1["frac"], 3)
                if pm1 and pm1["frac"] < 0.6:
                    fr["flags"].append("low_paint_match")
            if not f.keep.all():
                fr["flags"].append("outlier_click")
            if f.sb is None and f.keep.sum() <= 4:
                fr["flags"].append("unchecked_no_redundancy")
            if fr["moved_px_max"] is not None and fr["moved_px_max"] > 15:
                fr["flags"].append("moved_gt_15px")
            if fr["click_rms_after"] > 4 * max(self.sigma, 1.0):
                fr["flags"].append("high_rms")
            frames[f.id] = fr
        rep["frames"] = frames
        rep["flagged_frames"] = {k: v["flags"] for k, v in frames.items() if v["flags"]}
        return rep

    def _lab_hash(self):
        return float(self.frames[0].doc.get("hash") or LI.HASH_HS)


def solve_all(docs, frames_dir=None, verbose=True, **kw):
    """{game: report} for all games in docs."""
    games = {}
    for d in docs:
        games.setdefault(d["game"], []).append(d)
    out = {}
    for g, ds in sorted(games.items()):
        if verbose:
            print(f"game {g}: {len(ds)} frames")
        out[g] = GameSolver(ds, frames_dir=frames_dir, verbose=verbose, **kw).solve()
        r = out[g]
        if verbose:
            print(f"  k1 {r['k1']:+.4f} +- {r['k1_se']:.4f}" + (f"  hash {r.get('hash_est', r['hash']):.2f} -> {r['hash']}" if r["field"] == "football" else "")
                  + f"  click rms {r['before']['click_rms_px']:.2f} -> {r['after']['click_rms_px']:.2f} px"
                  + f"  flagged frames {len(r['flagged_frames'])}")
    return out


EXCLUDE_FLAGS = {"low_paint_match", "high_rms"}


def apply(docs, reports, exclude_flags=EXCLUDE_FLAGS):
    """Corrected training docs: adds H_u (undistorted H), k1, hash_used, width_used. Drops frames with severe flags."""
    out = []
    for d in docs:
        r = reports[d["game"]]; fr = r["frames"][d["frame"]]
        if set(fr["flags"]) & set(exclude_flags):
            continue
        e = dict(d)
        e["H_u"] = fr["H"]; e["k1"] = r["k1"]; e["hash_used"] = r["hash"]; e["width_used"] = r["width"]
        out.append(e)
    return out
