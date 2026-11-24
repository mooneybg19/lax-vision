"""Score set calls against Brian's answer key.

Usage: python pipeline/score.py results/<game>-sets[-v1]/moments.json answers/<game>.json

Each labeled frame is matched to the moment that contains its time (1 s slack). Two scores:
  - set: the moment's call at that time (the timeline segment containing it, when the run has a timeline;
    otherwise the moment's single call) vs Brian's set. 'Skip' frames are scored separately: right if the
    tool has no moment there or calls it 'Other'.
  - offense: the moment's offense color vs Brian's.
All calls compared top-first. This is the dev game: a score here is for building, not proof.
"""
import json, sys


def call_at(m, t):
    for s in m.get("timeline", []):
        if s["frm"] - 0.5 <= t <= s["to"] + 0.5:
            return s["set"]
    return m["set"]


def score_moments(M, K, quiet=False):
    """Brian's clip review: one set per v0 moment (matched by time overlap). The tool's possession call
    (set at initiation) vs his. 'Skip' is right only if the tool says Skip (or Other)."""
    ok = n = sk_ok = sk_n = 0
    conf = {}
    for k in K:
        m = max(M, key=lambda m: min(m["end"], k["end"]) - max(m["start"], k["start"]))
        if min(m["end"], k["end"]) - max(m["start"], k["start"]) <= 0:
            m = None
        tool = m["set"] if m else None
        if k["set"] == "Skip":
            sk_n += 1; sk_ok += tool in (None, "Skip", "Other")
        else:
            n += 1; ok += tool == k["set"]
        conf[(k["set"], tool)] = conf.get((k["set"], tool), 0) + 1
    if not quiet:
        print("answer -> tool (count)")
        for (a_, t_), c in sorted(conf.items(), key=lambda x: (x[0][0], -x[1])):
            print(f"  {a_:<6} -> {str(t_):<6} {c}")
        print(f"\nPossession sets right: {ok}/{n}   Skip handled: {sk_ok}/{sk_n}")
    return dict(set_ok=ok, set_n=n, skip_ok=sk_ok, skip_n=sk_n)


def score(moments_path, key_path, quiet=False):
    M = json.load(open(moments_path))["moments"]
    KK = json.load(open(key_path))
    if "moments" in KK:
        return score_moments(M, KK["moments"], quiet)
    K = KK["frames"]
    rows = []
    set_ok = set_n = off_ok = off_n = skip_ok = skip_n = 0
    for f in K:
        t = f["seconds"]
        m = next((m for m in M if m["start"] - 1 <= t <= m["end"] + 1), None)
        tool = call_at(m, t) if m else None
        poss = m["set"] if m else None
        off = (m.get("offense") or "").lower() if m else None
        if f["set"] == "Skip":
            skip_n += 1
            ok = tool in (None, "Other", "Skip")
            skip_ok += ok
        else:
            set_n += 1
            ok = tool == f["set"]
            set_ok += ok
            if m and f.get("offense"):
                off_n += 1
                off_ok += off == f["offense"].lower()
        rows.append((f["id"], f["set"], f.get("offense") or "", tool or "-", poss or "-", off or "-", "OK" if ok else ""))
    if not quiet:
        print(f"{'frame':<8} {'answer':<7} {'off':<6} {'tool@t':<7} {'poss':<7} {'tool off':<9}")
        for r in rows:
            print(f"{r[0]:<8} {r[1]:<7} {r[2]:<6} {r[3]:<7} {r[4]:<7} {r[5]:<9} {r[6]}")
        print(f"\nSets right: {set_ok}/{set_n}   Offense team right: {off_ok}/{off_n}   'Skip' frames handled: {skip_ok}/{skip_n}")
    return dict(set_ok=set_ok, set_n=set_n, off_ok=off_ok, off_n=off_n, skip_ok=skip_ok, skip_n=skip_n)


if __name__ == "__main__":
    score(sys.argv[1], sys.argv[2])
