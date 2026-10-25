"""Score the model against hand-counted images (practice for the judges' 'accuracy over 95%' tests)."""
from __future__ import annotations

import csv
from pathlib import Path


def evaluate(engine, folder: str, truth_csv: str, out_csv: str | None = None) -> dict:
    """truth_csv columns: image, then one column per target species, optional 'total'."""
    targets = list(engine.cfg.detect.targets)
    folder = Path(folder)
    with open(truth_csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    per = {t: {"hit": 0, "n": 0, "abs_err": 0, "true_sum": 0} for t in targets + ["total"]}
    detail = []
    for r in rows:
        img = folder / r["image"]
        if not img.exists():
            continue
        res = engine.detect_file(img)
        truth = {t: int(float(r.get(t, 0) or 0)) for t in targets}
        truth_total = int(float(r["total"])) if r.get("total") not in (None, "") else None
        pred = {t: res.counts.get(t, 0) for t in targets}
        row = {"image": r["image"]}
        for t in targets:
            per[t]["n"] += 1
            per[t]["hit"] += int(pred[t] == truth[t])
            per[t]["abs_err"] += abs(pred[t] - truth[t])
            per[t]["true_sum"] += truth[t]
            row[f"{t} true"], row[f"{t} pred"] = truth[t], pred[t]
        if truth_total is not None:
            per["total"]["n"] += 1
            per["total"]["hit"] += int(res.total == truth_total)
            per["total"]["abs_err"] += abs(res.total - truth_total)
            per["total"]["true_sum"] += truth_total
            row["total true"], row["total pred"] = truth_total, res.total
        detail.append(row)
    summary = {}
    for k, v in per.items():
        if v["n"]:
            count_acc = 1 - v["abs_err"] / v["true_sum"] if v["true_sum"] else (1.0 if v["abs_err"] == 0 else 0.0)
            summary[k] = {"images": v["n"], "exact_match": round(v["hit"] / v["n"], 3),
                          "count_accuracy": round(max(0.0, count_acc), 3),
                          "mean_abs_error": round(v["abs_err"] / v["n"], 3)}
    if out_csv and detail:
        cols = sorted({c for d in detail for c in d}, key=lambda c: (c != "image", c))
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(detail)
    return {"summary": summary, "images": len(detail)}
