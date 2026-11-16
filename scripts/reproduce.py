"""Reproduce one row of Table 1 (arXiv:2604.21991) for one algorithm configuration.

Reads a YAML spec (default: repro.yaml) naming the algorithm, the domains, the
seeds and the Hydra overrides, runs `run_algorithms.py` once per (domain, seed)
in parallel, and prints per-seed and aggregated final mean fitness and AUC.

Each job runs in its own working directory (symlinked task files) so the file
logger of parallel jobs never collides. Outputs land in ``outputs/repro/<label>/``:
``results.csv`` (one row per job) and ``curves.csv`` (mean fitness vs evals).
"""
import csv
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
FINAL_RE = re.compile(r"^final_(mean_fitness|auc): (\S+)$", re.M)
CURVE_RE = re.compile(r"^curve n_evals=(\d+) mean_fitness=(\S+)$", re.M)


def parse_seeds(spec):
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, list):
        return [int(s) for s in spec]
    lo, hi = str(spec).split("-")
    return list(range(int(lo), int(hi) + 1))


def run_job(spec, domain, seed, out_dir):
    job_dir = out_dir / "jobs" / f"{domain}_s{seed}"
    job_dir.mkdir(parents=True, exist_ok=True)
    for name in ("env_inits", "urdf"):
        link = job_dir / name
        if not link.is_symlink():
            try:
                link.symlink_to(REPO / name)
            except FileExistsError:
                pass
    overrides = [
        f"task={domain}",
        f"algorithm.name={spec['algorithm']}",
        f"seed={seed}",
        "wandb.enabled=false",
        f"algorithm.max_evals={spec['max_evals']}",
        f"hydra.run.dir={job_dir / 'hydra'}",
    ]
    overrides += list(spec.get("overrides", {}).get("common", []) or [])
    overrides += list(spec.get("overrides", {}).get(domain, []) or [])
    env = dict(os.environ)
    env.pop("SLURM_ARRAY_TASK_ID", None)
    t0 = time.time()
    with open(job_dir / "stdout.log", "w") as fh:
        proc = subprocess.run([sys.executable, str(REPO / "run_algorithms.py"), *overrides],
                              cwd=job_dir, stdout=fh, stderr=subprocess.STDOUT, env=env)
    wall = time.time() - t0
    text = (job_dir / "stdout.log").read_text(errors="replace")
    finals = dict(FINAL_RE.findall(text))
    curve = [(int(n), float(v)) for n, v in CURVE_RE.findall(text)]
    ok = proc.returncode == 0 and "mean_fitness" in finals and "auc" in finals
    if not ok:
        print(f"[FAIL] {domain} seed={seed} rc={proc.returncode}; tail:\n{text[-1500:]}", flush=True)
    return {
        "algorithm": spec["label"], "domain": domain, "seed": seed, "ok": ok,
        "final_mean_fitness": float(finals.get("mean_fitness", "nan")),
        "auc": float(finals.get("auc", "nan")),
        "wall_min": wall / 60.0, "curve": curve, "overrides": " ".join(overrides[6:]),
    }


def main():
    spec_path = REPO / (sys.argv[1] if len(sys.argv) > 1 else "repro.yaml")
    spec = yaml.safe_load(spec_path.read_text())
    spec.setdefault("label", spec["algorithm"])
    seeds = parse_seeds(spec["seeds"])
    out_dir = REPO / "outputs" / "repro" / spec["label"]
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(d, s) for d in spec["domains"] for s in seeds]
    print(f"spec: {spec}", flush=True)
    print(f"commit: {os.environ.get('ORX_COMMIT', '?')}  jobs: {len(jobs)}  workers: {spec['workers']}", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=spec["workers"]) as pool:
        futures = [pool.submit(run_job, spec, d, s, out_dir) for d, s in jobs]
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            print(f"done {r['domain']} seed={r['seed']} ok={r['ok']} "
                  f"final_mean_fitness={r['final_mean_fitness']:.6f} auc={r['auc']:.1f} "
                  f"wall={r['wall_min']:.1f}min [{len(results)}/{len(jobs)}]", flush=True)

    results.sort(key=lambda r: (r["domain"], r["seed"]))
    with open(out_dir / "results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[k for k in results[0] if k != "curve"])
        w.writeheader()
        for r in results:
            w.writerow({k: v for k, v in r.items() if k != "curve"})
    with open(out_dir / "curves.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["algorithm", "domain", "seed", "n_evals", "mean_fitness"])
        for r in results:
            for n, v in r["curve"]:
                w.writerow([r["algorithm"], r["domain"], r["seed"], n, v])

    print("\n=== per-seed results (csv) ===")
    print("algorithm,domain,seed,ok,final_mean_fitness,auc,wall_min")
    for r in results:
        print(f"{r['algorithm']},{r['domain']},{r['seed']},{r['ok']},{r['final_mean_fitness']:.6f},{r['auc']:.3f},{r['wall_min']:.2f}")

    print("\n=== summary (mean +- std over ok seeds; AUC x1e3 normalised = auc / max_evals * 1e3) ===")
    n_failed = 0
    for d in spec["domains"]:
        rs = [r for r in results if r["domain"] == d and r["ok"]]
        n_failed += sum(1 for r in results if r["domain"] == d and not r["ok"])
        if not rs:
            print(f"{spec['label']} {d}: no successful seeds")
            continue
        f = np.array([r["final_mean_fitness"] for r in rs])
        a = np.array([r["auc"] for r in rs]) / spec["max_evals"] * 1e3
        print(f"SUMMARY {spec['label']} {d} n={len(rs)} final_mean_fitness={f.mean():.4f} +- {f.std(ddof=1) if len(f) > 1 else 0:.4f} "
              f"auc_x1e3={a.mean():.1f} +- {a.std(ddof=1) if len(a) > 1 else 0:.1f}")
    if n_failed:
        print(f"{n_failed} job(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
