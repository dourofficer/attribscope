"""
experiments/svd/run.py

Minimal YAML-driven runner. Builds argv from config and shells out to
attribscope.svd.compute_svd / compute_scores unchanged.

    python -m experiments.run fit   --config experiments/configs/default.yaml
    python -m experiments.run score --config experiments/configs/default.yaml

    # override anything, one-shot:
    python -m experiments.run fit --config ... --set shared.models=[qwen3-8b]

    # see the commands that would run, without running them:
    python -m experiments.run score --config ... --dry-run
"""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from rich.console import Console
import yaml

CONSOLE = Console()

def load_cfg(path: Path, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(path.read_text())
    for ov in overrides:
        key, _, val = ov.partition("=")
        parts, node = key.split("."), cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


def poolings_for(cfg: dict, rep_kind: str) -> list[str]:
    return cfg["shared"]["poolings"] or cfg["shared"]["poolings_by_rep_kind"][rep_kind]

def format_command(module: str, argv: list[str]) -> str:
    head = f"{sys.executable} -m {module}"
    if not argv:
        return head

    groups, current = [], []
    for token in argv:
        if token.startswith("--") and current:
            groups.append(current)
            current = []
        current.append(token)
    groups.append(current)

    formatted_groups = [" ".join(shlex.quote(t) for t in g) 
                        for g in groups]
    args = " \\\n    ".join(formatted_groups)
    return f"{head} \\\n    {args}"

def run(module: str, argv: list[str], dry_run: bool) -> None:
    cmd = [sys.executable, "-m", module, *argv]
    CONSOLE.print(format_command(module, argv), style="green")
    CONSOLE.rule()
    if not dry_run:
        subprocess.run(cmd, check=True)


def run_fit(cfg: dict, dry_run: bool) -> None:
    s, f = cfg["shared"], cfg["fit"]
    for rep_kind in s["rep_kinds"]:
        for pooling in poolings_for(cfg, rep_kind):
            run("attribscope.svd.compute_svd", [
                "--models",       *s["models"],
                "--subsets",      *f["subsets"],
                "--rep-dir",      f"{s['outputs_root']}/{rep_kind}",
                "--data-dir",     s["data_dir"],
                "--pooling",      pooling,
                "--n-components", str(f["n_components"]),
                "--device",       s["device"],
            ], dry_run)


def run_score(cfg: dict, dry_run: bool) -> None:
    s, sc = cfg["shared"], cfg["score"]
    n_score = sc["n_components_score"]
    n_score = [str(n_score)] if isinstance(n_score, str) else [str(c) for c in n_score]

    for rep_kind in s["rep_kinds"]:
        rep_dir = f"{s['outputs_root']}/{rep_kind}"
        poolings = poolings_for(cfg, rep_kind)
        for fit_subset in sc["fit_subsets"]:
            run("attribscope.svd.compute_scores", [
                "--models",             *s["models"],
                "--score-subsets",      *sc["score_subsets"],
                "--fit-subset",         fit_subset,
                "--base-dir",           rep_dir,
                "--data-dir",           s["data_dir"],
                "--out-dir",            rep_dir,
                "--poolings",           *poolings,
                "--n-components-fit",   str(sc["n_components_fit"]),
                "--n-components-score", *n_score,
                "--ks",                 *[str(k) for k in sc["ks"]],
                "--device",             s["device"],
            ], dry_run)


def main() -> None:
    p = argparse.ArgumentParser(prog="experiments.run")
    sub = p.add_subparsers(dest="phase", required=True)
    for name in ("fit", "score"):
        sp = sub.add_parser(name)
        sp.add_argument("--config", type=Path, required=True)
        sp.add_argument("--set", dest="overrides", action="append", default=[],
                        metavar="KEY=VALUE", help="e.g. --set shared.models=[qwen3-8b]")
        sp.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    cfg = load_cfg(args.config, args.overrides)
    {"fit": run_fit, "score": run_score}[args.phase](cfg, args.dry_run)


if __name__ == "__main__":
    main()