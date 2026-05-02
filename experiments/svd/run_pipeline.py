from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from itertools import product
from pathlib import Path
from tqdm import tqdm

import yaml
from rich.console import Console

CONSOLE = Console()
MODULE  = "attribscope.svd2.pipeline"


# ── Helpers from caller ───────────────────────────────────────────────────────

def load_cfg(path: Path, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(path.read_text())
    for ov in overrides:
        key, _, val = ov.partition("=")
        parts, node = key.split("."), cfg
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = yaml.safe_load(val)
    return cfg


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
    args = " \\\n    ".join(" ".join(shlex.quote(t) for t in g) for g in groups)
    return f"{head} \\\n    {args}"


def run(module: str, argv: list[str], dry_run: bool) -> None:
    CONSOLE.print(format_command(module, argv), style="green")
    CONSOLE.rule()
    if not dry_run:
        subprocess.run([sys.executable, "-m", module, *argv], check=True)


# ── Sweep internals ───────────────────────────────────────────────────────────

def _iter_shared(shared: dict):
    """Yield (model, rep_type, pooling, loss, temperature) combos.
    Hidden reps are loss-agnostic (_resolve_dir ignores loss for hidden),
    so non-ntp losses are skipped to avoid redundant runs.
    """
    configs = []
    for model, (rep_type, pooling), (loss, temperature) in product(
        shared["models"], shared["reps"], shared["losses"]
    ):
        if rep_type == "hidden" and loss != "ntp":
            continue
        configs.append((model, rep_type, pooling, loss, temperature))
    print(f"Total: {len(configs)} valid (model x rep_type x pooling x loss x temperature) configs")
    return configs


def _common_argv(shared: dict, model: str, rep_type: str, pooling: str,
                 loss: str, temperature: float | None) -> list[str]:
    argv = [
        "--reps-root",          str(shared["reps-root"]),
        "--data-root",          str(shared["data-root"]),
        "--outputs-root",       str(shared["outputs-root"]),
        "--model",              model,
        "--rep-type",           rep_type,
        "--pooling",            pooling,
        "--loss",               loss,
        "--n-components-fit",   str(shared["n-components-fit"]),
        "--n-components-score", *[str(x) for x in shared["n-components-score"]],
        "--ks",                 *[str(k) for k in shared["ks"]],
        "--device",             shared["device"],
    ]
    if temperature is not None:
        argv += ["--temperature", str(temperature)]
    return argv


# ── Sweep functions ───────────────────────────────────────────────────────────

def sweep_indata(cfg: dict, dry_run: bool) -> None:
    shared = cfg["shared"]
    indata = cfg["indata"]
    for model, rep_type, pooling, loss, temperature in tqdm(_iter_shared(shared)):
        for subset, ratio, seed in product(
            indata["subsets"], indata["ratios"], indata["seeds"]
        ):
            run(MODULE, [
                "indata",
                *_common_argv(shared, model, rep_type, pooling, loss, temperature),
                "--subset",      subset,
                "--split-ratio", str(ratio),
                "--split-seed",  str(seed),
            ], dry_run)


def sweep_cross(cfg: dict, dry_run: bool) -> None:
    shared = cfg["shared"]
    cross  = cfg["cross"]
    for model, rep_type, pooling, loss, temperature in tqdm(_iter_shared(shared)):
        for fit_subset, score_subset in cross["subsets"]:
            run(MODULE, [
                "cross",
                *_common_argv(shared, model, rep_type, pooling, loss, temperature),
                "--fit-subset",   fit_subset,
                "--score-subset", score_subset,
            ], dry_run)


def sweep_self_ceiling(cfg: dict, dry_run: bool) -> None:
    shared   = cfg["shared"]
    self_cfg = cfg["self_ceiling"]
    for model, rep_type, pooling, loss, temperature in tqdm(_iter_shared(shared)):
        for subset in self_cfg["subsets"]:
            run(MODULE, [
                "self-ceiling",
                *_common_argv(shared, model, rep_type, pooling, loss, temperature),
                "--subset", subset,
            ], dry_run)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    def common(q: argparse.ArgumentParser) -> None:
        q.add_argument("--config",  type=Path, required=True,
                       help="Path to the YAML config file.")
        q.add_argument("--set",     nargs="*", default=[], metavar="KEY=VAL",
                       help="Override config values, e.g. shared.device=cpu")
        q.add_argument("--dry-run", action="store_true",
                       help="Print commands without running them.")

    for name in ("indata", "cross", "self"):
        common(sub.add_parser(name))

    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg  = load_cfg(args.config, args.set)

    if   args.mode == "indata": sweep_indata(cfg, args.dry_run)
    elif args.mode == "cross":  sweep_cross(cfg, args.dry_run)
    elif args.mode == "self":   sweep_self_ceiling(cfg, args.dry_run)


if __name__ == "__main__":
    main()