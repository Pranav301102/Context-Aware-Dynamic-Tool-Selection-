"""Hydra experiment runner (workbook Table 9: Hydra configs; NFR8: every run has a manifest).

Config groups live in ``conf/`` at the repo root:

  strategy     cats | all | flat_reranked | cats_gbt | cats_no_decompose  (CATSConfig + baselines)
  data         default (enriched) | raw; keys enrich, jev, llm_router, embedder, corpus_dir, tasks_dir
  calibration  lac | aps | independent; keys method, alpha_branch, alpha_tool, mode, split
  experiment   bench | ablate | validity | scale | cache

Each run calls the existing experiment functions in ``cats.bench.experiments`` (baseline_strategies
with the harness for bench, run_variants, coverage_validity, scaling, cache_sessions), writes
``results/<name>.json`` like ``cats <command>`` does, appends bench's per-task records to the
Parquet metrics store, and wraps everything in a run manifest (``results/manifests/<name>.json``).

    python -m cats.runner experiment=bench strategy=cats data.enrich=true
    python -m cats.runner experiment=bench strategy=all data=raw calibration=aps
    python -m cats.runner -m seed=0,1,2 experiment=bench            # one result + manifest per seed
    python -m cats.runner experiment=validity calibration=independent experiment.n_seeds=5
    python -m cats.runner --cfg job experiment=scale                # print the composed config

Hydra's own logs and the composed configs go to ``results/hydra/`` (``$CATS_HYDRA_DIR``).
Needs ``pip install hydra-core`` (extra ``runner``).
"""
from __future__ import annotations

import json
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .build import DATA, ROOT, System, build_registry, build_system
from .pipeline import CATSConfig
from .registry import Registry

CONF_DIR = ROOT / "conf"
_CATS_FIELDS = {f.name for f in fields(CATSConfig)}


# ------------------------------------------------------------------------- config
def _path(p: str | None) -> Path | None:
    """A configured path: as given when absolute or present under the launch directory, else under
    the repo root."""
    if p in (None, ""):
        return None
    path = Path(str(p))
    return path if path.is_absolute() or path.exists() else ROOT / path


def cats_config(cfg: DictConfig) -> CATSConfig:
    """CATSConfig from the strategy, calibration and data groups."""
    strategy = OmegaConf.to_container(cfg.strategy.cats, resolve=True)
    if not isinstance(strategy, dict):
        raise TypeError("strategy.cats must be a mapping of CATSConfig fields")
    kw: dict[str, Any] = {str(k): v for k, v in strategy.items() if k in _CATS_FIELDS}
    cal = cfg.calibration
    kw.update(method=cal.method, alpha_branch=float(cal.alpha_branch), alpha_tool=float(cal.alpha_tool),
              calibration=cal.mode, split_calibration=bool(cal.split))
    if cfg.data.jev:
        kw["jev"] = str(_path(cfg.data.jev_dir))
    if cfg.data.llm_router:
        kw["llm_router"] = str(_path(cfg.data.llm_router_dir))
    return replace(CATSConfig(), **kw)


def run_name(cfg: DictConfig) -> str:
    """``cfg.name`` or e.g. ``bench-cats-enrich-lac-s0`` (data and calibration choices, seed)."""
    if cfg.get("name"):
        return str(cfg.name)
    d, cal = cfg.data, cfg.calibration
    data = ("enrich" if d.enrich else "raw") + ("-jev" if d.jev else "") + ("-llmr" if d.llm_router else "")
    calib = cal.method + ("-indep" if cal.mode == "independent" else "")
    if float(cal.alpha_tool) != 0.05:
        calib += f"-at{cal.alpha_tool:g}"
    return f"{cfg.experiment.kind}-{cfg.strategy.name}-{data}-{calib}-s{cfg.seed}"


def results_dir(cfg: DictConfig) -> Path:
    return _path(cfg.get("results_dir")) or ROOT / "results"


def _enrichment(cfg: DictConfig) -> Path | None:
    return _path(cfg.data.enrich_dir) if cfg.data.enrich else None


def _registry(cfg: DictConfig) -> Registry | None:
    corpus = _path(cfg.data.corpus_dir)
    return build_registry([corpus]) if corpus is not None else None


def build(cfg: DictConfig, cats_cfg: CATSConfig) -> System:
    """The system for bench and cache, honouring data.corpus_dir / data.tasks_dir / data.embedder."""
    reg = _registry(cfg)
    tasks_dir = _path(cfg.data.tasks_dir)
    tasks = None
    if tasks_dir is not None:
        from .bench.data import load_tasks

        tasks = load_tasks(tasks_dir, list(reg.branches) if reg else None)
    return build_system(cats_cfg, registry=reg, tasks=tasks, embedder=cfg.data.embedder, seed=int(cfg.seed),
                        enrichment=_enrichment(cfg))


# --------------------------------------------------------------------- experiments
def _selected_strategies(cfg: DictConfig, s: System) -> list:
    from .bench.experiments import baseline_strategies

    strategies = baseline_strategies(s, int(cfg.strategy.k))
    wanted = cfg.strategy.baselines
    if wanted == "all":
        return strategies
    names = list(wanted)
    chosen = [st for st in strategies if any(st.name == n or st.name.startswith(n) for n in names)]
    if not chosen:
        raise ValueError(f"no strategy matches {names}; known: {[st.name for st in strategies]}")
    return chosen


def exp_bench(cfg: DictConfig, cats_cfg: CATSConfig, ctx: dict[str, Any]) -> dict:
    """Same output shape as ``run_baselines``, for the chosen strategies; records kept for the store."""
    from .bench.harness import evaluate, paired_diff, summarize

    s = build(cfg, cats_cfg)
    ctx["system"] = s
    static = s.registry.total_tokens()
    recs, out = {}, []
    for strat in _selected_strategies(cfg, s):
        r = evaluate(strat, s.test, s.registry)
        recs[strat.name] = r
        out.append(summarize(r, static))
    ctx["records"] = [x for r in recs.values() for x in r]
    paired = {n: paired_diff(recs["CATS"], r) for n, r in recs.items() if n != "CATS"} if "CATS" in recs else {}
    return {"registry": s.registry.stats(), "embedder": s.index.embedder.name,
            "calibration": s.cats.calibration_report, "summaries": out, "paired_full_cover_vs_cats": paired}


def exp_ablate(cfg: DictConfig, cats_cfg: CATSConfig, ctx: dict[str, Any]) -> list[dict]:
    from .bench.experiments import default_variants, run_variants

    return run_variants(default_variants(cats_cfg), seed=int(cfg.seed), enrichment=_enrichment(cfg),
                        registry=_registry(cfg))


def exp_validity(cfg: DictConfig, cats_cfg: CATSConfig, ctx: dict[str, Any]) -> dict:
    from .bench.experiments import coverage_validity

    seed, n = int(cfg.seed), int(cfg.experiment.n_seeds)
    return coverage_validity(cats_cfg, seeds=range(seed, seed + n), enrichment=_enrichment(cfg))


def exp_scale(cfg: DictConfig, cats_cfg: CATSConfig, ctx: dict[str, Any]) -> list[dict]:
    from .bench.experiments import scaling

    return scaling([int(x) for x in cfg.experiment.sizes], cats_cfg, seed=int(cfg.seed),
                   enrichment=_enrichment(cfg), k=int(cfg.experiment.k))


def exp_cache(cfg: DictConfig, cats_cfg: CATSConfig, ctx: dict[str, Any]) -> dict:
    from .bench.experiments import cache_sessions

    s = build(cfg, cats_cfg)
    ctx["system"] = s
    return cache_sessions(s, session_len=int(cfg.experiment.session_len),
                          n_sessions=int(cfg.experiment.n_sessions), seed=int(cfg.seed))


EXPERIMENTS: dict[str, Callable[[DictConfig, CATSConfig, dict[str, Any]], Any]] = {
    "bench": exp_bench, "ablate": exp_ablate, "validity": exp_validity, "scale": exp_scale, "cache": exp_cache,
}


# --------------------------------------------------------------------------- run
def _save(out_dir: Path, name: str, obj: Any) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.json"
    p.write_text(json.dumps(obj, indent=1, default=str))
    return p


def run(cfg: DictConfig, notes: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the configured experiment; returns the name and the paths written. ``notes`` go into
    the manifest (``main`` adds Hydra's output directory and overrides)."""
    from .manifest import run_manifest

    kind = str(cfg.experiment.kind)
    if kind not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {kind!r}; choose one of {sorted(EXPERIMENTS)}")
    name, out = run_name(cfg), results_dir(cfg)
    cats_cfg = cats_config(cfg)
    plain = OmegaConf.to_container(cfg, resolve=True)
    data_paths: list[Path | None] = [_path(cfg.data.corpus_dir) or DATA / "corpus",
                                     _path(cfg.data.tasks_dir) or DATA / "tasks"]
    for flag, key in (("enrich", "enrich_dir"), ("jev", "jev_dir"), ("llm_router", "llm_router_dir")):
        if cfg.data[flag]:
            data_paths.append(_path(cfg.data[key]))
    if kind == "scale":
        data_paths.append(DATA / "distractors")
    ctx: dict[str, Any] = {}
    written: dict[str, Any] = {"name": name, "experiment": kind}
    manifest_kw: dict[str, Any] = {
        "config": {"hydra": plain, "cats": cats_cfg}, "seed": int(cfg.seed), "experiment": kind,
        "data": [p for p in data_paths if p is not None], "out_dir": out / "manifests",
        "embedder": cfg.data.embedder, "notes": notes}
    with run_manifest(name, **manifest_kw) if cfg.manifest else nullcontext() as m:
        result = EXPERIMENTS[kind](cfg, cats_cfg, ctx)
        written["result_path"] = str(_save(out, name, result))
        if m is not None:
            if "system" in ctx:
                m.set_system(ctx["system"])
            m.add_output(written["result_path"])
        if ctx.get("records") and cfg.metrics.enabled:
            from .metrics_store import MetricsStore

            store = MetricsStore(_path(cfg.metrics.root) or out / "parquet")
            part = store.append_records(ctx["records"], run_id=name, experiment=kind, seed=int(cfg.seed),
                                        config=plain)
            written["metrics_path"] = str(part)
            if m is not None and part is not None:
                m.add_output(part)
    if m is not None:
        written["manifest_path"] = str(m.path)
    return written


def compose(overrides: list[str] | None = None, config_dir: Path | None = None) -> DictConfig:
    """The composed config without launching Hydra (tests, notebooks)."""
    from hydra import compose as hydra_compose
    from hydra import initialize_config_dir

    with initialize_config_dir(config_dir=str(config_dir or CONF_DIR), version_base=None):
        return hydra_compose(config_name="config", overrides=list(overrides or []))


def main() -> None:
    import hydra

    @hydra.main(version_base=None, config_path=str(CONF_DIR), config_name="config")
    def _main(cfg: DictConfig) -> None:
        from hydra.core.hydra_config import HydraConfig
        from omegaconf.errors import MissingMandatoryValue

        hc = HydraConfig.get()
        try:
            job_num = hc.job.num
        except MissingMandatoryValue:  # only multirun jobs are numbered
            job_num = None
        notes = {"hydra_output_dir": hc.runtime.output_dir, "hydra_mode": str(hc.mode),
                 "hydra_overrides": list(hc.overrides.task), "hydra_job_num": job_num}
        written = run(cfg, notes=notes)
        print(json.dumps(written))

    _main()


if __name__ == "__main__":
    main()
