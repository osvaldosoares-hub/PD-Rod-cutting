"""Módulo de execução dos experimentos, validação de corretude e persistência."""

from __future__ import annotations

import gc
import json
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil

from src.generator import generate_instances
from src.rod_cutting import (
    rod_cutting_iterative,
    rod_cutting_memoized,
    rod_cutting_naive,
    validate_solution,
)

# Configurações experimentais padrão
EXP1_SIZES = list(range(1, 23))               # n = 1, 2, ..., 22 (22 tamanhos)
EXP1_INSTANCES = 10                           # 10 entradas distintas por tamanho
EXP1_SEED = 42

EXP2_SIZES = list(range(10, 2000, 20))        # n = 10, 30, 50, ..., 1990 (100 tamanhos)
EXP2_INSTANCES = 10                           # 10 entradas distintas por tamanho
EXP2_SEED = 2026


def collect_environment_info() -> dict[str, Any]:
    """Coleta especificações de hardware e software do ambiente experimental."""
    cpu_name = platform.processor() or "Desconhecido"
    try:
        import cpuinfo  # type: ignore

        info = cpuinfo.get_cpu_info()
        if info.get("brand_raw"):
            cpu_name = info["brand_raw"]
    except Exception:
        pass

    vm = psutil.virtual_memory()
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python_version": platform.python_version(),
        "cpu": cpu_name,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(vm.total / (1024**3), 2),
    }


def _pin_to_single_core() -> bool:
    """Fixa a afinidade de CPU no núcleo 0 (P-core) para evitar migração entre núcleos."""
    try:
        proc = psutil.Process()
        if hasattr(proc, "cpu_affinity"):
            proc.cpu_affinity([0])
            return True
    except Exception:
        pass
    return False


def _warmup() -> None:
    """Executa um breve aquecimento do interpretador antes das medições."""
    sample = list(range(1, 16))
    # O algoritmo ingênuo retorna apenas um inteiro (lucro máximo), sem tupla de cortes
    _ = rod_cutting_naive(sample, 10)
    rod_cutting_memoized(sample, 15)
    rod_cutting_iterative(sample, 15)


def run_experiment_1(
    sizes: list[int] = EXP1_SIZES,
    num_instances: int = EXP1_INSTANCES,
    seed: int = EXP1_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Experimento 1: compara Ingênuo O(2^n), PD Memoizada O(n^2) e PD Iterativa O(n^2)."""
    dataset = generate_instances(sizes, num_instances, seed=seed)
    raw_rows: list[dict[str, Any]] = []
    verified_count = 0

    algorithms = [
        ("naive", rod_cutting_naive),
        ("memoized", rod_cutting_memoized),
        ("iterative", rod_cutting_iterative),
    ]

    for n in sizes:
        for inst_idx, prices in enumerate(dataset[n]):
            # Alterna a ordem de execução para mitigar viés de cache/estado do interpretador
            shift = inst_idx % len(algorithms)
            ordered = algorithms[shift:] + algorithms[:shift]

            outputs: dict[str, Any] = {}
            times_ms: dict[str, float] = {}
            for name, fn in ordered:
                gc.collect()
                t0 = time.perf_counter()
                out = fn(prices, n)
                t1 = time.perf_counter()
                outputs[name] = out
                times_ms[name] = (t1 - t0) * 1000.0

            # Ingênuo retorna apenas int (lucro); memoized e iterative retornam (lucro, cortes)
            rev_naive: int = outputs["naive"]
            rev_memo, cuts_memo = outputs["memoized"]
            rev_iter, cuts_iter = outputs["iterative"]
            t_naive = times_ms["naive"]
            t_memo = times_ms["memoized"]
            t_iter = times_ms["iterative"]

            # Verificação rigorosa de corretude em TODAS as entradas
            if not (rev_naive == rev_memo == rev_iter):
                raise AssertionError(
                    f"Divergência em Exp1 (n={n}, inst={inst_idx}): "
                    f"naive={rev_naive}, memo={rev_memo}, iter={rev_iter}"
                )
            # Validação dos cortes apenas nos algoritmos que retornam (lucro, cortes)
            validate_solution(prices, n, rev_memo, cuts_memo)
            validate_solution(prices, n, rev_iter, cuts_iter)
            verified_count += 1

            raw_rows.append(
                {
                    "n": n,
                    "instance": inst_idx,
                    "optimal_revenue": rev_iter,
                    "num_cuts_iter": len(cuts_iter),
                    "naive_ms": t_naive,
                    "memoized_ms": t_memo,
                    "iterative_ms": t_iter,
                }
            )
        print(
            f"[Exp1] n={n:2d} concluído "
            f"(média ingênuo: {np.mean([r['naive_ms'] for r in raw_rows if r['n'] == n]):.3f} ms)"
        )

    raw_df = pd.DataFrame(raw_rows)
    agg_df = (
        raw_df.groupby("n", as_index=False)
        .agg(
            naive_mean_ms=("naive_ms", "mean"),
            naive_std_ms=("naive_ms", "std"),
            naive_min_ms=("naive_ms", "min"),
            naive_max_ms=("naive_ms", "max"),
            memoized_mean_ms=("memoized_ms", "mean"),
            memoized_std_ms=("memoized_ms", "std"),
            iterative_mean_ms=("iterative_ms", "mean"),
            iterative_std_ms=("iterative_ms", "std"),
        )
        .sort_values("n")
        .reset_index(drop=True)
    )
    agg_df["speedup_memo_vs_naive"] = (
        agg_df["naive_mean_ms"] / agg_df["memoized_mean_ms"]
    )
    agg_df["speedup_iter_vs_naive"] = (
        agg_df["naive_mean_ms"] / agg_df["iterative_mean_ms"]
    )

    # Ajuste exponencial em log2(T) para n >= 10 (onde o termo exponencial domina ruído fixo)
    fit_df = agg_df[agg_df["n"] >= 10]
    slope, intercept = np.polyfit(fit_df["n"], np.log2(fit_df["naive_mean_ms"]), 1)
    y_pred = slope * fit_df["n"] + intercept
    ss_res = float(np.sum((np.log2(fit_df["naive_mean_ms"]) - y_pred) ** 2))
    ss_tot = float(
        np.sum((np.log2(fit_df["naive_mean_ms"]) - np.mean(np.log2(fit_df["naive_mean_ms"]))) ** 2)
    )
    r2_exp = 1.0 - ss_res / ss_tot

    # Exemplo ilustrativo completo (n=10, instância não-trivial com cortes variados)
    ex_n = 10
    ex_prices = dataset[ex_n][2] if len(dataset[ex_n]) > 2 else dataset[ex_n][0]
    ex_rev_n = rod_cutting_naive(ex_prices, ex_n)
    ex_rev_m, ex_cuts_m = rod_cutting_memoized(ex_prices, ex_n)
    ex_rev_i, ex_cuts_i = rod_cutting_iterative(ex_prices, ex_n)

    stats = {
        "sizes_min": int(min(sizes)),
        "sizes_max": int(max(sizes)),
        "num_sizes": len(sizes),
        "instances_per_size": num_instances,
        "total_instances_verified": verified_count,
        "mismatches": 0,
        "max_n_naive_mean_ms": float(agg_df.iloc[-1]["naive_mean_ms"]),
        "max_n_memo_mean_ms": float(agg_df.iloc[-1]["memoized_mean_ms"]),
        "max_n_iter_mean_ms": float(agg_df.iloc[-1]["iterative_mean_ms"]),
        "max_n_speedup_iter_vs_naive": float(agg_df.iloc[-1]["speedup_iter_vs_naive"]),
        "max_n_speedup_memo_vs_naive": float(agg_df.iloc[-1]["speedup_memo_vs_naive"]),
        "naive_exp_fit_n_ge_10": {
            "base_per_step": float(2.0**slope),
            "log2_slope": float(slope),
            "log2_intercept": float(intercept),
            "r2": float(r2_exp),
        },
        "example_n10": {
            "n": ex_n,
            "prices": ex_prices,
            "naive": {"revenue": ex_rev_n, "cuts": None},
            "memoized": {"revenue": ex_rev_m, "cuts": ex_cuts_m},
            "iterative": {"revenue": ex_rev_i, "cuts": ex_cuts_i},
        },
    }
    return raw_df, agg_df, stats


def run_experiment_2(
    sizes: list[int] = EXP2_SIZES,
    num_instances: int = EXP2_INSTANCES,
    seed: int = EXP2_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Experimento 2: compara PD Memoizada O(n^2) vs PD Iterativa O(n^2) em 100 tamanhos."""
    # Garante limite de recursão suficiente para a PD memoizada mesmo se chamada diretamente
    if sizes:
        required_limit = max(10000, max(sizes) + 500)
        if sys.getrecursionlimit() < required_limit:
            sys.setrecursionlimit(required_limit)

    dataset = generate_instances(sizes, num_instances, seed=seed)
    raw_rows: list[dict[str, Any]] = []
    verified_count = 0

    for idx_n, n in enumerate(sizes, start=1):
        for inst_idx, prices in enumerate(dataset[n]):
            if inst_idx % 2 == 0:
                order = [
                    ("memoized", rod_cutting_memoized),
                    ("iterative", rod_cutting_iterative),
                ]
            else:
                order = [
                    ("iterative", rod_cutting_iterative),
                    ("memoized", rod_cutting_memoized),
                ]

            results: dict[str, tuple[int, list[int], float]] = {}
            for name, fn in order:
                t0 = time.perf_counter()
                rev, cuts = fn(prices, n)
                t1 = time.perf_counter()
                results[name] = (rev, cuts, (t1 - t0) * 1000.0)

            rev_memo, cuts_memo, t_memo = results["memoized"]
            rev_iter, cuts_iter, t_iter = results["iterative"]

            if rev_memo != rev_iter:
                raise AssertionError(
                    f"Divergência em Exp2 (n={n}, inst={inst_idx}): "
                    f"memo={rev_memo}, iter={rev_iter}"
                )
            validate_solution(prices, n, rev_memo, cuts_memo)
            validate_solution(prices, n, rev_iter, cuts_iter)
            verified_count += 1

            raw_rows.append(
                {
                    "n": n,
                    "instance": inst_idx,
                    "optimal_revenue": rev_iter,
                    "num_cuts_iter": len(cuts_iter),
                    "memoized_ms": t_memo,
                    "iterative_ms": t_iter,
                }
            )
        if idx_n % 10 == 0 or idx_n == len(sizes):
            cur_memo = np.mean([r["memoized_ms"] for r in raw_rows if r["n"] == n])
            cur_iter = np.mean([r["iterative_ms"] for r in raw_rows if r["n"] == n])
            print(
                f"[Exp2] {idx_n:3d}/{len(sizes)} (n={n:4d}): "
                f"memo={cur_memo:7.2f} ms | iter={cur_iter:7.2f} ms"
            )

    raw_df = pd.DataFrame(raw_rows)
    agg_df = (
        raw_df.groupby("n", as_index=False)
        .agg(
            memoized_mean_ms=("memoized_ms", "mean"),
            memoized_std_ms=("memoized_ms", "std"),
            memoized_median_ms=("memoized_ms", "median"),
            iterative_mean_ms=("iterative_ms", "mean"),
            iterative_std_ms=("iterative_ms", "std"),
            iterative_median_ms=("iterative_ms", "median"),
        )
        .sort_values("n")
        .reset_index(drop=True)
    )
    agg_df["ratio_memo_over_iter"] = (
        agg_df["memoized_mean_ms"] / agg_df["iterative_mean_ms"]
    )
    agg_df["memoized_ns_per_n2"] = (
        agg_df["memoized_mean_ms"] * 1e6 / (agg_df["n"] ** 2)
    )
    agg_df["iterative_ns_per_n2"] = (
        agg_df["iterative_mean_ms"] * 1e6 / (agg_df["n"] ** 2)
    )

    # Ajustes quadráticos T(n) = a*n^2 + b*n + c
    def fit_quadratic(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        coeffs = np.polyfit(x, y, 2)
        y_hat = np.polyval(coeffs, x)
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot
        return {
            "a": float(coeffs[0]),
            "b": float(coeffs[1]),
            "c": float(coeffs[2]),
            "r2": float(r2),
        }

    x_arr = agg_df["n"].to_numpy(dtype=float)
    fit_memo = fit_quadratic(x_arr, agg_df["memoized_mean_ms"].to_numpy(dtype=float))
    fit_iter = fit_quadratic(x_arr, agg_df["iterative_mean_ms"].to_numpy(dtype=float))

    large_df = agg_df[agg_df["n"] >= 500]
    ratio_ge100 = agg_df[agg_df["n"] >= 100]["ratio_memo_over_iter"]

    stats = {
        "sizes_min": int(min(sizes)),
        "sizes_max": int(max(sizes)),
        "step": int(sizes[1] - sizes[0]) if len(sizes) > 1 else 0,
        "num_sizes": len(sizes),
        "instances_per_size": num_instances,
        "total_instances_verified": verified_count,
        "mismatches": 0,
        "max_n_memo_mean_ms": float(agg_df.iloc[-1]["memoized_mean_ms"]),
        "max_n_memo_std_ms": float(agg_df.iloc[-1]["memoized_std_ms"]),
        "max_n_iter_mean_ms": float(agg_df.iloc[-1]["iterative_mean_ms"]),
        "max_n_iter_std_ms": float(agg_df.iloc[-1]["iterative_std_ms"]),
        "ratio_memo_over_iter_ge_100_mean": float(ratio_ge100.mean()),
        "ratio_memo_over_iter_ge_100_std": float(ratio_ge100.std()),
        "memo_ns_per_n2_ge_500_mean": float(large_df["memoized_ns_per_n2"].mean()),
        "iter_ns_per_n2_ge_500_mean": float(large_df["iterative_ns_per_n2"].mean()),
        "quadratic_fit_memoized": fit_memo,
        "quadratic_fit_iterative": fit_iter,
    }
    return raw_df, agg_df, stats


def run_all_benchmarks(results_dir: Path) -> dict[str, Any]:
    """Executa ambos os experimentos e grava resultados em CSV e JSON."""
    # O Experimento 2 testa n até 1990 com a versão memoizada recursiva, atingindo
    # profundidade de recursão n + 1, acima do limite padrão do Python (1000).
    sys.setrecursionlimit(10000)

    results_dir.mkdir(parents=True, exist_ok=True)
    pinned = _pin_to_single_core()
    _warmup()

    env_info = collect_environment_info()
    env_info["cpu_affinity_pinned_core0"] = pinned
    print("=== Ambiente Experimental ===")
    print(json.dumps(env_info, indent=2, ensure_ascii=False))

    t_start = time.perf_counter()
    print("\n=== Executando Experimento 1 (Ingênuo vs PD Memoizada vs PD Iterativa) ===")
    exp1_raw, exp1_agg, exp1_stats = run_experiment_1()

    print("\n=== Executando Experimento 2 (PD Memoizada vs PD Iterativa — 100 tamanhos) ===")
    exp2_raw, exp2_agg, exp2_stats = run_experiment_2()
    total_elapsed = time.perf_counter() - t_start

    exp1_raw.to_csv(results_dir / "exp1_raw.csv", index=False)
    exp1_agg.to_csv(results_dir / "exp1_all_algorithms.csv", index=False)
    exp2_raw.to_csv(results_dir / "exp2_raw.csv", index=False)
    exp2_agg.to_csv(results_dir / "exp2_dp_algorithms.csv", index=False)

    summary = {
        "environment": env_info,
        "total_benchmark_seconds": round(total_elapsed, 2),
        "total_instances_verified_all": (
            exp1_stats["total_instances_verified"]
            + exp2_stats["total_instances_verified"]
        ),
        "experiment_1": exp1_stats,
        "experiment_2": exp2_stats,
    }
    (results_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nBenchmarks finalizados em {total_elapsed:.2f} s. Resultados salvos em {results_dir}")
    return summary
