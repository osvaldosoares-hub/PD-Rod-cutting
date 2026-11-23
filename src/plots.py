"""Geração dos gráficos comparativos do experimento de Corte de Barras."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Paleta amigável para daltonismo (Okabe-Ito / acadêmica)
COLOR_NAIVE = "#D55E00"      # Vermelho/laranja forte
COLOR_MEMO = "#0072B2"       # Azul profundo
COLOR_ITER = "#009E73"       # Verde esmeralda
COLOR_ACCENT = "#555555"     # Cinza escuro


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.titlesize": 12.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linestyle": "--",
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_exp1_linear(
    exp1_df: pd.DataFrame, summary: dict[str, Any], out_path: Path
) -> None:
    """Figura 1: Comparação dos 3 algoritmos em escala linear (n = 1..22)."""
    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    n = exp1_df["n"].to_numpy()
    ax.plot(
        n,
        exp1_df["naive_mean_ms"],
        marker="o",
        markersize=5,
        linewidth=2.2,
        color=COLOR_NAIVE,
        label="Ingênuo — O(2ⁿ)",
    )
    ax.fill_between(
        n,
        np.maximum(0, exp1_df["naive_mean_ms"] - exp1_df["naive_std_ms"]),
        exp1_df["naive_mean_ms"] + exp1_df["naive_std_ms"],
        color=COLOR_NAIVE,
        alpha=0.16,
    )

    ax.plot(
        n,
        exp1_df["memoized_mean_ms"],
        marker="s",
        markersize=4.5,
        linewidth=2.0,
        color=COLOR_MEMO,
        label="PD Recursiva c/ Memoização — O(n²)",
    )
    ax.plot(
        n,
        exp1_df["iterative_mean_ms"],
        marker="^",
        markersize=4.5,
        linewidth=2.0,
        color=COLOR_ITER,
        label="PD Iterativa (Bottom-Up) — O(n²)",
    )

    last = exp1_df.iloc[-1]
    speedup = last["speedup_iter_vs_naive"]
    ax.annotate(
        f"n = {int(last['n'])}\n"
        f"Ingênuo: {last['naive_mean_ms']:.1f} ms\n"
        f"PD Iter.: {last['iterative_mean_ms']:.4f} ms\n"
        f"Speedup ≈ {speedup:,.0f}×",
        xy=(last["n"], last["naive_mean_ms"]),
        xytext=(last["n"] - 7.5, last["naive_mean_ms"] * 0.70),
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=COLOR_NAIVE, lw=1.2),
        arrowprops=dict(arrowstyle="->", color=COLOR_NAIVE, lw=1.3),
    )

    ax.set_title("Corte de Barras — Experimento 1: Ingênuo vs. Programação Dinâmica (Escala Linear)")
    ax.set_xlabel("Tamanho da barra (n)")
    ax.set_ylabel("Tempo médio de execução (ms)")
    ax.set_xticks(range(2, int(n.max()) + 1, 2))
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_exp1_log(
    exp1_df: pd.DataFrame, summary: dict[str, Any], out_path: Path
) -> None:
    """Figura 2: Comparação dos 3 algoritmos em escala logarítmica no eixo Y."""
    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    n = exp1_df["n"].to_numpy()
    ax.semilogy(
        n,
        exp1_df["naive_mean_ms"],
        marker="o",
        markersize=5,
        linewidth=2.2,
        color=COLOR_NAIVE,
        label="Ingênuo — O(2ⁿ)",
    )

    # Curva exponencial ajustada para n >= 10
    fit = summary["experiment_1"]["naive_exp_fit_n_ge_10"]
    n_fit = np.linspace(10, float(n.max()), 100)
    y_fit = 2.0 ** (fit["log2_slope"] * n_fit + fit["log2_intercept"])
    ax.semilogy(
        n_fit,
        y_fit,
        linestyle="--",
        linewidth=1.5,
        color="#7A2E00",
        label=(
            f"Ajuste exponencial (n≥10): base ≈ {fit['base_per_step']:.3f}ⁿ "
            f"(R² = {fit['r2']:.5f})"
        ),
    )

    ax.semilogy(
        n,
        exp1_df["memoized_mean_ms"],
        marker="s",
        markersize=4.5,
        linewidth=2.0,
        color=COLOR_MEMO,
        label="PD Recursiva c/ Memoização — O(n²)",
    )
    ax.semilogy(
        n,
        exp1_df["iterative_mean_ms"],
        marker="^",
        markersize=4.5,
        linewidth=2.0,
        color=COLOR_ITER,
        label="PD Iterativa (Bottom-Up) — O(n²)",
    )

    ax.set_title("Corte de Barras — Experimento 1: Comparação em Escala Logarítmica (Semilog)")
    ax.set_xlabel("Tamanho da barra (n)")
    ax.set_ylabel("Tempo médio de execução (ms) — escala log₁₀")
    ax.set_xticks(range(2, int(n.max()) + 1, 2))
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_exp2_dp(
    exp2_df: pd.DataFrame, summary: dict[str, Any], out_path: Path
) -> None:
    """Figura 3: PD Recursiva com Memoização vs PD Iterativa (100 tamanhos)."""
    fig, ax = plt.subplots(figsize=(9.6, 5.6))

    n = exp2_df["n"].to_numpy()
    fit_m = summary["experiment_2"]["quadratic_fit_memoized"]
    fit_i = summary["experiment_2"]["quadratic_fit_iterative"]

    # Curvas experimentais + banda ±1 desvio padrão
    ax.plot(
        n,
        exp2_df["memoized_mean_ms"],
        linewidth=2.2,
        color=COLOR_MEMO,
        label="PD Recursiva c/ Memoização (média ± dp)",
    )
    ax.fill_between(
        n,
        np.maximum(0, exp2_df["memoized_mean_ms"] - exp2_df["memoized_std_ms"]),
        exp2_df["memoized_mean_ms"] + exp2_df["memoized_std_ms"],
        color=COLOR_MEMO,
        alpha=0.16,
    )

    ax.plot(
        n,
        exp2_df["iterative_mean_ms"],
        linewidth=2.2,
        color=COLOR_ITER,
        label="PD Iterativa Bottom-Up (média ± dp)",
    )
    ax.fill_between(
        n,
        np.maximum(0, exp2_df["iterative_mean_ms"] - exp2_df["iterative_std_ms"]),
        exp2_df["iterative_mean_ms"] + exp2_df["iterative_std_ms"],
        color=COLOR_ITER,
        alpha=0.18,
    )

    # Ajustes quadráticos
    n_smooth = np.linspace(float(n.min()), float(n.max()), 300)
    y_m_fit = fit_m["a"] * n_smooth**2 + fit_m["b"] * n_smooth + fit_m["c"]
    y_i_fit = fit_i["a"] * n_smooth**2 + fit_i["b"] * n_smooth + fit_i["c"]

    ax.plot(
        n_smooth,
        y_m_fit,
        linestyle="--",
        linewidth=1.4,
        color="#003E63",
        label=f"Ajuste quadrático Memoizada (R² = {fit_m['r2']:.5f})",
    )
    ax.plot(
        n_smooth,
        y_i_fit,
        linestyle="--",
        linewidth=1.4,
        color="#00573F",
        label=f"Ajuste quadrático Iterativa (R² = {fit_i['r2']:.5f})",
    )

    last = exp2_df.iloc[-1]
    ax.annotate(
        f"n = {int(last['n'])}\n"
        f"Memoizada: {last['memoized_mean_ms']:.1f} ms\n"
        f"Iterativa: {last['iterative_mean_ms']:.1f} ms\n"
        f"Razão: {last['ratio_memo_over_iter']:.2f}×",
        xy=(last["n"], last["memoized_mean_ms"]),
        xytext=(last["n"] - 930, last["memoized_mean_ms"] * 0.87),
        fontsize=9.5,
        bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=COLOR_MEMO, lw=1.2),
        arrowprops=dict(arrowstyle="->", color=COLOR_MEMO, lw=1.3),
    )

    ax.set_title(
        "Corte de Barras — Experimento 2: PD Memoizada vs. PD Iterativa "
        "(100 tamanhos, 10 entradas/tamanho)"
    )
    ax.set_xlabel("Tamanho da barra (n)")
    ax.set_ylabel("Tempo médio de execução (ms)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_exp2_analysis(
    exp2_df: pd.DataFrame, summary: dict[str, Any], out_path: Path
) -> None:
    """Figura 4: Análise de constante assintótica T(n)/n² e razão Memoizada/Iterativa."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.8))

    df_ge100 = exp2_df[exp2_df["n"] >= 100]
    n = df_ge100["n"].to_numpy()

    # Subgráfico 1: T(n) / n^2 (ns por unidade de n^2)
    ax1.plot(
        n,
        df_ge100["memoized_ns_per_n2"],
        linewidth=2.0,
        color=COLOR_MEMO,
        label="PD Memoizada: T(n) / n²",
    )
    ax1.plot(
        n,
        df_ge100["iterative_ns_per_n2"],
        linewidth=2.0,
        color=COLOR_ITER,
        label="PD Iterativa: T(n) / n²",
    )
    ax1.axhline(
        summary["experiment_2"]["memo_ns_per_n2_ge_500_mean"],
        linestyle="--",
        linewidth=1.2,
        color="#003E63",
        alpha=0.8,
        label=f"Média Memoizada (n≥500): {summary['experiment_2']['memo_ns_per_n2_ge_500_mean']:.1f} ns/n²",
    )
    ax1.axhline(
        summary["experiment_2"]["iter_ns_per_n2_ge_500_mean"],
        linestyle="--",
        linewidth=1.2,
        color="#00573F",
        alpha=0.8,
        label=f"Média Iterativa (n≥500): {summary['experiment_2']['iter_ns_per_n2_ge_500_mean']:.1f} ns/n²",
    )
    ax1.set_title("Tempo Normalizado por n² (n ≥ 100)")
    ax1.set_xlabel("Tamanho da barra (n)")
    ax1.set_ylabel("T(n) / n²  (ns)")
    y_max_norm = float(df_ge100["memoized_ns_per_n2"].max())
    ax1.set_ylim(15, y_max_norm * 1.28)
    ax1.legend(loc="upper left", fontsize=8.8)

    # Subgráfico 2: Razão Memoizada / Iterativa
    ratio_mean = summary["experiment_2"]["ratio_memo_over_iter_ge_100_mean"]
    ratio_std = summary["experiment_2"]["ratio_memo_over_iter_ge_100_std"]
    ax2.plot(
        n,
        df_ge100["ratio_memo_over_iter"],
        linewidth=2.0,
        color=COLOR_NAIVE,
        label="Razão T_memo(n) / T_iter(n)",
    )
    ax2.axhline(
        ratio_mean,
        linestyle="--",
        linewidth=1.5,
        color=COLOR_ACCENT,
        label=f"Média (n≥100): {ratio_mean:.2f}× (± {ratio_std:.2f})",
    )
    ax2.axhline(1.0, linestyle=":", linewidth=1.2, color="#999999")
    ax2.set_title("Overhead da Recursão: T_memo(n) / T_iter(n)")
    ax2.set_xlabel("Tamanho da barra (n)")
    ax2.set_ylabel("Razão de tempo (×)")
    ax2.set_ylim(bottom=0.8)
    ax2.legend(loc="best", fontsize=9.2)

    fig.suptitle(
        "Análise Assintótica Empírica — Confirmação de Θ(n²) e Custo de Chamadas Recursivas",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_rod_cutting_mechanism(out_path: Path) -> None:
    """Figura ilustrativa: funcionamento da recorrência, tabelas de PD e reconstrução (n=4)."""
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12.2, 5.2), gridspec_kw={"width_ratios": [1.08, 1.0]}
    )

    # ------------------------------------------------------------------
    # PAINEL ESQUERDO: 4 escolhas do 1º corte para n = 4, p = [1, 5, 8, 9]
    # ------------------------------------------------------------------
    ax1.set_title(
        "(A) Recorrência para n = 4  (p = [1, 5, 8, 9])",
        fontsize=11.2,
        fontweight="bold",
        color="#1B365D",
        pad=10,
    )

    cases = [
        # (i, p_i, rem, r_rem, total, is_opt)
        (1, 1, 3, 8, 9, False),
        (2, 5, 2, 5, 10, True),
        (3, 8, 1, 1, 9, False),
        (4, 9, 0, 0, 9, False),
    ]

    bar_h = 0.56
    y_positions = [3.15, 2.15, 1.15, 0.15]

    for (i, p_i, rem, r_rem, total, is_opt), y in zip(cases, y_positions):
        # Fundo suave destacando a linha ótima
        if is_opt:
            hl = mpatches.FancyBboxPatch(
                (-0.98, y - 0.14),
                7.68,
                bar_h + 0.28,
                boxstyle="round,pad=0.04,rounding_size=0.14",
                facecolor="#ECFDF5",
                edgecolor="#059669",
                linewidth=1.6,
                zorder=1,
            )
            ax1.add_patch(hl)

        # Rótulo à esquerda (i = ...)
        ax1.text(
            -0.52,
            y + bar_h / 2,
            f"i = {i}",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="#047857" if is_opt else "#1E293B",
            zorder=4,
        )

        # Bloco 1: primeiro pedaço de tamanho i (vendido por p[i])
        fc_first = "#059669" if is_opt else "#0072B2"
        rect1 = mpatches.FancyBboxPatch(
            (0.0, y),
            float(i) - 0.04,
            bar_h,
            boxstyle="round,pad=0.0,rounding_size=0.08",
            facecolor=fc_first,
            edgecolor="#1E293B",
            linewidth=1.1,
            zorder=3,
        )
        ax1.add_patch(rect1)
        ax1.text(
            (float(i) - 0.04) / 2.0,
            y + bar_h / 2,
            f"p[{i}] = {p_i}",
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            color="white",
            zorder=4,
        )

        # Bloco 2: pedaço restante de tamanho (4 - i) resolvido otimamente por r(4 - i)
        if rem > 0:
            # Linha tracejada indicando o ponto de corte
            ax1.plot(
                [float(i) + 0.04, float(i) + 0.04],
                [y - 0.08, y + bar_h + 0.08],
                color="#DC2626",
                linestyle="--",
                linewidth=1.5,
                zorder=5,
            )
            fc_rem = "#A7F3D0" if is_opt else "#CBD5E1"
            rect2 = mpatches.FancyBboxPatch(
                (float(i) + 0.12, y),
                float(rem) - 0.04,
                bar_h,
                boxstyle="round,pad=0.0,rounding_size=0.08",
                facecolor=fc_rem,
                edgecolor="#475569",
                linestyle="--",
                linewidth=1.1,
                zorder=3,
            )
            ax1.add_patch(rect2)
            ax1.text(
                float(i) + 0.12 + (float(rem) - 0.04) / 2.0,
                y + bar_h / 2,
                f"r({rem}) = {r_rem}",
                ha="center",
                va="center",
                fontsize=8.8,
                fontweight="bold" if is_opt else "normal",
                color="#065F46" if is_opt else "#1E293B",
                zorder=4,
            )

        # Resultado à direita
        res_x = 4.35
        if is_opt:
            ax1.text(
                res_x,
                y + bar_h / 2,
                f"→ {p_i} + {r_rem} = {total}  ★ ÓTIMO",
                ha="left",
                va="center",
                fontsize=9.4,
                fontweight="bold",
                color="#047857",
                zorder=4,
            )
        else:
            suffix = " (sem corte)" if rem == 0 else ""
            ax1.text(
                res_x,
                y + bar_h / 2,
                f"→ {p_i} + {r_rem} = {total}{suffix}",
                ha="left",
                va="center",
                fontsize=8.9,
                color="#334155",
                zorder=4,
            )

    # Legenda de cores no rodapé do painel esquerdo
    leg_patches = [
        mpatches.Patch(facecolor="#0072B2", edgecolor="#1E293B", label="1º pedaço cortado: preço p[i]"),
        mpatches.Patch(facecolor="#CBD5E1", edgecolor="#475569", linestyle="--", label="Restante: subproblema ótimo r(n − i)"),
        mpatches.Patch(facecolor="#059669", edgecolor="#1E293B", label="Escolha ótima (máxima receita)"),
    ]
    ax1.legend(
        handles=leg_patches,
        loc="lower center",
        bbox_to_anchor=(0.48, -0.02),
        ncol=1,
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=8.5,
    )

    ax1.set_xlim(-1.1, 6.85)
    ax1.set_ylim(-1.15, 4.0)
    ax1.axis("off")

    # ------------------------------------------------------------------
    # PAINEL DIREITO: Tabelas da PD (Bottom-Up) + Reconstrução dos cortes
    # ------------------------------------------------------------------
    ax2.set_title(
        "(B) Tabelas da PD (Bottom-Up) e Reconstrução dos Cortes",
        fontsize=11.2,
        fontweight="bold",
        color="#1B365D",
        pad=10,
    )

    rows_meta = [
        ("Tamanho j", ["0", "1", "2", "3", "4"], "#1B365D", "white", True),
        ("Preço p[j]", ["—", "1", "5", "8", "9"], "#F1F5F9", "#1E293B", False),
        ("Lucro dp[j]", ["0", "1", "5", "8", "10"], "#E0F2FE", "#0C4A6E", True),
        ("first_cut[j]", ["0", "1", "2", "3", "2"], "#FEF3C7", "#92400E", True),
    ]

    cell_w = 0.85
    cell_h = 0.46
    x_label = 0.0
    label_w = 1.75
    x_start = label_w + 0.08
    y_top = 3.55

    for r_idx, (r_label, vals, bg, fg, is_bold) in enumerate(rows_meta):
        y = y_top - r_idx * (cell_h + 0.06)
        # Célula de rótulo da linha
        lbl_box = mpatches.FancyBboxPatch(
            (x_label, y),
            label_w,
            cell_h,
            boxstyle="round,pad=0.0,rounding_size=0.05",
            facecolor="#1B365D" if r_idx == 0 else "#F8FAFC",
            edgecolor="#94A3B8",
            linewidth=0.9,
            zorder=3,
        )
        ax2.add_patch(lbl_box)
        ax2.text(
            x_label + label_w / 2,
            y + cell_h / 2,
            r_label,
            ha="center",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color="white" if r_idx == 0 else "#1E293B",
            zorder=4,
        )

        # Células de valores j = 0..4
        for c_idx, val in enumerate(vals):
            x = x_start + c_idx * (cell_w + 0.06)
            cell_bg = bg
            cell_fg = fg
            edge_c = "#94A3B8"
            lw = 0.9

            # Destaca dp[4] = 10 em verde e first_cut[4]=2, first_cut[2]=2 em âmbar forte
            if r_idx == 2 and c_idx == 4:
                cell_bg, cell_fg, edge_c, lw = "#059669", "white", "#065F46", 1.6
            elif r_idx == 3 and c_idx in (2, 4):
                cell_bg, cell_fg, edge_c, lw = "#F59E0B", "#1E293B", "#B45309", 1.6

            cbox = mpatches.FancyBboxPatch(
                (x, y),
                cell_w,
                cell_h,
                boxstyle="round,pad=0.0,rounding_size=0.05",
                facecolor=cell_bg,
                edgecolor=edge_c,
                linewidth=lw,
                zorder=3,
            )
            ax2.add_patch(cbox)
            ax2.text(
                x + cell_w / 2,
                y + cell_h / 2,
                val,
                ha="center",
                va="center",
                fontsize=9.2,
                fontweight="bold" if is_bold else "normal",
                color=cell_fg,
                zorder=4,
            )

    # Seta curva ilustrando o salto na reconstrução: first_cut[4]=2 -> sobra j=2 -> first_cut[2]=2 -> j=0
    y_fc_bottom = y_top - 3 * (cell_h + 0.06)
    x_c4 = x_start + 4 * (cell_w + 0.06) + cell_w / 2
    x_c2 = x_start + 2 * (cell_w + 0.06) + cell_w / 2
    x_c0 = x_start + 0 * (cell_w + 0.06) + cell_w / 2

    ax2.annotate(
        "",
        xy=(x_c2 + 0.08, y_fc_bottom - 0.04),
        xytext=(x_c4, y_fc_bottom - 0.04),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.30", color="#B45309", lw=1.7),
        zorder=5,
    )
    ax2.text(
        (x_c4 + x_c2) / 2,
        y_fc_bottom - 0.56,
        "4 − 2 = 2",
        ha="center",
        va="center",
        fontsize=8.4,
        fontweight="bold",
        color="#B45309",
    )

    ax2.annotate(
        "",
        xy=(x_c0 + 0.08, y_fc_bottom - 0.04),
        xytext=(x_c2 - 0.08, y_fc_bottom - 0.04),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.30", color="#B45309", lw=1.7),
        zorder=5,
    )
    ax2.text(
        (x_c2 + x_c0) / 2,
        y_fc_bottom - 0.56,
        "2 − 2 = 0 (fim)",
        ha="center",
        va="center",
        fontsize=8.4,
        fontweight="bold",
        color="#B45309",
    )

    # Caixa inferior com passo a passo e resultado visual final da barra cortada
    box_y = -0.95
    summary_box = mpatches.FancyBboxPatch(
        (0.0, box_y),
        6.35,
        2.08,
        boxstyle="round,pad=0.06,rounding_size=0.14",
        facecolor="#F8FAFC",
        edgecolor="#CBD5E1",
        linewidth=1.2,
        zorder=2,
    )
    ax2.add_patch(summary_box)

    ax2.text(
        0.22,
        box_y + 1.82,
        "Reconstrução via _reconstruct_cuts(first_cut, n = 4):",
        ha="left",
        va="center",
        fontsize=9.1,
        fontweight="bold",
        color="#1B365D",
        zorder=4,
    )
    ax2.text(
        0.22,
        box_y + 1.44,
        "1. remaining = 4  →  corta first_cut[4] = 2  (sobra 2)",
        ha="left",
        va="center",
        fontsize=8.6,
        color="#1E293B",
        zorder=4,
    )
    ax2.text(
        0.22,
        box_y + 1.10,
        "2. remaining = 2  →  corta first_cut[2] = 2  (sobra 0 → fim)",
        ha="left",
        va="center",
        fontsize=8.6,
        color="#1E293B",
        zorder=4,
    )

    # Barra final cortada [2] + [2] = 10
    rod_y = box_y + 0.22
    for idx_p, x_p in enumerate([0.35, 2.30]):
        pbox = mpatches.FancyBboxPatch(
            (x_p, rod_y),
            1.75,
            0.52,
            boxstyle="round,pad=0.0,rounding_size=0.08",
            facecolor="#059669",
            edgecolor="#065F46",
            linewidth=1.2,
            zorder=4,
        )
        ax2.add_patch(pbox)
        ax2.text(
            x_p + 0.875,
            rod_y + 0.26,
            "Pedaço 2  (R$ 5)",
            ha="center",
            va="center",
            fontsize=8.8,
            fontweight="bold",
            color="white",
            zorder=5,
        )
    ax2.text(2.19, rod_y + 0.26, "+", ha="center", va="center", fontsize=11, fontweight="bold", color="#065F46", zorder=5)
    ax2.text(
        4.22,
        rod_y + 0.26,
        "= Lucro 10\nCortes: [2, 2]",
        ha="left",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color="#047857",
        zorder=5,
    )

    ax2.set_xlim(-0.15, 6.55)
    ax2.set_ylim(-1.15, 4.2)
    ax2.axis("off")

    fig.suptitle(
        "Funcionamento do Algoritmo de Corte de Barras (Rod Cutting) — Exemplo Passo a Passo",
        fontsize=12.6,
        fontweight="bold",
        color="#0F172A",
        y=0.99,
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_recursion_tree(out_path: Path, n: int = 4) -> None:
    """Figura 0: Árvore de recursão do algoritmo ingênuo para n = 4 (2^4 = 16 nós)."""
    # Constrói a árvore recursiva atribuindo cada folha a uma coluna inteira [0..2^(n-1)-1]
    nodes: list[dict[str, Any]] = []
    edges: list[tuple[int, int, int]] = []  # (parent_id, child_id, cut_i)
    leaf_cursor = [0]

    def build(k: int, depth: int) -> tuple[int, float]:
        node_id = len(nodes)
        nodes.append({"id": node_id, "k": k, "depth": depth, "x": 0.0, "y": float(n - depth)})
        if k == 0:
            x = float(leaf_cursor[0])
            leaf_cursor[0] += 1
            nodes[node_id]["x"] = x
            return node_id, x

        child_xs: list[float] = []
        for i in range(1, k + 1):
            cid, cx = build(k - i, depth + 1)
            edges.append((node_id, cid, i))
            child_xs.append(cx)
        x = (child_xs[0] + child_xs[-1]) / 2.0
        nodes[node_id]["x"] = x
        return node_id, x

    build(n, 0)

    # Paleta por subproblema para destacar sobreposição de subproblemas
    node_colors = {
        4: ("#1B365D", "#FFFFFF"),  # Azul marinho (raiz — 1x)
        3: ("#0072B2", "#FFFFFF"),  # Azul profundo (1x)
        2: ("#E69F00", "#1E293B"),  # Âmbar (repetido 2x)
        1: ("#D55E00", "#FFFFFF"),  # Laranja/vermelho (repetido 4x)
        0: ("#CBD5E1", "#1E293B"),  # Cinza claro (caso base — 8x)
    }

    fig, ax = plt.subplots(figsize=(10.4, 5.8))
    ax.set_aspect("equal")

    # Destaque suave nas duas subárvores idênticas de r(2)
    r2_nodes = [nd for nd in nodes if nd["k"] == 2]
    for idx_r2, nd in enumerate(r2_nodes):
        # Coleta descendentes de nd para enquadrar a subárvore
        stack = [nd["id"]]
        sub_ids = {nd["id"]}
        while stack:
            cur = stack.pop()
            for p_id, c_id, _ in edges:
                if p_id == cur:
                    sub_ids.add(c_id)
                    stack.append(c_id)
        xs = [nodes[i]["x"] for i in sub_ids]
        ys = [nodes[i]["y"] for i in sub_ids]
        rect = mpatches.FancyBboxPatch(
            (min(xs) - 0.42, min(ys) - 0.36),
            (max(xs) - min(xs)) + 0.84,
            (max(ys) - min(ys)) + 0.72,
            boxstyle="round,pad=0.06,rounding_size=0.18",
            facecolor="#FEF3C7",
            edgecolor="#D97706",
            linestyle="--",
            linewidth=1.3,
            alpha=0.55,
            zorder=1,
        )
        ax.add_patch(rect)
        label_txt = "Subárvore r(2)\n(1ª vez)" if idx_r2 == 0 else "Subárvore r(2)\n(recalculada!)"
        ax.text(
            (min(xs) + max(xs)) / 2.0,
            min(ys) - 0.56,
            label_txt,
            ha="center",
            va="top",
            fontsize=8.2,
            fontweight="bold",
            color="#B45309",
            zorder=4,
        )

    # Desenha arestas e rótulos do corte (i)
    for p_id, c_id, cut_i in edges:
        px, py = nodes[p_id]["x"], nodes[p_id]["y"]
        cx, cy = nodes[c_id]["x"], nodes[c_id]["y"]
        ax.plot([px, cx], [py, cy], color="#64748B", linewidth=1.4, zorder=2)

        # Rótulo do tamanho do corte posicionado a 42% do caminho pai -> filho
        t = 0.44
        mx = px + t * (cx - px)
        my = py + t * (cy - py)
        # Pequeno deslocamento lateral para não sobrepor a linha
        dx = cx - px
        offset_x = -0.13 if dx < -0.1 else (0.14 if dx > 0.1 else -0.16)
        ax.text(
            mx + offset_x,
            my + 0.04,
            f"i={cut_i}",
            ha="center",
            va="center",
            fontsize=7.8,
            color="#334155",
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.82),
            zorder=3,
        )

    # Desenha os nós
    radius = 0.27
    for nd in nodes:
        fc, tc = node_colors[nd["k"]]
        circ = mpatches.Circle(
            (nd["x"], nd["y"]),
            radius=radius,
            facecolor=fc,
            edgecolor="#1E293B",
            linewidth=1.2,
            zorder=5,
        )
        ax.add_patch(circ)
        ax.text(
            nd["x"],
            nd["y"],
            f"r({nd['k']})",
            ha="center",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            color=tc,
            zorder=6,
        )

    # Legenda explicativa de sobreposição e contagem de nós
    legend_handles = [
        mpatches.Patch(facecolor="#1B365D", edgecolor="#1E293B", label="r(4): 1 chamada (raiz)"),
        mpatches.Patch(facecolor="#0072B2", edgecolor="#1E293B", label="r(3): 1 chamada"),
        mpatches.Patch(facecolor="#E69F00", edgecolor="#1E293B", label="r(2): 2 chamadas (recalculado)"),
        mpatches.Patch(facecolor="#D55E00", edgecolor="#1E293B", label="r(1): 4 chamadas (recalculado)"),
        mpatches.Patch(facecolor="#CBD5E1", edgecolor="#1E293B", label="r(0): 8 folhas (2ⁿ⁻¹ cortes)"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=True,
        facecolor="white",
        edgecolor="#CBD5E1",
        fontsize=8.8,
        title="Frequência de cada subproblema",
        title_fontsize=9.0,
    )

    # Caixa de resumo matemático no canto superior esquerdo
    ax.text(
        -0.45,
        4.32,
        "Recorrência: T(n) = 1 + Σ T(k) = 2ⁿ\n"
        "Para n = 4: 16 nós (chamadas) e 8 folhas\n"
        "Com PD/Memoização: apenas n + 1 = 5 estados!",
        ha="left",
        va="top",
        fontsize=8.8,
        color="#1E293B",
        bbox=dict(boxstyle="round,pad=0.45", fc="#F8FAFC", ec="#1B365D", lw=1.1),
        zorder=7,
    )

    ax.set_title(
        "Árvore de Recursão do Algoritmo Ingênuo para n = 4 — Sobreposição de Subproblemas (Θ(2ⁿ))",
        fontsize=11.8,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlim(-0.7, 7.7)
    ax.set_ylim(-0.95, 4.55)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(results_dir: Path, figures_dir: Path) -> list[Path]:
    """Lê os resultados salvos em `results_dir` e gera todos os gráficos em `figures_dir`."""
    _setup_style()
    figures_dir.mkdir(parents=True, exist_ok=True)

    exp1_df = pd.read_csv(results_dir / "exp1_all_algorithms.csv")
    exp2_df = pd.read_csv(results_dir / "exp2_dp_algorithms.csv")
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))

    paths = [
        figures_dir / "fig_rod_cutting_mechanism.png",
        figures_dir / "fig0_recursion_tree.png",
        figures_dir / "fig1_all_algorithms_linear.png",
        figures_dir / "fig2_all_algorithms_log.png",
        figures_dir / "fig3_dp_comparison.png",
        figures_dir / "fig4_dp_analysis.png",
    ]
    plot_rod_cutting_mechanism(paths[0])
    plot_recursion_tree(paths[1], n=4)
    plot_exp1_linear(exp1_df, summary, paths[2])
    plot_exp1_log(exp1_df, summary, paths[3])
    plot_exp2_dp(exp2_df, summary, paths[4])
    plot_exp2_analysis(exp2_df, summary, paths[5])
    print(f"Gráficos gerados em {figures_dir}: {[p.name for p in paths]}")
    return paths
