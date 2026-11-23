"""Ponto de entrada principal para executar os experimentos e gerar os gráficos."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.benchmark import run_all_benchmarks
from src.plots import generate_all_plots
from src.report_pdf import generate_pdf_report

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
PDF_PATH = ROOT / "RELATORIO.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experimento de Análise de Algoritmos — Problema do Corte de Barras"
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Regera apenas os gráficos e o relatório PDF a partir dos dados já salvos em results/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT,
        help="Diretório base de saída para results/, figures/ e RELATORIO.pdf.",
    )
    args = parser.parse_args()

    out_root = args.output_dir.resolve()
    results_dir = out_root / "results"
    figures_dir = out_root / "figures"
    pdf_path = out_root / "RELATORIO.pdf"

    if not args.plots_only:
        run_all_benchmarks(results_dir)

    generate_all_plots(results_dir, figures_dir)
    generate_pdf_report(results_dir, figures_dir, pdf_path)


if __name__ == "__main__":
    main()
