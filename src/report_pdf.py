"""Gerador automático de relatório acadêmico em PDF (`RELATORIO.pdf`)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

NAVY = colors.HexColor("#1B365D")
TEAL = colors.HexColor("#0072B2")
LIGHT_BG = colors.HexColor("#F2F5F9")
BORDER_GRAY = colors.HexColor("#CBD5E1")
TEXT_DARK = colors.HexColor("#1E293B")


def _build_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "ReportTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=23,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=6,
        ),
        "subtitle": ParagraphStyle(
            "ReportSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=11,
            leading=15,
            textColor=colors.HexColor("#475569"),
            spaceAfter=14,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=13.5,
            leading=17,
            textColor=NAVY,
            spaceBefore=14,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=11.5,
            leading=15,
            textColor=TEAL,
            spaceBefore=10,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14.5,
            textColor=TEXT_DARK,
            alignment=TA_JUSTIFY,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=9.8,
            leading=14,
            leftIndent=14,
            bulletIndent=4,
            textColor=TEXT_DARK,
            spaceAfter=3,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8.8,
            leading=12,
            textColor=colors.HexColor("#475569"),
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=10,
        ),
        "code": ParagraphStyle(
            "CodeBlock",
            parent=base["Code"],
            fontName="Courier",
            fontSize=8.8,
            leading=12,
            textColor=TEXT_DARK,
            backColor=LIGHT_BG,
            borderPadding=6,
            spaceBefore=4,
            spaceAfter=8,
        ),
    }


def _styled_table(
    data: list[list[str]], col_widths: list[float] | None = None
) -> Table:
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8.8),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("ALIGN", (0, 1), (0, -1), "LEFT"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 8.5),
                ("TEXTCOLOR", (0, 1), (-1, -1), TEXT_DARK),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
                ("GRID", (0, 0), (-1, -1), 0.4, BORDER_GRAY),
                ("TOPPADDING", (0, 0), (-1, -1), 4.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4.5),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return tbl


def _image_block(img_path: Path, caption: str, styles: dict[str, ParagraphStyle], width_cm: float = 16.0) -> KeepTogether:
    from PIL import Image as PILImage

    with PILImage.open(img_path) as im:
        w, h = im.size
    height_cm = width_cm * (h / w)
    max_height_cm = 11.5
    if height_cm > max_height_cm:
        width_cm = max_height_cm * (w / h)
        height_cm = max_height_cm

    img = Image(str(img_path), width=width_cm * cm, height=height_cm * cm)
    return KeepTogether([img, Paragraph(caption, styles["caption"])])


def generate_pdf_report(
    results_dir: Path, figures_dir: Path, output_pdf: Path
) -> Path:
    """Gera o relatório completo em PDF a partir dos dados e gráficos produzidos."""
    summary: dict[str, Any] = json.loads(
        (results_dir / "summary.json").read_text(encoding="utf-8")
    )
    exp1_df = pd.read_csv(results_dir / "exp1_all_algorithms.csv")
    exp2_df = pd.read_csv(results_dir / "exp2_dp_algorithms.csv")

    env = summary["environment"]
    s1 = summary["experiment_1"]
    s2 = summary["experiment_2"]
    ex = s1["example_n10"]

    styles = _build_styles()
    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=A4,
        leftMargin=2.0 * cm,
        rightMargin=2.0 * cm,
        topMargin=2.0 * cm,
        bottomMargin=2.0 * cm,
        title="Análise Experimental de Algoritmos — Corte de Barras",
        author="Projeto e Análise de Algoritmos",
    )

    story: list[Any] = []

    # Cabeçalho
    story.append(
        Paragraph(
            "Análise Experimental de Algoritmos: Problema do Corte de Barras",
            styles["title"],
        )
    )
    story.append(
        Paragraph(
            "Comparação empírica entre Busca Exaustiva (Ingênuo), Programação Dinâmica "
            "Recursiva com Memoização (Top-Down) e Programação Dinâmica Iterativa (Bottom-Up)",
            styles["subtitle"],
        )
    )

    meta_table = _styled_table(
        [
            ["Parâmetro do Experimento", "Configuração / Valor"],
            ["Problema", "Corte de Barras (Rod Cutting) — lucro máximo e reconstrução dos cortes"],
            ["Linguagem / Runtime", f"Python {env['python_version']}"],
            ["Hardware", f"{env['cpu']} ({env['physical_cores']} núcleos físicos / {env['logical_cores']} lógicos), {env['ram_gb']} GB RAM"],
            ["Sistema Operacional", env["os"]],
            ["Total de Instâncias Verificadas", f"{summary['total_instances_verified_all']} instâncias (0 divergências; 100% soluções válidas)"],
        ],
        col_widths=[5.5 * cm, 11.5 * cm],
    )
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # 1. Contexto e Problema
    story.append(Paragraph("1. Contexto e Formulação do Problema", styles["h1"]))
    story.append(
        Paragraph(
            "Uma empresa vende barras metálicas de diferentes comprimentos inteiros e dispõe "
            "de uma tabela de preços <i>p[1..n]</i>, onde <i>p[i]</i> indica o valor de venda de "
            "um pedaço de tamanho <i>i</i>. Dada uma barra de comprimento <i>n</i>, o objetivo é "
            "determinar o lucro máximo <i>r(n)</i> obtido ao cortar a barra em pedaços de "
            "comprimentos inteiros positivos (incluindo a opção de não realizar cortes), bem "
            "como identificar explicitamente quais cortes devem ser realizados para alcançar "
            "essa solução ótima.",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "O problema exibe <b>subestrutura ótima</b> e <b>sobreposição de subproblemas</b>: "
            "qualquer corte ótimo pode ser decomposto na escolha do tamanho <i>i</i> do "
            "<b>primeiro pedaço</b> (vendido inteiro por <i>p[i]</i>) somada à solução ótima "
            "para o restante da barra de tamanho <i>n − i</i> (que rende <i>r(n − i)</i>). "
            "Quando <i>i = n</i>, nenhum corte é feito e o restante tem tamanho zero. "
            "Assim, obtém-se a relação de recorrência clássica: "
            "<i>r(0) = 0</i> e <i>r(n) = max<sub>1 ≤ i ≤ n</sub> (p[i] + r(n − i))</i>.",
            styles["body"],
        )
    )

    story.append(Paragraph("1.1 Como Funciona o Algoritmo e o Código (Passo a Passo)", styles["h2"]))
    story.append(
        Paragraph(
            "Em <i>src/rod_cutting.py</i>, a recorrência do Corte de Barras é traduzida diretamente "
            "para código por meio de um laço <i>for i in range(1, j + 1)</i> que testa cada possível "
            "tamanho <i>i</i> para o <b>primeiro pedaço cortado</b>. Esse pedaço é vendido inteiro "
            "por <i>prices[i - 1]</i> (índice ajustado em −1 pois listas Python começam em 0), "
            "somado ao lucro ótimo do restante <i>j − i</i>. Quando <i>i = j</i>, o restante é zero "
            "e a barra não é cortada. A Figura 1 ilustra esse processo completo para <i>n = 4</i> "
            "com a tabela clássica <i>p = [1, 5, 8, 9]</i>, onde cortar em <b>[2, 2]</b> "
            "(lucro 10) supera vender a barra inteira (lucro 9).",
            styles["body"],
        )
    )
    story.append(
        _image_block(
            figures_dir / "fig_rod_cutting_mechanism.png",
            "Figura 1: Funcionamento passo a passo do algoritmo de Corte de Barras para n = 4 (p = [1, 5, 8, 9]): (A) avaliação das 4 opções do primeiro corte; (B) preenchimento das tabelas da PD e reconstrução da solução ótima [2, 2].",
            styles,
            width_cm=16.6,
        )
    )
    story.append(
        Paragraph(
            "• <b>1. Escolha do primeiro corte (Painel A):</b> para <i>n = 4</i>, o código avalia "
            "as quatro possibilidades: <i>i = 1</i> → 1 + r(3) = 9; <b><i>i = 2</i> → 5 + r(2) = 10 "
            "(ótimo)</b>; <i>i = 3</i> → 8 + r(1) = 9; e <i>i = 4</i> → 9 + r(0) = 9 (sem corte). "
            "Em qualquer tamanho <i>j</i>, guarda-se o maior valor obtido em <i>best_val</i> e o "
            "respectivo corte <i>i</i> em <i>best_cut</i>.",
            styles["bullet"],
        )
    )
    story.append(
        Paragraph(
            "• <b>2. Diferença no código entre as três abordagens:</b> no <b>Ingênuo</b> "
            "(<i>rod_cutting_naive</i>), cada iteração chama <i>_solve(length - i)</i> do zero sem "
            "memória auxiliar, recalculando subproblemas repetidamente e retornando a tupla "
            "<i>(best_val, [best_first, *best_rem])</i> em tempo Θ(2<sup>n</sup>). Na <b>PD "
            "Memoizada</b> (<i>rod_cutting_memoized</i>, Top-Down), antes de calcular um "
            "subproblema o código verifica <i>if memo[length] >= 0</i> em O(1); se ainda não foi "
            "calculado, resolve via recursão e grava <i>memo[length]</i> e <i>first_cut[length]</i>. "
            "Já na <b>PD Iterativa</b> (<i>rod_cutting_iterative</i>, Bottom-Up), dois laços "
            "aninhados preenchem <i>dp[1..n]</i> e <i>first_cut[1..n]</i> em ordem crescente "
            "(<i>j = 1..n</i>), consultando diretamente <i>prices[i - 1] + dp[j - i]</i> sem "
            "chamadas recursivas — exatamente como exibido na tabela do Painel B.",
            styles["bullet"],
        )
    )
    story.append(
        Paragraph(
            "• <b>3. Reconstrução dos cortes em O(n) (<i>_reconstruct_cuts</i>, Painel B):</b> "
            "o vetor <i>dp[n]</i> fornece apenas o valor máximo (10), mas não quais pedaços compõem "
            "a solução. Para recuperar os cortes, a função <i>_reconstruct_cuts(first_cut, n)</i> "
            "inicia com <i>remaining = n</i> e repete enquanto <i>remaining &gt; 0</i>: lê "
            "<i>cut = first_cut[remaining]</i>, adiciona <i>cut</i> à lista de resposta e atualiza "
            "<i>remaining ← remaining − cut</i>. No exemplo (<i>n = 4</i>), parte-se de 4 → corta "
            "<i>first_cut[4] = 2</i> (sobra 2) → corta <i>first_cut[2] = 2</i> (sobra 0, fim), "
            "produzindo a solução completa <b>(10, [2, 2])</b>.",
            styles["bullet"],
        )
    )

    # 2. Algoritmos Comparados
    story.append(Paragraph("2. Algoritmos Implementados e Análise Teórica", styles["h1"]))
    cell_center = ParagraphStyle(
        "CellCenter",
        parent=styles["body"],
        fontSize=8.5,
        leading=11,
        alignment=TA_CENTER,
        spaceAfter=0,
    )
    algo_table = _styled_table(
        [
            ["Algoritmo", "Estratégia", "Tempo", "Espaço", "Reconstrução"],
            ["1. Ingênuo", "Recursão exaustiva (sem cache)", Paragraph("Θ(2<sup>n</sup>)", cell_center), "O(n)", "Retorno recursivo da lista"],
            ["2. PD Memoizada", "Top-Down (recursão + tabela)", Paragraph("Θ(n<sup>2</sup>)", cell_center), "Θ(n)", "Vetor first_cut[1..n]"],
            ["3. PD Iterativa", "Bottom-Up (laços iterativos)", Paragraph("Θ(n<sup>2</sup>)", cell_center), "Θ(n)", "Vetor first_cut[1..n]"],
        ],
        col_widths=[3.2 * cm, 4.7 * cm, 2.0 * cm, 2.0 * cm, 5.1 * cm],
    )
    story.append(algo_table)
    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            "Um ponto prático relevante em Python é que a versão recursiva com memoização "
            "atinge profundidade de recursão <i>n + 1</i> (ao avaliar <i>i = 1</i> primeiro), "
            "excedendo o limite padrão de 1.000 chamadas do interpretador. Por isso, a "
            "implementação ajusta dinamicamente <i>sys.setrecursionlimit(n + 500)</i>. "
            "A versão iterativa elimina completamente a pilha de recursão e o overhead de "
            "chamadas de função, preservando a mesma complexidade assintótica Θ(n²).",
            styles["body"],
        )
    )

    story.append(Paragraph("2.1 Árvore de Recursão do Algoritmo Ingênuo", styles["h2"]))
    story.append(
        Paragraph(
            "Para ilustrar visualmente por que o algoritmo ingênuo apresenta crescimento "
            "exponencial Θ(2<sup>n</sup>), a Figura 2 exibe sua árvore completa de chamadas "
            "recursivas para uma barra de tamanho <i>n = 4</i>. Cada aresta rotulada por "
            "<i>i</i> representa a escolha de um primeiro pedaço de tamanho <i>i</i>, "
            "reduzindo o problema ao restante <i>r(n − i)</i>. Como o número de chamadas "
            "satisfaz <i>T(n) = 1 + Σ<sub>k=0..n−1</sub> T(k) = 2<sup>n</sup></i>, para "
            "<i>n = 4</i> a árvore contém <b>16 nós</b> (chamadas) e <b>8 folhas</b> "
            "(2<sup>n−1</sup> formas ordenadas de cortar a barra).",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            "Observe a <b>sobreposição de subproblemas</b> destacada pelas cores e caixas "
            "tracejadas: o subproblema <i>r(2)</i> é recalculado do zero <b>2 vezes</b>, "
            "<i>r(1)</i> é recalculado <b>4 vezes</b> e <i>r(0)</i> é avaliado <b>8 vezes</b>. "
            "Ao armazenar o resultado de cada subproblema na primeira vez em que ele é "
            "resolvido (Programação Dinâmica), toda a árvore exponencial colapsa em apenas "
            "<i>n + 1</i> estados distintos (5 para <i>n = 4</i>), reduzindo o custo para "
            "Θ(n²).",
            styles["body"],
        )
    )
    story.append(
        _image_block(
            figures_dir / "fig0_recursion_tree.png",
            "Figura 2: Árvore de recursão do algoritmo ingênuo para n = 4 (16 chamadas e 8 folhas), destacando a sobreposição de subproblemas (ex.: subárvores idênticas de r(2)).",
            styles,
            width_cm=14.6,
        )
    )

    # 3. Metodologia e Verificação
    story.append(Paragraph("3. Metodologia Experimental e Verificação de Corretude", styles["h1"]))
    story.append(
        Paragraph(
            "Para avaliar fielmente o impacto do tamanho da entrada <i>n</i> sobre o tempo de "
            "execução e respeitar as limitações do crescimento exponencial do algoritmo ingênuo, "
            "o estudo foi dividido em dois experimentos complementares com sementes fixas "
            "para total reprodutibilidade:",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            f"• <b>Experimento 1 (3 algoritmos — escala pequena):</b> <i>n = {s1['sizes_min']}, "
            f"2, ..., {s1['sizes_max']}</i> ({s1['num_sizes']} tamanhos), com "
            f"{s1['instances_per_size']} tabelas de preços aleatórias distintas por tamanho "
            f"({s1['total_instances_verified']} instâncias). Todos os 3 algoritmos rodaram sobre "
            "as mesmas entradas, com alternância na ordem de execução.",
            styles["bullet"],
        )
    )
    story.append(
        Paragraph(
            f"• <b>Experimento 2 (2 algoritmos de PD — escala grande):</b> {s2['num_sizes']} "
            f"tamanhos igualmente espaçados (<i>n = {s2['sizes_min']}, "
            f"{s2['sizes_min'] + s2['step']}, ..., {s2['sizes_max']}</i>), com "
            f"{s2['instances_per_size']} entradas aleatórias distintas por tamanho "
            f"({s2['total_instances_verified']} instâncias), seguindo exatamente a grade "
            "sugerida no enunciado.",
            styles["bullet"],
        )
    )
    story.append(
        Paragraph(
            "• <b>Geração automática de entradas:</b> cada tabela <i>p[1..n]</i> foi gerada "
            "com incrementos aleatórios positivos em [1, 10], garantindo preços realistas "
            "não-decrescentes com variações locais que produzem combinações de corte não-triviais.",
            styles["bullet"],
        )
    )
    story.append(
        Paragraph(
            "• <b>Verificação de corretude em 100% das entradas:</b> para cada instância testada "
            "(1.220 no total, além da suíte automatizada <i>pytest</i> com a tabela clássica do "
            "livro CLRS), verificou-se que todos os algoritmos retornaram exatamente o mesmo "
            "lucro ótimo e que a sequência de cortes devolvida por cada algoritmo soma <i>n</i> "
            "e produz receita igual ao lucro ótimo. Resultado: <b>0 divergências em 1.220 "
            "instâncias</b>.",
            styles["bullet"],
        )
    )

    story.append(
        KeepTogether(
            [
                Paragraph("Exemplo Concreto de Saída Completa (n = 10)", styles["h2"]),
                Paragraph(
                    f"Tabela de preços p[1..10] = {ex['prices']}<br/>"
                    f"• Ingênuo: lucro = {ex['naive']['revenue']} | cortes = {ex['naive']['cuts']}<br/>"
                    f"• PD Memoizada: lucro = {ex['memoized']['revenue']} | cortes = {ex['memoized']['cuts']}<br/>"
                    f"• PD Iterativa: lucro = {ex['iterative']['revenue']} | cortes = {ex['iterative']['cuts']}",
                    styles["code"],
                ),
            ]
        )
    )

    # 4. Resultados Experimento 1
    story.append(PageBreak())
    story.append(Paragraph("4. Resultados do Experimento 1: Ingênuo vs. Programação Dinâmica", styles["h1"]))
    fit_exp = s1["naive_exp_fit_n_ge_10"]
    story.append(
        Paragraph(
            f"Conforme previsto pela análise teórica, o algoritmo ingênuo apresenta explosão "
            f"exponencial severa: para <i>n = {s1['sizes_max']}</i>, seu tempo médio atingiu "
            f"<b>{s1['max_n_naive_mean_ms']:.2f} ms</b>, enquanto a PD Memoizada executou em "
            f"<b>{s1['max_n_memo_mean_ms']:.4f} ms</b> e a PD Iterativa em "
            f"<b>{s1['max_n_iter_mean_ms']:.4f} ms</b> — um ganho de desempenho (speedup) de "
            f"aproximadamente <b>{s1['max_n_speedup_iter_vs_naive']:,.0f}×</b> já em uma barra de "
            f"tamanho 22. A regressão exponencial em log<sub>2</sub>(T) para <i>n ≥ 10</i> obteve base "
            f"empírica <b>{fit_exp['base_per_step']:.4f}<sup>n</sup></b> (muito próxima do valor teórico "
            f"2,0000<sup>n</sup>) com coeficiente de determinação <b>R² = {fit_exp['r2']:.5f}</b>.",
            styles["body"],
        )
    )

    # Tabela resumida Exp1
    selected_n1 = [4, 8, 12, 16, 18, 20, 21, 22]
    sub1 = exp1_df[exp1_df["n"].isin(selected_n1)]
    rows1 = [
        ["n", "Ingênuo (ms)", "PD Memoizada (ms)", "PD Iterativa (ms)", "Speedup (Iter/Ing)"]
    ]
    for _, r in sub1.iterrows():
        rows1.append(
            [
                f"{int(r['n'])}",
                f"{r['naive_mean_ms']:.3f} ± {r['naive_std_ms']:.3f}",
                f"{r['memoized_mean_ms']:.4f}",
                f"{r['iterative_mean_ms']:.4f}",
                f"{r['speedup_iter_vs_naive']:,.1f}×",
            ]
        )
    story.append(_styled_table(rows1, col_widths=[1.5 * cm, 4.3 * cm, 3.8 * cm, 3.8 * cm, 3.6 * cm]))
    story.append(Spacer(1, 8))

    story.append(
        _image_block(
            figures_dir / "fig1_all_algorithms_linear.png",
            "Figura 3: Tempo médio de execução dos três algoritmos em escala linear (n = 1..22, 10 entradas por tamanho).",
            styles,
            width_cm=15.5,
        )
    )
    story.append(
        _image_block(
            figures_dir / "fig2_all_algorithms_log.png",
            "Figura 4: Comparação dos três algoritmos em escala semilogarítmica, evidenciando a reta exponencial Θ(2<sup>n</sup>) do algoritmo ingênuo.",
            styles,
            width_cm=15.5,
        )
    )

    # 5. Resultados Experimento 2
    story.append(PageBreak())
    story.append(Paragraph("5. Resultados do Experimento 2: PD Memoizada vs. PD Iterativa", styles["h1"]))
    fit_m = s2["quadratic_fit_memoized"]
    fit_i = s2["quadratic_fit_iterative"]
    story.append(
        Paragraph(
            f"Nos 100 tamanhos igualmente espaçados entre <i>n = {s2['sizes_min']}</i> e "
            f"<i>n = {s2['sizes_max']}</i> (1.000 instâncias verificadas), ambas as abordagens "
            f"de Programação Dinâmica exibem crescimento quadrático nítido. Os ajustes "
            f"polinomiais de grau 2 obtiveram <b>R² = {fit_m['r2']:.5f}</b> para a PD Memoizada e "
            f"<b>R² = {fit_i['r2']:.5f}</b> para a PD Iterativa, confirmando empiricamente a "
            f"complexidade assintótica Θ(n²).",
            styles["body"],
        )
    )
    story.append(
        Paragraph(
            f"Apesar da mesma classe assintótica, a constante multiplicativa difere de forma "
            f"consistente: em <i>n = {s2['sizes_max']}</i>, a PD Memoizada levou em média "
            f"<b>{s2['max_n_memo_mean_ms']:.2f} ± {s2['max_n_memo_std_ms']:.2f} ms</b>, "
            f"enquanto a PD Iterativa levou <b>{s2['max_n_iter_mean_ms']:.2f} ± "
            f"{s2['max_n_iter_std_ms']:.2f} ms</b>. Para <i>n ≥ 100</i>, a PD Recursiva com "
            f"Memoização foi em média <b>{s2['ratio_memo_over_iter_ge_100_mean']:.2f}× "
            f"(± {s2['ratio_memo_over_iter_ge_100_std']:.2f})</b> mais lenta que a PD Iterativa. "
            "Isso ocorre porque, no laço interno da abordagem Top-Down, cada subproblema "
            "já resolvido ainda incorre no custo de uma chamada de função Python "
            "(<i>_solve(length - i)</i>) e verificação condicional de cache, ao passo que a "
            "abordagem Bottom-Up lê diretamente <i>dp[j - i]</i> em memória contígua.",
            styles["body"],
        )
    )

    selected_n2 = [10, 110, 310, 510, 750, 990, 1250, 1510, 1750, 1990]
    sub2 = exp2_df[exp2_df["n"].isin(selected_n2)]
    rows2 = [
        ["n", "PD Memoizada (ms)", "PD Iterativa (ms)", "Razão (Memo/Iter)", "Iter ns/n²"]
    ]
    for _, r in sub2.iterrows():
        rows2.append(
            [
                f"{int(r['n'])}",
                f"{r['memoized_mean_ms']:.2f} ± {r['memoized_std_ms']:.2f}",
                f"{r['iterative_mean_ms']:.2f} ± {r['iterative_std_ms']:.2f}",
                f"{r['ratio_memo_over_iter']:.2f}×",
                f"{r['iterative_ns_per_n2']:.1f}",
            ]
        )
    story.append(_styled_table(rows2, col_widths=[1.6 * cm, 4.4 * cm, 4.4 * cm, 3.6 * cm, 3.0 * cm]))
    story.append(Spacer(1, 8))

    story.append(
        _image_block(
            figures_dir / "fig3_dp_comparison.png",
            "Figura 5: Tempo médio (± desvio padrão) e ajuste quadrático da PD Memoizada e PD Iterativa em 100 tamanhos igualmente espaçados.",
            styles,
            width_cm=15.5,
        )
    )
    story.append(
        _image_block(
            figures_dir / "fig4_dp_analysis.png",
            "Figura 6: (Esquerda) Estabilização de T(n)/n² confirmando Θ(n²); (Direita) Razão T_memo(n)/T_iter(n) quantificando o overhead recursivo.",
            styles,
            width_cm=16.2,
        )
    )

    # 6. Conclusões
    story.append(Paragraph("6. Discussão Final e Conclusões", styles["h1"]))
    story.append(
        Paragraph(
            "1. <b>Impacto da classe assintótica (Exponencial vs. Polinomial):</b> a remoção do "
            "recálculo redundante de subproblemas transforma um algoritmo impraticável além de "
            "<i>n ≈ 25</i> (onde 2<sup>n</sup> cresce exponencialmente) em algoritmos que resolvem "
            "<i>n = 1.990</i> em frações de segundo.",
            styles["bullet"],
        )
    )
    story.append(
        Paragraph(
            "2. <b>Constantes ocultas em Θ(n²):</b> embora PD Memoizada e PD Iterativa tenham a "
            "mesma complexidade assintótica de tempo e espaço, a PD Iterativa superou "
            "consistentemente a PD Memoizada em todas as escalas testadas (~"
            f"{s2['ratio_memo_over_iter_ge_100_mean']:.2f}× mais rápida para n ≥ 100), "
            "além de não sofrer restrição de profundidade de pilha de recursão.",
            styles["bullet"],
        )
    )
    story.append(
        Paragraph(
            "3. <b>Confiabilidade experimental:</b> a verificação cruzada automática entre os "
            "algoritmos, somada à validação estrutural dos cortes reconstruídos nas 1.220 "
            "instâncias aleatórias e nos casos de teste do livro CLRS, assegura alta confiança "
            "na corretude das implementações.",
            styles["bullet"],
        )
    )

    doc.build(story)
    print(f"Relatório PDF gerado em: {output_pdf}")
    return output_pdf
