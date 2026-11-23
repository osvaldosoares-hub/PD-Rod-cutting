"""Implementações do Problema do Corte de Barras (Rod Cutting).

Cada algoritmo recebe uma tabela de preços `prices` (onde `prices[i - 1]` é o preço
de um pedaço de tamanho `i`) e o comprimento total da barra `n`:
- `rod_cutting_naive`: busca exaustiva recursiva — O(2^n) em tempo, O(n) em espaço.
  Retorna o lucro máximo (`int`).
- `rod_cutting_memoized`: programação dinâmica top-down (recursiva com memoização)
  — O(n^2) em tempo, O(n) em espaço. Implementação completa: retorna `(lucro_maximo, cortes)`.
- `rod_cutting_iterative`: programação dinâmica bottom-up (iterativa)
  — O(n^2) em tempo, O(n) em espaço. Implementação completa: retorna `(lucro_maximo, cortes)`.
"""

from __future__ import annotations

import sys
from typing import Sequence


def _validate_inputs(prices: Sequence[int], n: int) -> None:
    if n < 0:
        raise ValueError(f"O comprimento n deve ser não-negativo (recebido: {n}).")
    if n > len(prices):
        raise ValueError(
            f"A tabela de preços possui {len(prices)} entradas, mas n = {n}."
        )


def _reconstruct_cuts(first_cut: list[int], n: int) -> list[int]:
    """Reconstrói a sequência ótima de cortes a partir do vetor `first_cut`."""
    cuts: list[int] = []
    remaining = n
    while remaining > 0:
        cut = first_cut[remaining]
        cuts.append(cut)
        remaining -= cut
    return cuts


def rod_cutting_naive(prices: Sequence[int], n: int) -> int:
    """Resolve o Corte de Barras via recursão exaustiva (O(2^n))."""
    _validate_inputs(prices, n)

    def resolver(k: int) -> int:
        if k == 0:
            return 0

        lucro = -1
        for i in range(1, k + 1):
            valor = prices[i - 1] + resolver(k - i)
            if valor > lucro:
                lucro = valor

        return lucro

    return resolver(n)


def rod_cutting_memoized(prices: Sequence[int], n: int) -> tuple[int, list[int]]:
    """Resolve o Corte de Barras via Programação Dinâmica Top-Down (memoização)."""
    _validate_inputs(prices, n)

    if sys.getrecursionlimit() < n + 500:
        sys.setrecursionlimit(n + 500)

    B = [-1] * (n + 1)
    S = [0] * (n + 1)
    B[0] = 0

    def resolver(k: int) -> int:
        if B[k] == -1:
            lucro = -1

            for i in range(1, k + 1):
                valor = prices[i - 1] + resolver(k - i)

                if valor > lucro:
                    lucro = valor
                    S[k] = i

            B[k] = lucro

        return B[k]

    lucro_maximo = resolver(n)
    return lucro_maximo, _reconstruct_cuts(S, n)


def rod_cutting_iterative(prices: Sequence[int], n: int) -> tuple[int, list[int]]:
    """Resolve o Corte de Barras via Programação Dinâmica Bottom-Up (iterativa)."""
    _validate_inputs(prices, n)

    B = [0] * (n + 1)
    S = [0] * (n + 1)

    for k in range(1, n + 1):
        lucro = -1

        for i in range(1, k + 1):
            valor = prices[i - 1] + B[k - i]

            if valor > lucro:
                lucro = valor
                S[k] = i

        B[k] = lucro

    return B[n], _reconstruct_cuts(S, n)


def validate_solution(
    prices: Sequence[int], n: int, revenue: int, cuts: Sequence[int]
) -> None:
    """Valida se a solução `(revenue, cuts)` é consistente para `(prices, n)`.

    Verifica:
    1. Todos os cortes possuem tamanho válido em [1, n].
    2. A soma dos comprimentos dos cortes é exatamente igual a `n`.
    3. A soma dos preços dos cortes é exatamente igual a `revenue`.
    """
    if n == 0:
        if revenue != 0 or len(cuts) != 0:
            raise AssertionError(
                f"Para n=0 esperava-se (0, []), obtido ({revenue}, {list(cuts)})."
            )
        return

    if any(c < 1 or c > n for c in cuts):
        raise AssertionError(f"Cortes fora do intervalo [1, {n}]: {list(cuts)}")

    total_len = sum(cuts)
    if total_len != n:
        raise AssertionError(
            f"Soma dos cortes ({total_len}) difere do tamanho da barra ({n})."
        )

    computed_revenue = sum(prices[c - 1] for c in cuts)
    if computed_revenue != revenue:
        raise AssertionError(
            f"Receita dos cortes ({computed_revenue}) difere da receita reportada ({revenue})."
        )
