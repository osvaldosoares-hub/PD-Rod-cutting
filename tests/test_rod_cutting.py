"""Testes automatizados de corretude para os algoritmos de Corte de Barras."""

from __future__ import annotations

import random
import pytest

from src.generator import generate_price_table
from src.rod_cutting import (
    rod_cutting_iterative,
    rod_cutting_memoized,
    rod_cutting_naive,
    validate_solution,
)

# Tabela clássica do livro CLRS (Cormen et al., Capítulo 15)
CLRS_PRICES = [1, 5, 8, 9, 10, 17, 17, 20, 24, 30]
CLRS_EXPECTED_REVENUE = {
    0: 0,
    1: 1,
    2: 5,
    3: 8,
    4: 10,
    5: 13,
    6: 17,
    7: 18,
    8: 22,
    9: 25,
    10: 30,
}


@pytest.mark.parametrize("n,expected", list(CLRS_EXPECTED_REVENUE.items()))
def test_clrs_benchmark_all_algorithms(n: int, expected: int) -> None:
    """Verifica todos os algoritmos contra a tabela clássica do CLRS."""
    assert rod_cutting_naive(CLRS_PRICES, n) == expected, f"rod_cutting_naive falhou para n={n}"
    for algo in (rod_cutting_memoized, rod_cutting_iterative):
        rev, cuts = algo(CLRS_PRICES, n)
        assert rev == expected, f"{algo.__name__} falhou para n={n}"
        validate_solution(CLRS_PRICES, n, rev, cuts)


def test_edge_cases() -> None:
    """Testa casos de borda: n=0, barra indivisível, cortes unitários e erros."""
    # n = 0
    assert rod_cutting_naive([], 0) == 0
    for algo in (rod_cutting_memoized, rod_cutting_iterative):
        assert algo([], 0) == (0, [])

    # Preço unitário muito alto -> cortar tudo em pedaços de tamanho 1
    prices_unit = [100, 1, 1, 1, 1]
    assert rod_cutting_naive(prices_unit, 5) == 500
    for algo in (rod_cutting_memoized, rod_cutting_iterative):
        rev, cuts = algo(prices_unit, 5)
        assert rev == 500
        assert cuts == [1, 1, 1, 1, 1]
        validate_solution(prices_unit, 5, rev, cuts)

    # Barra inteira é estritamente melhor
    prices_whole = [1, 2, 3, 4, 100]
    assert rod_cutting_naive(prices_whole, 5) == 100
    for algo in (rod_cutting_memoized, rod_cutting_iterative):
        rev, cuts = algo(prices_whole, 5)
        assert rev == 100
        assert cuts == [5]
        validate_solution(prices_whole, 5, rev, cuts)

    # Entradas inválidas
    for algo in (rod_cutting_naive, rod_cutting_memoized, rod_cutting_iterative):
        with pytest.raises(ValueError):
            algo([1, 2], -1)
        with pytest.raises(ValueError):
            algo([1, 2], 3)


def test_random_small_equivalence_all_three() -> None:
    """Compara os 3 algoritmos em 60 instâncias aleatórias pequenas."""
    rng = random.Random(12345)
    for n in range(1, 13):
        for _ in range(5):
            prices = generate_price_table(n, rng)
            rev_n = rod_cutting_naive(prices, n)
            rev_m, cuts_m = rod_cutting_memoized(prices, n)
            rev_i, cuts_i = rod_cutting_iterative(prices, n)

            assert rev_n == rev_m == rev_i
            validate_solution(prices, n, rev_m, cuts_m)
            validate_solution(prices, n, rev_i, cuts_i)


def test_random_large_equivalence_dp() -> None:
    """Compara PD Memoizada e PD Iterativa em entradas maiores (até n=1500)."""
    rng = random.Random(98765)
    for n in [50, 200, 600, 1200, 1500]:
        prices = generate_price_table(n, rng)
        rev_m, cuts_m = rod_cutting_memoized(prices, n)
        rev_i, cuts_i = rod_cutting_iterative(prices, n)

        assert rev_m == rev_i
        validate_solution(prices, n, rev_m, cuts_m)
        validate_solution(prices, n, rev_i, cuts_i)
