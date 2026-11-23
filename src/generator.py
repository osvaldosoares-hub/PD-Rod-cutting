"""Gerador automático e reprodutível de instâncias para o Corte de Barras."""

from __future__ import annotations

import random


def generate_price_table(n: int, rng: random.Random) -> list[int]:
 
    prices: list[int] = []
    current = 0
    for _ in range(n):
        current += rng.randint(1, 10)
        prices.append(current)
    return prices


def generate_instances(
    sizes: list[int], num_instances: int, seed: int = 42
) -> dict[int, list[list[int]]]:
    
    rng = random.Random(seed)
    dataset: dict[int, list[list[int]]] = {}
    for n in sizes:
        instances: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        while len(instances) < num_instances:
            table = generate_price_table(n, rng)
            key = tuple(table)
           
            if key not in seen or len(seen) >= 10**min(n, 6):
                seen.add(key)
                instances.append(table)
        dataset[n] = instances
    return dataset
