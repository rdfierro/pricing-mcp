"""Load the bundled synthetic dataset (or one you point at with PRICING_MCP_DATA)."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data"


@dataclass(frozen=True)
class Product:
    sku: str
    name: str
    category: str
    tier: str
    size_oz: float
    unit_cost: float
    current_price: float

    @property
    def price_per_oz(self) -> float:
        return self.current_price / self.size_oz


@dataclass(frozen=True)
class Sale:
    sku: str
    week: int
    price: float
    units: int


@dataclass(frozen=True)
class CompetitorPrice:
    sku: str
    competitor: str
    price: float


@dataclass(frozen=True)
class Dataset:
    products: dict[str, Product]
    sales: tuple[Sale, ...]
    competitors: tuple[CompetitorPrice, ...]
    policy: dict


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def load_dataset(data_dir: str | Path | None = None) -> Dataset:
    d = Path(data_dir or os.environ.get("PRICING_MCP_DATA") or DEFAULT_DATA_DIR)
    products = {
        r["sku"]: Product(r["sku"], r["name"], r["category"], r["tier"],
                          float(r["size_oz"]), float(r["unit_cost"]), float(r["current_price"]))
        for r in _rows(d / "products.csv")
    }
    sales = tuple(Sale(r["sku"], int(r["week"]), float(r["price"]), int(r["units"]))
                  for r in _rows(d / "weekly_sales.csv"))
    comps = tuple(CompetitorPrice(r["sku"], r["competitor"], float(r["price"]))
                  for r in _rows(d / "competitor_prices.csv"))
    policy = json.loads((d / "policy.json").read_text())
    return Dataset(products, sales, comps, policy)


@lru_cache(maxsize=1)
def default_dataset() -> Dataset:
    return load_dataset()
