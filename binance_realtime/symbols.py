"""Utilities for loading Binance symbols from a text file."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re


_KNOWN_QUOTES = (
    "USDT",
    "USDC",
    "FDUSD",
    "TUSD",
    "BUSD",
    "BTC",
    "ETH",
    "BNB",
    "TRY",
    "EUR",
    "BRL",
)


@dataclass(frozen=True, slots=True)
class MarketSpec:
    """Normalized spot market symbol loaded from user input."""

    symbol: str
    raw: str

    @property
    def stream_symbol(self) -> str:
        return self.symbol.lower()


def load_market_specs(path: Path, default_quote: str = "USDT") -> list[MarketSpec]:
    """Load normalized Binance symbols from a text file.

    Accepted line examples:
      - ``DOGE``
      - ``DOGEUSDT``
      - ``DOGE/USDT``
      - ``symbol=DOGEUSDT, name=Dogecoin``

    Empty lines and everything after ``#`` are ignored. Duplicate symbols are
    removed while preserving the first occurrence.
    """

    default_quote = default_quote.strip().upper()
    seen: set[str] = set()
    markets: list[MarketSpec] = []

    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue

        token = _extract_symbol_token(line)
        if token is None:
            raise ValueError(f"{path}:{line_number}: cannot find a symbol in {raw_line!r}")

        symbol = normalize_symbol(token, default_quote)
        if symbol not in seen:
            markets.append(MarketSpec(symbol=symbol, raw=raw_line.strip()))
            seen.add(symbol)

    if not markets:
        raise ValueError(f"{path} did not contain any Binance symbols")
    return markets


def normalize_symbol(value: str, default_quote: str = "USDT") -> str:
    """Normalize a base coin or trading pair into a Binance spot symbol."""

    value = value.strip().upper()
    default_quote = default_quote.strip().upper()

    if "=" in value:
        value = value.split("=", 1)[1]
    value = value.strip().strip("\"'")
    value = re.sub(r"[^A-Z0-9/_:-]", "", value)
    if not value:
        raise ValueError("empty symbol")

    if any(separator in value for separator in ("/", "-", "_", ":")):
        parts = [part for part in re.split(r"[/_:-]+", value) if part]
        if len(parts) < 2:
            raise ValueError(f"invalid separated symbol {value!r}")
        return f"{parts[0]}{parts[1]}"

    if default_quote and not _has_known_quote(value):
        return f"{value}{default_quote}"
    return value


def _extract_symbol_token(line: str) -> str | None:
    for token in re.split(r"[\s,;]+", line):
        token = token.strip()
        if not token:
            continue
        key = token.split("=", 1)[0].strip().lower()
        if key in {"symbol", "pair", "market"} and "=" in token:
            return token
        if key in {"symbol", "pair", "market", "name", "coin", "base"}:
            continue
        if re.search(r"[A-Za-z0-9]", token):
            return token
    return None


def _has_known_quote(symbol: str) -> bool:
    return any(symbol.endswith(quote) and len(symbol) > len(quote) for quote in _KNOWN_QUOTES)
