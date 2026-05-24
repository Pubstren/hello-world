"""Trade state for Binance trade streams."""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from typing import Deque


@dataclass(frozen=True, slots=True)
class Trade:
    symbol: str
    trade_id: int
    price: str
    quantity: str
    buyer_order_id: int
    seller_order_id: int
    trade_time_ms: int
    event_time_ms: int
    buyer_is_market_maker: bool

    @classmethod
    def from_stream_event(cls, event: dict) -> "Trade":
        return cls(
            symbol=str(event["s"]),
            trade_id=int(event["t"]),
            price=str(event["p"]),
            quantity=str(event["q"]),
            buyer_order_id=int(event["b"]),
            seller_order_id=int(event["a"]),
            trade_time_ms=int(event["T"]),
            event_time_ms=int(event["E"]),
            buyer_is_market_maker=bool(event["m"]),
        )

    def to_json(self) -> dict:
        return asdict(self)


class TradeCache:
    """Bounded per-symbol recent trade cache."""

    def __init__(self, maxlen: int) -> None:
        if maxlen <= 0:
            raise ValueError("trade cache size must be positive")
        self._trades: dict[str, Deque[Trade]] = {}
        self._maxlen = maxlen

    def append(self, trade: Trade) -> None:
        trades = self._trades.setdefault(trade.symbol, deque(maxlen=self._maxlen))
        trades.append(trade)

    def recent(self, symbol: str, limit: int | None = None) -> list[Trade]:
        trades = list(self._trades.get(symbol, ()))
        if limit is not None:
            return trades[-limit:]
        return trades

    def counts(self) -> dict[str, int]:
        return {symbol: len(trades) for symbol, trades in self._trades.items()}
