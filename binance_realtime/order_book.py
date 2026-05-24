"""In-memory Binance spot order book state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class OrderBookGap(RuntimeError):
    """Raised when a depth update cannot be applied without missing data."""


@dataclass(frozen=True, slots=True)
class BookTop:
    symbol: str
    last_update_id: int
    best_bid: tuple[str, str] | None
    best_ask: tuple[str, str] | None
    bid_levels: int
    ask_levels: int


@dataclass(slots=True)
class OrderBook:
    """Maintain one Binance local order book from REST snapshots and deltas."""

    symbol: str
    bids: dict[str, str] = field(default_factory=dict)
    asks: dict[str, str] = field(default_factory=dict)
    last_update_id: int = 0
    initialized: bool = False
    buffered_events: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.bids.clear()
        self.asks.clear()
        self.last_update_id = 0
        self.initialized = False
        self.buffered_events.clear()

    def load_snapshot(self, snapshot: dict[str, Any]) -> None:
        self.bids = _levels_to_dict(snapshot.get("bids", ()))
        self.asks = _levels_to_dict(snapshot.get("asks", ()))
        self.last_update_id = int(snapshot["lastUpdateId"])
        self.initialized = False

    def apply_or_buffer(self, event: dict[str, Any], max_buffered_events: int) -> None:
        if not self.initialized:
            self.buffered_events.append(event)
            if len(self.buffered_events) > max_buffered_events:
                raise OrderBookGap(
                    f"{self.symbol}: buffered too many depth events before snapshot"
                )
            return
        self.apply_depth_event(event)

    def finish_initialization(self) -> None:
        """Apply buffered events according to Binance's local book algorithm."""

        remaining = [
            event for event in self.buffered_events if int(event["u"]) > self.last_update_id
        ]
        self.buffered_events.clear()

        for event in remaining:
            self.apply_depth_event(event)
        self.initialized = True

    def apply_depth_event(self, event: dict[str, Any]) -> bool:
        """Apply one diff depth event.

        Returns ``False`` when the event is stale. Raises ``OrderBookGap`` when
        the event starts after the next expected update ID.
        """

        first_update_id = int(event["U"])
        final_update_id = int(event["u"])

        if final_update_id <= self.last_update_id:
            return False

        expected_next = self.last_update_id + 1
        if first_update_id > expected_next:
            raise OrderBookGap(
                f"{self.symbol}: depth gap, expected update {expected_next}, "
                f"got range {first_update_id}-{final_update_id}"
            )

        _apply_side(self.bids, event.get("b", ()))
        _apply_side(self.asks, event.get("a", ()))
        self.last_update_id = final_update_id
        return True

    def top(self) -> BookTop:
        best_bid = max(self.bids.items(), key=lambda item: float(item[0]), default=None)
        best_ask = min(self.asks.items(), key=lambda item: float(item[0]), default=None)
        return BookTop(
            symbol=self.symbol,
            last_update_id=self.last_update_id,
            best_bid=best_bid,
            best_ask=best_ask,
            bid_levels=len(self.bids),
            ask_levels=len(self.asks),
        )

    def depth(self, levels: int) -> dict[str, Any]:
        """Return top-N levels as a JSON-serializable snapshot."""

        bids = sorted(self.bids.items(), key=lambda item: float(item[0]), reverse=True)[:levels]
        asks = sorted(self.asks.items(), key=lambda item: float(item[0]))[:levels]
        return {
            "symbol": self.symbol,
            "last_update_id": self.last_update_id,
            "bids": bids,
            "asks": asks,
        }


def _levels_to_dict(levels: Any) -> dict[str, str]:
    return {str(price): str(quantity) for price, quantity in levels if float(quantity) != 0.0}


def _apply_side(side: dict[str, str], updates: Any) -> None:
    for price, quantity in updates:
        price_key = str(price)
        quantity_value = str(quantity)
        if float(quantity_value) == 0.0:
            side.pop(price_key, None)
        else:
            side[price_key] = quantity_value
