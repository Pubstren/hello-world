"""Async Binance spot websocket client."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import time
from typing import Any, Iterable

import aiohttp

from .order_book import OrderBook, OrderBookGap
from .symbols import MarketSpec
from .trades import Trade, TradeCache
from .writer import JsonlWriter


LOG = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ClientConfig:
    rest_base_url: str = "https://api.binance.com"
    ws_base_url: str = "wss://stream.binance.com:9443"
    depth_limit: int = 1000
    max_symbols_per_connection: int = 100
    snapshot_concurrency: int = 5
    max_buffered_depth_events: int = 20_000
    trade_cache_size: int = 10_000
    report_interval_seconds: float = 5.0
    report_symbol_limit: int = 20
    output_dir: Path | None = None
    writer_queue_size: int = 100_000


class BinanceRealtimeClient:
    """Maintain local order books and recent trades for many Binance symbols."""

    def __init__(self, markets: Iterable[MarketSpec], config: ClientConfig) -> None:
        self.markets = list(markets)
        if not self.markets:
            raise ValueError("at least one market is required")
        if config.max_symbols_per_connection <= 0:
            raise ValueError("max symbols per connection must be positive")
        if config.snapshot_concurrency <= 0:
            raise ValueError("snapshot concurrency must be positive")

        self.config = config
        self.books = {market.symbol: OrderBook(market.symbol) for market in self.markets}
        self._locks = {symbol: asyncio.Lock() for symbol in self.books}
        self.trades = TradeCache(config.trade_cache_size)
        self.stats: dict[str, int] = {
            "depth_events": 0,
            "trade_events": 0,
            "reconnects": 0,
            "snapshot_requests": 0,
        }
        self._session: aiohttp.ClientSession | None = None
        self._snapshot_sem = asyncio.Semaphore(config.snapshot_concurrency)
        self._writer: JsonlWriter | None = None

    async def run_forever(self) -> None:
        timeout = aiohttp.ClientTimeout(total=30)
        connector = aiohttp.TCPConnector(limit=0, ttl_dns_cache=300)
        async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
            self._session = session
            if self.config.output_dir is not None:
                self._writer = JsonlWriter(
                    self.config.output_dir,
                    queue_size=self.config.writer_queue_size,
                )
                await self._writer.start()

            tasks = [
                asyncio.create_task(
                    self._run_stream_group(index, group),
                    name=f"binance-ws-{index}",
                )
                for index, group in enumerate(
                    _chunks(self.markets, self.config.max_symbols_per_connection), 1
                )
            ]
            if self.config.report_interval_seconds > 0:
                tasks.append(
                    asyncio.create_task(self._report_loop(), name="binance-report")
                )

            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                if self._writer is not None:
                    await self._writer.close()

    async def _run_stream_group(self, index: int, markets: list[MarketSpec]) -> None:
        symbols = [market.symbol for market in markets]
        streams = [
            stream
            for market in markets
            for stream in (
                f"{market.stream_symbol}@depth@100ms",
                f"{market.stream_symbol}@trade",
            )
        ]
        url = f"{self.config.ws_base_url.rstrip('/')}/stream?streams={'/'.join(streams)}"
        backoff = 1.0

        while True:
            snapshot_tasks: list[asyncio.Task[None]] = []
            try:
                await self._reset_books(symbols)
                LOG.info("connecting stream group %s with %s symbols", index, len(symbols))
                session = self._require_session()
                async with session.ws_connect(url, heartbeat=30, receive_timeout=90) as ws:
                    LOG.info("connected stream group %s", index)
                    backoff = 1.0
                    snapshot_tasks = [
                        asyncio.create_task(
                            self._initialize_book(symbol),
                            name=f"snapshot-{symbol}",
                        )
                        for symbol in symbols
                    ]

                    async for message in ws:
                        if message.type == aiohttp.WSMsgType.TEXT:
                            await self._handle_ws_message(message.data)
                        elif message.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            break
            except asyncio.CancelledError:
                raise
            except Exception:
                self.stats["reconnects"] += 1
                LOG.exception(
                    "stream group %s failed; reconnecting in %.1fs", index, backoff
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)
            finally:
                for task in snapshot_tasks:
                    task.cancel()
                await asyncio.gather(*snapshot_tasks, return_exceptions=True)

    async def _reset_books(self, symbols: Iterable[str]) -> None:
        for symbol in symbols:
            async with self._locks[symbol]:
                self.books[symbol].reset()

    async def _initialize_book(self, symbol: str) -> None:
        while True:
            try:
                snapshot = await self._fetch_depth_snapshot(symbol)
                async with self._locks[symbol]:
                    book = self.books[symbol]
                    book.load_snapshot(snapshot)
                    try:
                        book.finish_initialization()
                    except OrderBookGap:
                        LOG.warning("%s snapshot missed buffered deltas; retrying", symbol)
                        continue

                    top = book.top()
                    LOG.info(
                        "%s order book initialized at update %s (%s bids/%s asks)",
                        symbol,
                        top.last_update_id,
                        top.bid_levels,
                        top.ask_levels,
                    )
                    return
            except asyncio.CancelledError:
                raise
            except Exception:
                LOG.exception("%s snapshot initialization failed; retrying", symbol)
                await asyncio.sleep(2.0)

    async def _fetch_depth_snapshot(self, symbol: str) -> dict[str, Any]:
        session = self._require_session()
        url = f"{self.config.rest_base_url.rstrip('/')}/api/v3/depth"
        params = {"symbol": symbol, "limit": str(self.config.depth_limit)}
        async with self._snapshot_sem:
            self.stats["snapshot_requests"] += 1
            async with session.get(url, params=params) as response:
                if response.status != 200:
                    body = await response.text()
                    raise RuntimeError(
                        f"{symbol}: snapshot request failed with "
                        f"HTTP {response.status}: {body[:500]}"
                    )
                return await response.json()

    async def _handle_ws_message(self, raw: str) -> None:
        message = json.loads(raw)
        event = message.get("data", message)
        event_type = event.get("e")
        if event_type == "depthUpdate":
            await self._handle_depth_event(event)
        elif event_type == "trade":
            self._handle_trade_event(event)

    async def _handle_depth_event(self, event: dict[str, Any]) -> None:
        symbol = str(event["s"])
        lock = self._locks.get(symbol)
        if lock is None:
            return

        async with lock:
            self.books[symbol].apply_or_buffer(
                event,
                max_buffered_events=self.config.max_buffered_depth_events,
            )
        self.stats["depth_events"] += 1

    def _handle_trade_event(self, event: dict[str, Any]) -> None:
        trade = Trade.from_stream_event(event)
        self.trades.append(trade)
        self.stats["trade_events"] += 1
        if self._writer is not None:
            self._writer.write_nowait("trades.jsonl", trade.to_json())

    async def _report_loop(self) -> None:
        while True:
            await asyncio.sleep(self.config.report_interval_seconds)
            report = await self._build_report()
            LOG.info(
                "status books=%s trades=%s depth_events=%s trade_events=%s reconnects=%s",
                report["initialized_books"],
                sum(self.trades.counts().values()),
                self.stats["depth_events"],
                self.stats["trade_events"],
                self.stats["reconnects"],
            )
            for top in report["tops"][: self.config.report_symbol_limit]:
                LOG.info(
                    "%s bid=%s ask=%s update=%s",
                    top["symbol"],
                    top["best_bid"],
                    top["best_ask"],
                    top["last_update_id"],
                )
            if self._writer is not None:
                self._writer.write_nowait("book_tops.jsonl", report)

    async def _build_report(self) -> dict[str, Any]:
        tops: list[dict[str, Any]] = []
        initialized_books = 0
        for symbol in sorted(self.books):
            async with self._locks[symbol]:
                book = self.books[symbol]
                if book.initialized:
                    initialized_books += 1
                top = book.top()
                tops.append(
                    {
                        "symbol": top.symbol,
                        "last_update_id": top.last_update_id,
                        "best_bid": top.best_bid,
                        "best_ask": top.best_ask,
                        "bid_levels": top.bid_levels,
                        "ask_levels": top.ask_levels,
                        "initialized": book.initialized,
                    }
                )
        return {
            "time_ms": int(time.time() * 1000),
            "initialized_books": initialized_books,
            "total_books": len(self.books),
            "stats": dict(self.stats),
            "writer_dropped": dict(self._writer.dropped) if self._writer else {},
            "tops": tops,
        }

    def _require_session(self) -> aiohttp.ClientSession:
        if self._session is None:
            raise RuntimeError("client session is not started")
        return self._session


def _chunks(items: list[MarketSpec], size: int) -> Iterable[list[MarketSpec]]:
    for index in range(0, len(items), size):
        yield items[index : index + size]
