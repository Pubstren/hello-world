"""Command-line entry point for Binance realtime market data."""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from .stream import BinanceRealtimeClient, ClientConfig
from .symbols import load_market_specs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Maintain Binance spot local order books and raw trades from a "
            "text file of altcoin symbols."
        )
    )
    parser.add_argument("symbols_file", type=Path, help="TXT file with one coin or pair per line")
    parser.add_argument(
        "--quote",
        default="USDT",
        help="quote asset appended when a line only contains a base coin",
    )
    parser.add_argument("--rest-base-url", default="https://api.binance.com")
    parser.add_argument("--ws-base-url", default="wss://stream.binance.com:9443")
    parser.add_argument("--depth-limit", type=int, default=1000)
    parser.add_argument("--max-symbols-per-connection", type=int, default=100)
    parser.add_argument("--snapshot-concurrency", type=int, default=5)
    parser.add_argument("--max-buffered-depth-events", type=int, default=20_000)
    parser.add_argument("--trade-cache-size", type=int, default=10_000)
    parser.add_argument("--report-interval", type=float, default=5.0)
    parser.add_argument("--report-symbol-limit", type=int, default=20)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="optional directory for trades.jsonl and periodic book_tops.jsonl",
    )
    parser.add_argument("--writer-queue-size", type=int, default=100_000)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
    )
    return parser


async def async_main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    markets = load_market_specs(args.symbols_file, default_quote=args.quote)
    logging.getLogger(__name__).info(
        "loaded %s symbols: %s",
        len(markets),
        ", ".join(market.symbol for market in markets[:20]),
    )

    config = ClientConfig(
        rest_base_url=args.rest_base_url,
        ws_base_url=args.ws_base_url,
        depth_limit=args.depth_limit,
        max_symbols_per_connection=args.max_symbols_per_connection,
        snapshot_concurrency=args.snapshot_concurrency,
        max_buffered_depth_events=args.max_buffered_depth_events,
        trade_cache_size=args.trade_cache_size,
        report_interval_seconds=args.report_interval,
        report_symbol_limit=args.report_symbol_limit,
        output_dir=args.output_dir,
        writer_queue_size=args.writer_queue_size,
    )
    await BinanceRealtimeClient(markets, config).run_forever()


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
