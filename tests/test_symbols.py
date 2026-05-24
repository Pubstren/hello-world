from pathlib import Path
import tempfile
import unittest

from binance_realtime.symbols import load_market_specs, normalize_symbol


class SymbolTests(unittest.TestCase):
    def test_normalize_symbol_appends_default_quote(self) -> None:
        self.assertEqual(normalize_symbol("doge"), "DOGEUSDT")
        self.assertEqual(normalize_symbol("doge", default_quote="USDC"), "DOGEUSDC")

    def test_normalize_symbol_keeps_explicit_pair(self) -> None:
        self.assertEqual(normalize_symbol("doge/usdt"), "DOGEUSDT")
        self.assertEqual(normalize_symbol("linkusdt"), "LINKUSDT")

    def test_load_market_specs_ignores_comments_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "symbols.txt"
            path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "doge",
                        "DOGEUSDT # duplicate",
                        "symbol=SOL/USDT, name=Solana",
                    ]
                ),
                encoding="utf-8",
            )

            markets = load_market_specs(path)

        self.assertEqual([market.symbol for market in markets], ["DOGEUSDT", "SOLUSDT"])


if __name__ == "__main__":
    unittest.main()
