import unittest

from binance_realtime.order_book import OrderBook, OrderBookGap


class OrderBookTests(unittest.TestCase):
    def test_finish_initialization_applies_bridging_event(self) -> None:
        book = OrderBook("DOGEUSDT")
        book.apply_or_buffer(
            {
                "U": 98,
                "u": 100,
                "b": [["0.09", "1"]],
                "a": [["0.11", "1"]],
            },
            max_buffered_events=10,
        )
        book.apply_or_buffer(
            {
                "U": 101,
                "u": 102,
                "b": [["0.10", "5"]],
                "a": [["0.12", "3"]],
            },
            max_buffered_events=10,
        )

        book.load_snapshot(
            {
                "lastUpdateId": 100,
                "bids": [["0.09", "1"]],
                "asks": [["0.11", "1"]],
            }
        )
        book.finish_initialization()

        self.assertTrue(book.initialized)
        self.assertEqual(book.last_update_id, 102)
        self.assertEqual(book.bids["0.10"], "5")
        self.assertEqual(book.asks["0.12"], "3")

    def test_stale_event_is_ignored(self) -> None:
        book = OrderBook("DOGEUSDT")
        book.load_snapshot(
            {
                "lastUpdateId": 100,
                "bids": [["0.09", "1"]],
                "asks": [["0.11", "1"]],
            }
        )
        book.initialized = True

        applied = book.apply_depth_event(
            {
                "U": 90,
                "u": 99,
                "b": [["0.10", "5"]],
                "a": [],
            }
        )

        self.assertFalse(applied)
        self.assertNotIn("0.10", book.bids)

    def test_gap_raises(self) -> None:
        book = OrderBook("DOGEUSDT")
        book.load_snapshot(
            {
                "lastUpdateId": 100,
                "bids": [["0.09", "1"]],
                "asks": [["0.11", "1"]],
            }
        )
        book.initialized = True

        with self.assertRaises(OrderBookGap):
            book.apply_depth_event(
                {
                    "U": 105,
                    "u": 106,
                    "b": [],
                    "a": [],
                }
            )

    def test_zero_quantity_removes_level(self) -> None:
        book = OrderBook("DOGEUSDT")
        book.load_snapshot(
            {
                "lastUpdateId": 100,
                "bids": [["0.09", "1"]],
                "asks": [["0.11", "1"]],
            }
        )
        book.initialized = True

        book.apply_depth_event(
            {
                "U": 101,
                "u": 101,
                "b": [["0.09", "0.00000000"]],
                "a": [],
            }
        )

        self.assertNotIn("0.09", book.bids)


if __name__ == "__main__":
    unittest.main()
