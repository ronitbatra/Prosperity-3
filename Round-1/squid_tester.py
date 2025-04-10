import unittest
from collections import defaultdict
import numpy as np

from datamodel import OrderDepth, TradingState, Order
from prosperity_r1_v2_mean_reversion import Trader, Product

class TestSquidStrategy(unittest.TestCase):
    def setUp(self):
        self.trader = Trader()
        self.trader.LIMIT = {
            Product.SQUID_INK: 20,
            Product.KELP: 1000,
            Product.RAINFOREST_RESIN: 300
        }
        self.trader.history_window = 20

        # Initialize market history
        self.trader.market_history = defaultdict(lambda: {
            "mid_prices": [],
            "vwap_prices": [],
            "total_volumes": [],
            "positions": [],
            "timestamps": []
        })

        # Populate mock mid prices for z-score
        mid_prices = [10000, 10010, 9990, 10005, 10015]
        for price in mid_prices:
            self.trader.market_history[Product.SQUID_INK]["mid_prices"].append(price)

    def mock_order_depth(self, bid_price=9995, bid_vol=5, ask_price=10005, ask_vol=-5):
        od = OrderDepth()
        od.buy_orders = {bid_price: bid_vol}
        od.sell_orders = {ask_price: ask_vol}
        return od

    def test_squid_orders_behavior(self):
        state = TradingState(
            traderData='',
            timestamp=1000,
            listings={Product.SQUID_INK: None},
            order_depths={Product.SQUID_INK: self.mock_order_depth()},
            own_trades={},
            market_trades={},
            position={Product.SQUID_INK: 0},
            observations=None
        )

        # Update history with the latest price to simulate a full run
        self.trader.update_history(Product.SQUID_INK, state)

        orders = self.trader.squid_orders(state, lookback=2, z_cutoff=0.3)

        z_score = self.trader.calculate_z_score(Product.SQUID_INK)
        print(f"Z-score: {z_score:.4f}")
        print("Generated Orders:")
        for order in orders:
            print(order)

        self.assertIsInstance(orders, list)

if __name__ == '__main__':
    print(OrderDepth.__init__.__annotations__)

    unittest.main()
