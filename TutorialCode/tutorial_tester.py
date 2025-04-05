from datamodel import OrderDepth, TradingState, Order
from typing import Dict
from prosperity_tutorial_v1_visualizer import Trader, Product  # Assuming your code is in trader.py

def create_mock_order_depth(bids, asks):
    """Helper to create mock OrderDepth object from dicts"""
    od = OrderDepth()
    od.buy_orders = bids
    od.sell_orders = asks
    return od

def main():
    trader = Trader()

    # Example mock state
    timestamp = 0
    listings = {}  # Optional: Add if needed by your run()
    own_trades = {}  # Empty for testing
    market_trades = {}  # Empty for now

    # Two mock products
    order_depths = {
        Product.RAINFOREST_RESIN: create_mock_order_depth(
            {995: 10, 996: 5, 997: 10},
            {1005: -5, 1004: -5, 1003: -10}
        ),
        Product.KELP: create_mock_order_depth(
            {2024: 3, 2023: 2},
            {2026: -4, 2027: -1}
        )
    }

    # Mock position (start flat)
    position = {
        Product.RAINFOREST_RESIN: 0,
        Product.KELP: 0
    }

    observations = {
        "DOLPHIN_SIGHTINGS": 0,
        "UNDERWATER_CURRENTS": 0,
    }

    # Construct TradingState
    state = TradingState(
        traderData="",
        timestamp=timestamp,
        listings=listings,
        order_depths=order_depths,
        own_trades=own_trades,
        market_trades=market_trades,
        position=position,
        observations=observations
    )

    # Call run and print the output
    result, conversions, traderData = trader.run(state)
    for product, orders in result.items():
        print(f"{product} Orders:")
        for o in orders:
            print(f"  {o}")

if __name__ == "__main__":
    main()
