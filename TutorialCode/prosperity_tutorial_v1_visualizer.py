from typing import List
import string
import jsonpickle
import numpy as np
import math

import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[: max_length - 3] + "..."


logger = Logger()


class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"

class Trader:

    def __init__(self, params = None):
        self.default_params = {
            Product.RAINFOREST_RESIN: {
                "fair_value": 1000,
                "take_width": 1,
                "clear_width": 1,
                "make_edge": 0,
                "position_limit": 50
            },
            Product.KELP: {
                "fair_value": 2025,
                "take_width": 1,
                "clear_width": 1,
                "make_edge": 0,
                "position_limit": 50
            }
        }

        self.params = params if params else self.default_params

        #Position limits
        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50
        }

        self.history = {}

    #Takes in an order_depth object for the given product
    def calculate_fair_value(self, product: str, order_depth: OrderDepth, method = "mid_price") -> float:
        if method == "mid_price" and len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
            #Calculate Fair Value by Current Mid Price
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            return (best_bid + best_ask) / 2

        elif method == "vwap" and len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
            #Calculate Fair Value by Current Vwap (only orders for current state)
            total_volume = 0
            total_value = 0

            for price, volume in order_depth.buy_orders.items():
                total_value += price*volume
                total_volume += volume
            for price, volume in order_depth.sell_orders.items():
                total_value += abs(price*volume)
                total_volume += abs(volume)
            
            return total_value/total_volume if total_volume != 0 else None

        else:
            return self.params[product].get("fair_value", 1000)
    
    #Update history of prices (using mid prices)
    def update_history(self, product: str, state: TradingState):
        if product not in self.history:
            self.history[product] = {
                "prices": [],
                "positions": [],
                "timestamps": []
            }
        if product in state.order_depths and len(state.order_depths[product].sell_orders) > 0 and len(state.order_depths[product].buy_orders) > 0:
            mid_price = self.calculate_fair_value(product, state.order_depths[product])
            self.history[product]["prices"].append(mid_price)
        
        position = state.position.get(product, 0)
        self.history[product]["positions"].append(position)

        self.history[product]["timestamps"].append(state.timestamp)

    #Generic market taking order generation
    def market_take(
        self, 
        product: str,
        fair_value: float,
        take_width:float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int = 0,
        sell_order_volume: int = 0,
        ) -> (int, int):

        position_limit = self.LIMIT[product]

        for ask_price in sorted(order_depth.sell_orders.keys()):
            if ask_price <= fair_value - take_width:
                volume = -1 * order_depth.sell_orders[ask_price]

                if position + buy_order_volume + volume <= position_limit:
                    buy_order_volume += volume
                    orders.append(Order(product, ask_price, volume))
            else:
                break
        
        for bid_price in sorted(order_depth.buy_orders.keys(), reverse = True):
            if bid_price >= fair_value + take_width:
                volume = order_depth.buy_orders[bid_price]
                if position - sell_order_volume - volume >= -position_limit:
                    sell_order_volume += volume
                    orders.append(Order(product, bid_price, -volume))

        return buy_order_volume, sell_order_volume

    #market taking control function
    def market_take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
    ) -> (List[Order], int, int):
        
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0
    
        buy_order_volume, sell_order_volume = self.market_take(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume
        )
        
        return orders, buy_order_volume, sell_order_volume
    
    #Generic market making function
    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: float,
        ask: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int
    ) -> (int, int):

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, bid, buy_quantity))
            buy_order_volume += buy_quantity

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, ask, -sell_quantity))
            sell_order_volume += sell_quantity
        
        return buy_order_volume, sell_order_volume

    #Market make cutting barely inside previous best offer
    def aggressive_market_make(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        min_edge: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        price_improvement: int
    ) -> (int, int):

        orders: List[Order] = []

        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        theoretical_bid = round(fair_value - min_edge)
        theoretical_ask = round(fair_value + min_edge)

        # Undercut best bid/ask if there's room
        bid_price = min(theoretical_bid, round(best_bid + price_improvement)) if best_bid and best_bid < theoretical_bid else theoretical_bid
        ask_price = max(theoretical_ask, round(best_ask - price_improvement)) if best_ask and best_ask > theoretical_ask else theoretical_ask

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid_price,
            ask_price,
            position,
            buy_order_volume,
            sell_order_volume
        )

        return orders, buy_order_volume, sell_order_volume

    #Market make at a fixed spread
    def passive_market_make(
        self,
        product: str,
        fair_value: str,
        min_edge: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int
    ) -> (int, int):

        orders: List[Order] = []

        bid_price = round(fair_value - min_edge)
        ask_price = round(fair_value + min_edge)

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid_price,
            ask_price,
            position,
            buy_order_volume,
            sell_order_volume
        )

        return orders, buy_order_volume, sell_order_volume

    #Market making control function
    def market_make_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        min_edge: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        aggressive: bool = True,
        price_improvement: int = 1
    ) -> (List[Order], int, int):
        
        if aggressive:
            return self.aggressive_market_make(
                product,
                order_depth,
                fair_value,
                min_edge,
                position,
                buy_order_volume,
                sell_order_volume,
                price_improvement
            )
        else:
            return self.passive_market_make(
                product,
                fair_value,
                min_edge,
                position,
                buy_order_volume,
                sell_order_volume
            )
    
    def run(self, state: TradingState):
        result = {}
        traderObject = {}

        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        conversions = 0
        timestamp = state.timestamp

        for product in state.order_depths.keys():
            if product not in self.params:
                continue
            
            position = state.position.get(product, 0)
            order_depth = state.order_depths[product]
            self.update_history(product, state)

            fair_value = self.calculate_fair_value(product, order_depth, method = "vwap")
            param = self.params[product]
            take_width = param["take_width"]
            make_edge = param["make_edge"]

            all_orders = []
            buy_order_volume = 0
            sell_order_volume = 0

            take_orders, buy_order_volume, sell_order_volume = self.market_take_orders(
                product,
                order_depth,
                fair_value,
                take_width,
                position
            )

            all_orders.extend(take_orders)

            make_orders, buy_order_volume, sell_order_volume = self.market_make_orders(
                product,
                order_depth,
                fair_value,
                make_edge,
                position,
                buy_order_volume,
                sell_order_volume,
                aggressive = True,
                price_improvement = 1
            )

            all_orders.extend(make_orders)

            result[product] = all_orders
        
        logger.flush(state, result, conversions, "")
        return result, conversions, jsonpickle.encode(traderObject)

