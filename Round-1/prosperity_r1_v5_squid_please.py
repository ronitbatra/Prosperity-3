from typing import List
import string
import jsonpickle
import numpy as np
import pandas as pd
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
    SQUID_INK = "SQUID_INK"

class Trader:

    def __init__(self, params = None):
        self.default_params = {
            Product.RAINFOREST_RESIN: {
                "fair_value": 1000,
                "take_width": .5,
                "clear_width": .5,
                "make_width": 1,
                "position_limit": 50
            },
            Product.KELP: {
                "fair_value": 2000,
                "take_width": 1,
                "clear_width": .5,
                "make_width": 1,
                "position_limit": 50
            },
            Product.SQUID_INK: {
                "fair_value": 2000,
                "take_width": 1,
                "clear_width": .5,
                "make_width": 1,
                "position_limit": 50
            }
        }

        self.params = params if params else self.default_params

        #Position limits
        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50 
        }

        self.market_history = {}
        self.history_window = 100

    
    def update_history(self, product: str, state: TradingState):
        if product not in self.market_history:
            self.market_history[product] = {
                "mid_prices": [],
                "vwap_prices": [],
                "total_volumes": [],
                "positions": [],
                "timestamps": []
            }

        if product in state.order_depths and len(state.order_depths[product].sell_orders) > 0 and len(state.order_depths[product].buy_orders) > 0:
            mid_price = self.calculate_fair_value(product, state.order_depths[product])
            self.market_history[product]["mid_prices"].append(mid_price)
            total_value, total_volume = self.calculate_vwap(product, state.order_depths[product])
            vwap_price = total_value/total_volume
            self.market_history[product]["vwap_prices"].append(vwap_price)
            self.market_history[product]["total_volumes"].append(total_volume)

        else:
            self.market_history[product]["mid_prices"].append(-1)
            self.market_history[product]["vwap_prices"].append(-1)
            self.market_history[product]["total_volumes"].append(0)

        self.market_history[product]["positions"].append(state.position.get(product, 0))
        self.market_history[product]["timestamps"].append(state.timestamp)
    
    def calculate_fair_value(self, product: str, order_depth: OrderDepth, method = "mid_price", lookback = 10) -> float:
        if method == "mid_price" and len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
            #Calculate Fair Value by Current Mid Price
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            return (best_bid + best_ask) / 2

        elif method == "vwap" and len(order_depth.sell_orders) > 0 and len(order_depth.buy_orders) > 0:
            #Calculate Fair Value by Current Vwap (only orders for current state)
            total_value, total_volume = self.calculate_vwap(product, order_depth)

            return total_value/total_volume if total_volume != 0 else None
        
        
        
    def calculate_vwap(self, product: str, order_depth: OrderDepth) -> (int, int):
        total_volume = 0
        total_value = 0
        

        for price, volume in order_depth.buy_orders.items():
            total_value += price*volume
            total_volume += volume
        for price, volume in order_depth.sell_orders.items():
            total_value += abs(price*volume)
            total_volume += abs(volume)
        
        return total_value, total_volume
    
    def squid_orders(
            self,
            state: TradingState,
            product: str = Product.SQUID_INK,
            short: int = 20,
            long: int = 50,
    ):
        
        orders = []

        order_depth = state.order_depths[product]
        position = state.position.get(product, 0)
        max_position = self.LIMIT[product]

        if(len(self.market_history[product]['mid_prices'])<long+10):
            return []
        
        past_orders = pd.Series(self.market_history[product]['mid_prices'][-long-10:])
        short_ema = past_orders.ewm(span = short).mean()
        long_ema = past_orders.ewm(span = long).mean()
        
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None

        best_bid_vol = order_depth.buy_orders[best_bid]
        best_ask_vol = order_depth.sell_orders[best_ask]

        bid_quantity = min(max_position-position, best_bid_vol)
        ask_quantity = min(max_position + position, best_ask_vol)



        prev_short = short_ema.iloc[-2]
        prev_long = long_ema.iloc[-2]
        curr_short = short_ema.iloc[-1]
        curr_long = long_ema.iloc[-1]

        if(curr_short < curr_long and prev_short > prev_long):
            orders.append(Order(product, best_ask, -ask_quantity))
        elif(curr_short > curr_long and prev_short < prev_long):
            orders.append(Order(product, best_bid, bid_quantity))
        
        return orders

    def run(self, state: TradingState):
        result = {}

        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
            
            if "market_history" in traderObject:
                self.market_history = traderObject["market_history"]
        else:
            traderObject = {}

        conversions = 0
        timestamp = state.timestamp
        
        for product in state.listings.keys():
            self.update_history(Product.KELP, state)
            self.update_history(Product.RAINFOREST_RESIN, state)
            self.update_history(Product.SQUID_INK, state)
        

        squid = self.squid_orders(state, 
                                  )        
        result[Product.SQUID_INK] = squid
    
        for product in self.market_history:
            for key in self.market_history[product]:
                self.market_history[product][key] = self.market_history[product][key][-self.history_window:]

        traderObject["market_history"] = self.market_history


        logger.flush(state, result, conversions, "")
        return result, conversions, jsonpickle.encode(traderObject)

