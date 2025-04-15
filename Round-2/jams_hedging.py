from typing import List, Any
import string
import jsonpickle
import numpy as np
import pandas as pd
import math
import json
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

        # Truncate state.traderData, trader_data, and self.logs to the same max. length.
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
    PICNIC_BASKET1 = "PICNIC_BASKET1"
    PICNIC_BASKET2 = "PICNIC_BASKET2"
    DJEMBES = "DJEMBES"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"


class Trader:
    def __init__(self, params=None):
        self.default_params = {
            Product.RAINFOREST_RESIN: {
                "fair_value": 1000,
                "take_width": 1,
                "clear_width": 0.5,
                "make_width": 1,
                "position_limit": 50,
            },
            Product.KELP: {
                "fair_value": 2000,
                "take_width": 1,
                "clear_width": 0.5,
                "make_width": 1,
                "position_limit": 50,
            },
            Product.SQUID_INK: {
                "fair_value": 2000,
                "take_width": 1,
                "clear_width": 0.5,
                "make_width": 1,
                "position_limit": 50,
            },
        }
        self.params = params if params else self.default_params

        # Position limits
        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
            Product.PICNIC_BASKET1: 60,
            Product.PICNIC_BASKET2: 100,
            Product.DJEMBES: 60,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
        }

        self.market_history = {}
        self.history_window = 200

    def calculate_vwap(self, product: str, order_depth: OrderDepth) -> (int, int):
        total_volume = 0
        total_value = 0
        for price, volume in order_depth.buy_orders.items():
            total_value += price * volume
            total_volume += volume
        for price, volume in order_depth.sell_orders.items():
            total_value += abs(price * volume)
            total_volume += abs(volume)
        return total_value, total_volume

    # The fair value is computed symmetrically so it remains unchanged.
    def calculate_fair_value(self, product: str, order_depth: OrderDepth, method="mid_price") -> float:
        if method == "mid_price" and order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            return (best_bid + best_ask) / 2
        elif method == "vwap" and order_depth.sell_orders and order_depth.buy_orders:
            total_value, total_volume = self.calculate_vwap(product, order_depth)
            return total_value / total_volume if total_volume != 0 else None

    def update_history(self, product: str, state: TradingState):
        if product not in self.market_history:
            self.market_history['price_history'][product] = {
                "mid_prices": [],
                "vwap_prices": [],
                "total_volumes": [],
                "positions": [],
                "timestamps": [],
            }

        if product in state.order_depths and state.order_depths[product].sell_orders and state.order_depths[product].buy_orders:
            mid_price = self.calculate_fair_value(product, state.order_depths[product])
            self.market_history['price_history'][product]["mid_prices"].append(mid_price)
            total_value, total_volume = self.calculate_vwap(product, state.order_depths[product])
            vwap_price = total_value / total_volume
            self.market_history['price_history'][product]["vwap_prices"].append(vwap_price)
            self.market_history['price_history'][product]["total_volumes"].append(total_volume)
        else:
            self.market_history['price_history'][product]["mid_prices"].append(-1)
            self.market_history['price_history'][product]["vwap_prices"].append(-1)
            self.market_history['price_history'][product]["total_volumes"].append(0)

        self.market_history['price_history'][product]["positions"].append(state.position.get(product, 0))
        self.market_history['price_history'][product]["timestamps"].append(state.timestamp)

    def get_max_units(self, state: TradingState, target: dict, limits: dict = None):
        max_units_possible = float("inf")
        for product, qty_per_unit in target.items():
            order_depth = state.order_depths[product]

            # Determine current position and position limit.
            current_pos = state.position.get(product, 0)
            if(limits):
                limit = limits[product]
            else:
                limit = self.LIMIT[product]

            # Available capacity is how many additional units you can add before reaching your limit.
            if qty_per_unit > 0:  # Buying
                capacity = limit - current_pos
            elif qty_per_unit < 0:  # Selling
                capacity = limit + current_pos  # How much more we can sell
            else:
                capacity = float("inf")
            # Calculate trade units allowed by position limits.
            units_capacity = capacity // abs(qty_per_unit) if abs(qty_per_unit) > 0 else float("inf")
            
            if qty_per_unit > 0:
                # For a positive leg, we are buying; so consider the liquidity at the best ask.
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    liquidity = abs(order_depth.sell_orders[best_ask])
                else:
                    liquidity = float("inf")
            elif qty_per_unit < 0:
                # For a negative leg, we are selling; so consider the liquidity at the best bid.
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    liquidity = order_depth.buy_orders[best_bid]
                else:
                    liquidity = float("inf")
            else:
                liquidity = float("inf")
            
            units_liquidity = liquidity // abs(qty_per_unit) if abs(qty_per_unit) > 0 else float("inf")

            # For this product, the maximum tradeable units is limited by the smaller of the two factors.
            max_units_for_product = min(units_capacity, units_liquidity)
            max_units_possible = min(max_units_possible, max_units_for_product)

        if max_units_possible == float("inf"):
            return 0
        return int(max_units_possible)

    def place_orders(self, state: TradingState, target: dict, max_units: int):
        orders = {}
        for product, quantity in target.items():
            volume = quantity * max_units
            if volume > 0:
                best_ask = min(state.order_depths[product].sell_orders.keys())
                orders[product] = [Order(product, best_ask, round(volume))]
            else:
                best_bid = max(state.order_depths[product].buy_orders.keys())
                orders[product] = [Order(product, best_bid, round(volume))]
        
        return orders

    def clear_spread_orders(self, state: TradingState, spread_value: str) -> dict:
        
        
        if spread_value not in self.market_history or not self.market_history[spread_value]:
            return {}
        
        
        current_spread = self.market_history[spread_value]
        
        clearing_target = {product: -volume for product, volume in current_spread.items()}
        
        max_units = self.get_max_units(state, clearing_target)
        
        # Scale the clearing target by max_units.
        # For example, if clearing_target["PICNIC_BASKET1"] is -1 and max_units equals 3,
        # then the effective order volume for PICNIC_BASKET1 will be -3.
        scaled_target = {product: clearing_target[product] * max_units for product in clearing_target}
        
        # For tracking/logging, store the scaled clearing target in market_history.
        self.market_history[spread_value] = scaled_target
        
        # Build the clearing orders based on the scaled target.
        orders = {}
        for product, volume in scaled_target.items():
            order_depth = state.order_depths[product]
            if volume > 0:
                # A positive volume means we need to buy; use the best ask price.
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    orders[product] = [Order(product, best_ask, round(volume))]
            elif volume < 0:
                # A negative volume means we need to sell; use the best bid price.
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    orders[product] = [Order(product, best_bid, round(volume))]
            # If volume is zero, no order is generated.
        
        # Clear the spread position in market_history once the clear orders are placed.
        self.market_history[spread_value] = {}
        
        return orders

    def basket_orders(self, state: TradingState, zscore_entry: int, zscore_exit: int, lookback: int):
        b1_price = self.calculate_fair_value(Product.PICNIC_BASKET1, state.order_depths[Product.PICNIC_BASKET1])
        b2_price = self.calculate_fair_value(Product.PICNIC_BASKET2, state.order_depths[Product.PICNIC_BASKET2])
        jam_price = self.calculate_fair_value(Product.JAMS, state.order_depths[Product.JAMS])
        croissant_price = self.calculate_fair_value(Product.CROISSANTS, state.order_depths[Product.CROISSANTS])
        djembe_price = self.calculate_fair_value(Product.DJEMBES, state.order_depths[Product.DJEMBES])

        synthetic = croissant_price * 2 + jam_price + djembe_price

        spread = b1_price - b2_price - synthetic

        if(not "spread_diff_history" in self.market_history):
            self.market_history["spread_diff_history"] = []
        
        self.market_history["spread_diff_history"].append(spread)
        
        past_spreads = self.market_history["spread_diff_history"][-lookback:]
        if(len(past_spreads) < lookback):
            return {}
        
        std = np.std(past_spreads)
        #std = 91
        mean = 30
        #mean = np.mean(past_spreads)
        #std = 120
        z_score = (spread - mean)/std
        if(z_score >= zscore_entry):
            target = {"PICNIC_BASKET1": -1, "PICNIC_BASKET2": 1, "JAMS": 1, "CROISSANTS": 2, "DJEMBES": 1}
        elif(z_score <= -zscore_entry):
            target = {"PICNIC_BASKET1": 1, "PICNIC_BASKET2": -1, "JAMS": -1, "CROISSANTS": -2, "DJEMBES": -1}
        elif(abs(z_score)<=zscore_exit):
            target = {"PICNIC_BASKET1": 0, "PICNIC_BASKET2": 0, "JAMS": 0, "CROISSANTS": 0, "DJEMBES": 0}
            return self.clear_spread_orders(state, "current_spread")
        else:
            return {}
        max_units = self.get_max_units(state, target)
        self.market_history["current_spread"] =  {p: v * max_units for p, v in target.items()}
        orders = self.place_orders(state, target, max_units)

        return orders

    def jam_spreads(self, state: TradingState, zscore_entry, zscore_exit, lookback = 200):
        all_orders = []
        
        #croissants to jams
        croissant_jam_hedge  = 2.504
        #jams to baskets
        basket_jam_hedge = 3.705
        #max position sizes for these spreads
        jc_max_c = 130
        jc_max_j = 130//croissant_jam_hedge

        jb_max_j = 148
        jb_max_b = 40

        basket2_price = self.calculate_fair_value("PICNIC_BASKET2", state.order_depths[Product.PICNIC_BASKET2])
        jam_price = self.calculate_fair_value("JAMS", state.order_depths[Product.JAMS])
        croissant_price = self.calculate_fair_value("CROISSANTS", state.order_depths[Product.CROISSANTS])

        spread1 = basket2_price - basket_jam_hedge * jam_price
        spread2 = croissant_jam_hedge * croissant_price - jam_price

        if(not "jam_croissant_spread" in self.market_history):
            self.market_history["jam_croissant_spread"] = []
            self.market_history["jam_basket2_spread"] = []
        self.market_history["jam_croissant_spread"].append([spread2])
        self.market_history["jam_basket2_spread"].append([spread1])

        past_jc_spread = self.market_history["jam_croissant_spread"][-lookback:]
        past_jb_spread = self.market_history["jam_basket2_spread"][-lookback:]
        if(len(past_jb_spread) < lookback or len(past_jb_spread) < lookback):
            return {}

        #std_jc = np.std(past_jc_spread)
        #std_jb = np.std(past_jb_spread)

        mean_jc = 4168
        mean_jb = 5975
        #mean = np.mean(past_spreads)
        std_jb = 63.9
        z_score_jb = (spread1 - mean_jb)/std_jb
        #z_score_jc = (spread2 - mean_jc)/std_jc

        if(z_score_jb >= zscore_entry):
            target = {"PICNIC_BASKET2": -1, "JAMS": basket_jam_hedge}
        elif(z_score_jb <= -zscore_entry):
            target = {"PICNIC_BASKET2": 1, "JAMS": -basket_jam_hedge}
        elif(abs(z_score_jb)<=zscore_exit):
            target = {"PICNIC_BASKET2": 0, "JAMS": 0}
            if("current_jb_spread" in self.market_history):
                return self.clear_spread_orders(state, 'current_jb_spread')
            else:
                return {}
        else:
            return {}
        limits = {"PICNIC_BASKET2": jb_max_b, "JAMS": jb_max_j}
        #limits = {}
        max_units = self.get_max_units(state, target, limits)
        self.market_history["current_jb_spread"] =  {p: v * max_units for p, v in target.items()}
        orders = self.place_orders(state, target, max_units)

        all_orders.extend(orders)

        return orders
   
    def run(self, state: TradingState):
        result = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
            if "market_history" in traderObject:
                self.market_history = traderObject["market_history"]
        else:
            traderObject = {}

        conversions = 0
        timestamp = state.timestamp

        # Use the second basket arbitrage method.
        result_basket = self.basket_orders(state, zscore_entry = 6, zscore_exit = 4, lookback = 100)
        result_jb = self.jam_spreads(state, 1.2, .2, 1)
        result_jb = {}
        #result_jb = self.jam_spreads(state, 5, 3, 200) works decently
        result = {
            key: result_basket.get(key, []) + result_jb.get(key, [])
            for key in set(result_basket.keys()) | set(result_jb.keys())
        }      

        
        # Truncate market history arrays.
        #for product in self.market_history['price_history']:
        #    for key in self.market_history[product]:
        #        self.market_history['price_history'][product][key] = self.market_history[product][key][-self.history_window:]

        self.market_history['spread_diff_history'] = self.market_history["spread_diff_history"][-self.history_window:]
        self.market_history['jam_croissant_spread'] = self.market_history["jam_croissant_spread"][-self.history_window:]
        self.market_history["jam_basket2_spread"] = self.market_history["jam_basket2_spread"][-self.history_window:]
        
        traderObject["market_history"] = self.market_history
        logger.flush(state, result, conversions, "")
        return result, conversions, jsonpickle.encode(traderObject)
