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
                "take_width": 1,
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
        self.history_window = 260

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
            

    def calculate_vwap_excluding_top(self, product: str, order_depth: OrderDepth) -> (int, int):
        total_volume = 0
        total_value = 0

        # Sort buy orders descending (best price first), skip the top one
        sorted_buy_orders = sorted(order_depth.buy_orders.items(), reverse= True)[1:]

        for price, volume in sorted_buy_orders:
            total_value += price * volume
            total_volume += volume

        # Sort sell orders ascending (best price first), skip the top one
        sorted_sell_orders = sorted(order_depth.sell_orders.items())[1:]

        for price, volume in sorted_sell_orders:
            total_value += abs(price * volume)
            total_volume += abs(volume)

        return total_value, total_volume

    def calculate_ema(self, product: str, value_type: str = "mid_prices", lookback: int = 10):
        values = self.market_history[product][value_type][-lookback:]
        filtered = []
        prev_valid = None

        for v in values:
            if v is None:
                if prev_valid is not None:
                    filtered.append(prev_valid)
            elif v>= 0:
                filtered.append(v)
                prev_valid = v

        if not filtered:
            return None
        
        series = pd.Series(filtered)

        return series.ewm(span = lookback, adjust = False).mean().iloc[-1]
           
    def lookback_vwap(self, product: str, value_type: str = "vwap_prices", lookback = 10):
        prices = self.market_history[product][value_type][-lookback:]
        volumes = self.market_history[product]["total_volumes"][-lookback:]
        filtered = []

        total_value = 0
        total_volume = 0

        for price, volume in zip(prices, volumes):
            total_value += price*volume
            total_volume += volume
        
        return total_value/total_volume
            


    #Takes in an order_depth object for the given product
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
        
        elif method == "mid_price_ema":
            return self.calculate_ema(product, value_type = "mid_prices", lookback = lookback)
        
        elif method == "vwap_ema":
            return self.calculate_ema(product, value_type = "vwap_prices", lookback = lookback)

        elif method == "mid_price_sma":
            return self.calculate_sma(product, value_type= "mid_prices", lookback = lookback)
        
        elif method == "vwap_sma":
            return self.calculate_sma(product, value_type = "vwap_prices", lookback = lookback)
        
        elif method == "multi_day_vwap_mid":
            return self.lookback_vwap(product, "mid_prices", lookback = lookback)
        
        elif method == "multi_day_vwap_vwap":
            return self.lookback_vwap(product, "vwap_prices", lookback = lookback)
        
        else:
            return self.params[product].get("fair_value", 1000)
    
    def calculate_sma(self, product: str, value_type: str = "mid_prices", lookback: int = 10):
        values = self.market_history[product][value_type][-lookback:]
        filtered = []
        prev_valid = None

        for v in values:
            if v is None:
                if prev_valid is not None:
                    filtered.append(prev_valid)
            elif v>= 0:
                filtered.append(v)
                prev_valid = v

        if not filtered:
            return None
        
        series = np.array(filtered)
        return np.mean(series)

    #def fair_value_kelp(self, lookback, method = "vwap"):
        

    #Update history of prices (using mid prices)
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

                if position + buy_order_volume < position_limit:

                    #Maybe add buy_order_volume (in bracket with position +)
                    volume = min(position_limit - (position), volume)
                    buy_order_volume += volume
                    orders.append(Order(product, ask_price, volume))
                    order_depth.sell_orders[ask_price] += volume
                    if order_depth.sell_orders[ask_price] == 0:
                        del order_depth.sell_orders[ask_price]
            else:
                break
        
        for bid_price in sorted(order_depth.buy_orders.keys(), reverse = True):
            if bid_price >= fair_value + take_width:
                volume = order_depth.buy_orders[bid_price]
                if position - sell_order_volume > -position_limit:
                    #Maybe/probably add sell_order_volume below (negative)
                    volume = min(position_limit + position , volume)
                    sell_order_volume += volume
                    orders.append(Order(product, bid_price, -volume))
                    order_depth.buy_orders[bid_price] -= volume
                    if order_depth.buy_orders[bid_price] == 0:
                        del order_depth.buy_orders[bid_price]
            else:
                break

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
    

    def clear_position(
        self,    
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        orders: List[Order],
        width: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ):
        
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_bid = round(fair_value - width)
        fair_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position(
            product,
            order_depth,
            fair_value,
            position,
            orders,
            clear_width,
            buy_order_volume,
            sell_order_volume,
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

    def resin_orders(
        self,
        state: TradingState
    ):
        product = Product.RAINFOREST_RESIN
        position = state.position.get(product, 0)
        order_depth = state.order_depths[product]

        #Methods: "vwap", "mid_price", "vwap_ema", "mid_price_ema", "multi_day_vwap_mid", "multi_day_vwap_vwap"
        fair_value = self.calculate_fair_value(product, order_depth, 
                                                method = "mid_price_ema", lookback = 10)
        
        fair_value = 10000
        
        param = self.params[product]
        take_width = param["take_width"]
        make_edge = param["make_width"]/2

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
        
        clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
            product,
            order_depth,
            fair_value,
            param["clear_width"],
            position,
            buy_order_volume,
            sell_order_volume
        )

        all_orders.extend(clear_orders)
        

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
        return all_orders

    def kelp_orders(
            self,
            state: TradingState
        ):
            product = Product.KELP
            position = state.position.get(product, 0)
            order_depth = state.order_depths[product]

            #Methods: "vwap", "mid_price", "vwap_ema", "mid_price_ema", "multi_day_vwap_mid", "multi_day_vwap_vwap"
            fair_value = self.calculate_fair_value(product, order_depth, 
                                                    method = "vwap_ema", lookback = 12)
            
            
            param = self.params[product]
            take_width = param["take_width"]
            make_edge = param["make_width"]/2

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
            
            clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                product,
                order_depth,
                fair_value,
                param["clear_width"],
                position,
                buy_order_volume,
                sell_order_volume
            )

            all_orders.extend(clear_orders)
            
            
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
            
            return all_orders

    def calculate_z_score(self, product, value_type = "mid_prices", lookback = 5):
        values = (self.market_history[product][value_type][-lookback:])
        current_price = self.market_history[product][value_type][-1]
        values = pd.Series(values)
        
        if len(values) < lookback or values.std() == 0:
            return None
        
        z_score = (current_price - values.mean())/values.std()

        return z_score

    def squid_orders(
        self,
        state: TradingState,
        lookback: int,
        z_cutoff:float     
    ):
        product = Product.SQUID_INK
        position = state.position.get(product, 0)
        order_depth = state.order_depths[product]

        all_orders = []

        z_score = self.calculate_z_score(product, "mid_prices", lookback)

        best_ask = min(order_depth.sell_orders.keys(), default=None)
        best_bid = max(order_depth.buy_orders.keys(), default=None)

        logger.print(z_score)

        if z_score is None or best_ask is None or best_bid is None:
            return []
        
        max_position = self.LIMIT[product]
        
        if z_score >= z_cutoff and position > -max_position:
            volume = min(-order_depth.sell_orders[best_ask], max_position + position)
            volume = max_position + position
            #if volume > 0:
            all_orders.append(Order(product, best_ask, -volume))

        # If z_score is low, expect reversion up: LONG
        elif z_score <= -z_cutoff and position < max_position:
            volume = min(order_depth.buy_orders[best_bid], max_position - position)
            volume = max_position - position
            #if volume < 0:
            all_orders.append(Order(product, best_bid , volume))

        # If z_score reverts to ~0: CLOSE POSITION
        elif abs(z_score) < 0:  # Can tweak the band
            if position > 0:
                volume = min(-order_depth.buy_orders[best_bid], position)
                if volume > 0:
                    all_orders.append(Order(product, best_bid, -volume))
            elif position < 0:
                volume = min(order_depth.sell_orders[best_ask], -position)
                if volume > 0:
                    all_orders.append(Order(product, best_ask, volume))

        return all_orders



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
        
        resin = self.resin_orders(state)
        result[Product.RAINFOREST_RESIN] = resin

        kelp = self.kelp_orders(state)
        result[Product.KELP] = kelp
        
        squid = self.squid_orders(state, 100, 1)
        result[Product.SQUID_INK] = squid
        
        for product in self.market_history:
            for key in self.market_history[product]:
                self.market_history[product][key] = self.market_history[product][key][-self.history_window:]

        traderObject["market_history"] = self.market_history
        logger.flush(state, result, conversions, "")
        return result, conversions, jsonpickle.encode(traderObject)

