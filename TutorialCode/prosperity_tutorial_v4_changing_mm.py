from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle
import numpy as np
import pandas as pd
import math

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
                "make_width": 1,
                "position_limit": 50
            },
            Product.KELP: {
                "fair_value": 2020,
                "take_width": 1,
                "clear_width": 1,
                "make_width": 1,
                "position_limit": 50
            }
        }

        self.params = params if params else self.default_params

        #Position limits
        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50
        }

        self.market_history = {}
        self.history_window = 50

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

        elif method == "multi_day_vwap_mid":
            return self.lookback_vwap(product, "mid_prices", lookback = lookback)
        
        elif method == "multi_day_vwap_vwap":
            return self.lookback_vwap(product, "vwap_prices", lookback = lookback)
        
        else:
            return self.params[product].get("fair_value", 1000)
    
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
                    volume = min(position_limit - (position+buy_order_volume), volume)
                    buy_order_volume += volume
                    orders.append(Order(product, ask_price, volume))
                    order_depth.sell_orders[ask_price] += volume
            else:
                break
        
        for bid_price in sorted(order_depth.buy_orders.keys(), reverse = True):
            if bid_price >= fair_value + take_width:
                volume = order_depth.buy_orders[bid_price]
                if position - sell_order_volume > -position_limit:
                    volume = min(position_limit + position - sell_order_volume, volume)
                    sell_order_volume += volume
                    orders.append(Order(product, bid_price, -volume))
                    order_depth.buy_orders[bid_price] -= volume



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
    def improved_market_make(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        min_edge: float,     # Base spread from fair value
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        mode: str = "aggressive", # "aggressive" or "passive"
        price_improvement: int = 1,  # Amount to improve on best price when aggressive
        manage_position: bool = False,
        soft_position_limit: int = 10
    ) -> (List[Order], int, int):
        
        orders: List[Order] = []
        
        if mode == "passive":
            # Passive mode - just use fixed spread
            bid_price = round(fair_value - min_edge)
            ask_price = round(fair_value + min_edge)
            
        else:  # Aggressive mode
            # Find current best prices
            best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            
            # Calculate theoretical prices based on fair value
            theoretical_bid = round(fair_value - min_edge)
            theoretical_ask = round(fair_value + min_edge)
            
            # Aggressive pricing - improve on existing prices if within reasonable bounds
            if best_ask and best_ask > theoretical_ask:
                ask_price = round(min(best_ask - price_improvement, theoretical_ask + min_edge*2))
            else:
                ask_price = theoretical_ask
                
            if best_bid and best_bid < theoretical_bid:
                bid_price = round(max(best_bid + price_improvement, theoretical_bid - min_edge*2))
            else:
                bid_price = theoretical_bid
        
        # Position management - adjust quotes based on current position
        if manage_position:
            if position > soft_position_limit:
                # More aggressive to sell when long
                ask_price = max(ask_price - 1, bid_price + 1)
            elif position < -soft_position_limit:
                # More aggressive to buy when short
                bid_price = min(bid_price + 1, ask_price - 1)
        
        # Place orders
        position_limit = self.LIMIT[product]
        
        buy_quantity = position_limit - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, bid_price, buy_quantity))
            buy_order_volume += buy_quantity
        
        sell_quantity = position_limit + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, ask_price, -sell_quantity))
            sell_order_volume += sell_quantity
        
        return orders, buy_order_volume, sell_order_volume
    
    
    def market_make_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        min_edge: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        mode: str = "aggressive",  # "aggressive" or "passive"
        price_improvement: int = 1,
        manage_position: bool = False,
        soft_position_limit: int = 10
    ) -> (List[Order], int, int):
        """
        Control function for market making.
        """
        return self.improved_market_make(
            product,
            order_depth,
            fair_value,
            min_edge,
            position,
            buy_order_volume,
            sell_order_volume,
            mode,
            price_improvement,
            manage_position,
            soft_position_limit
        )


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

        for product in state.order_depths.keys():

            position = state.position.get(product, 0)
            order_depth = state.order_depths[product]

            self.update_history(product, state)

            #Methods: "vwap", "mid_price", "vwap_ema", "mid_price_ema", "multi_day_vwap_mid", "multi_day_vwap_vwap"
            fair_value_take = self.calculate_fair_value(product, order_depth, 
                                                   method = "vwap_ema", lookback = 10)
            
            fair_value_make = self.calculate_fair_value(product, order_depth,
                                                    method = "vwap_ema", lookback = 10)
            
            param = self.params[product]
            take_width = param["take_width"]
            make_edge = param["make_width"]/2

            all_orders = []
            buy_order_volume = 0
            sell_order_volume = 0

            take_orders, buy_order_volume, sell_order_volume = self.market_take_orders(
                product,
                order_depth,
                fair_value_take,
                take_width,
                position
            )

            all_orders.extend(take_orders)

            make_orders, buy_order_volume, sell_order_volume = self.market_make_orders(
                product,
                order_depth,
                fair_value_make,
                make_edge,
                position,
                buy_order_volume,
                sell_order_volume,
                price_improvement = 1,
                soft_position_limit= 10
            )

            all_orders.extend(make_orders)
            result[product] = all_orders

            for product in self.market_history:
                for key in self.market_history[product]:
                    self.market_history[product][key] = self.market_history[product][key][-self.history_window:]

            traderObject["market_history"] = self.market_history

        return result, conversions, jsonpickle.encode(traderObject)

