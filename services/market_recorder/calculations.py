"""
Order Book Analysis Calculations

Ported from web/orderbook-tug/src/lib/utils/calculations.ts
to enable server-side computation during replay.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Tuple, Dict


@dataclass
class OrderLevel:
    """Single price level in order book."""
    price: float
    quantity: float


@dataclass
class CenterOfMass:
    """Center of mass for one side of the order book."""
    price: float
    total_quantity: float
    active_quantity: float  # Volume within 1 sigma of COM
    sigma: float  # Standard deviation
    side: str  # "bid" or "ask"


@dataclass
class QuantileLine:
    """A quantile line connecting bid and ask distributions."""
    percentile: int
    bid_price: float
    ask_price: float


@dataclass
class TensionField:
    """Full tension field between bid and ask distributions."""
    lines: List[QuantileLine]
    bid_total: float
    ask_total: float
    convergence_ratio: float
    shape: str  # "hourglass", "diamond", or "parallel"


@dataclass
class TensionSnapshot:
    """Snapshot of tension field state at a point in time."""
    tick: int
    timestamp: datetime
    bid_com: CenterOfMass
    ask_com: CenterOfMass
    spread: float
    slope_angle: float
    bid_dominance: float


@dataclass
class Trade:
    """Individual trade."""
    price: float
    quantity: float
    timestamp: datetime
    is_buyer_maker: bool
    trade_id: Optional[int] = None


@dataclass
class TradeDensity:
    """Trade density at a price level."""
    price: float
    density: float  # Normalized (0-1)
    count: int


@dataclass
class Candle:
    """OHLCV candlestick."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    trade_count: int = 0


@dataclass
class TrendState:
    """Trend analysis state from candle data (ThinkScript-style)."""
    body_mid: float
    range_mid: float
    center: float
    momentum: float
    upper_mid: float
    lower_mid: float
    dynamic_support: float
    dynamic_resistance: float
    bull_pressure: float
    bear_pressure: float
    bull_positive: float
    bull_negative: float
    bear_positive: float
    bear_negative: float
    bull_strength: float
    bear_strength: float
    trend_support: float
    trend_resistance: float


def calculate_com(levels: List[OrderLevel], side: str) -> CenterOfMass:
    """
    Calculate center of mass for one side of the order book.

    COM = Σ(price × quantity) / Σ(quantity)
    Also calculates volume within ±1 sigma of COM (the "active" volume).

    Args:
        levels: List of order levels [(price, qty), ...]
        side: "bid" or "ask"

    Returns:
        CenterOfMass with price, quantities, and sigma
    """
    if not levels:
        return CenterOfMass(
            price=0.0,
            total_quantity=0.0,
            active_quantity=0.0,
            sigma=0.0,
            side=side,
        )

    weighted_sum = 0.0
    total_qty = 0.0

    for level in levels:
        weighted_sum += level.price * level.quantity
        total_qty += level.quantity

    com_price = weighted_sum / total_qty if total_qty > 0 else levels[0].price

    # Calculate weighted standard deviation
    variance_sum = 0.0
    for level in levels:
        diff = level.price - com_price
        variance_sum += level.quantity * diff * diff

    sigma = math.sqrt(variance_sum / total_qty) if total_qty > 0 else 0.0

    # Calculate volume within COM ± 1 sigma (the "active" volume)
    active_qty = 0.0
    for level in levels:
        if abs(level.price - com_price) <= sigma:
            active_qty += level.quantity

    return CenterOfMass(
        price=com_price,
        total_quantity=total_qty,
        active_quantity=active_qty,
        sigma=sigma,
        side=side,
    )


def calculate_quantiles(
    levels: List[OrderLevel],
    percentiles: List[int] = None,
) -> Dict[int, float]:
    """
    Calculate quantile prices for a distribution.

    Treats volume as probability mass and finds CDF percentiles.

    Args:
        levels: List of order levels (should be sorted appropriately)
        percentiles: List of percentiles to calculate (default: 10, 20, ..., 90)

    Returns:
        Dict mapping percentile to price
    """
    if percentiles is None:
        percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    if not levels:
        return {}

    # Calculate total volume
    total = sum(level.quantity for level in levels)
    if total <= 0:
        return {}

    # Build CDF
    cdf: List[Tuple[float, float]] = []  # (price, cumulative_pct)
    cumulative = 0.0

    for level in levels:
        cumulative += level.quantity
        cdf.append((level.price, (cumulative / total) * 100))

    # Interpolate quantiles
    result: Dict[int, float] = {}

    for target_pct in percentiles:
        prev_price = cdf[0][0]
        prev_pct = 0.0

        for price, cum_pct in cdf:
            if cum_pct >= target_pct:
                # Interpolate
                if cum_pct == prev_pct:
                    result[target_pct] = price
                else:
                    ratio = (target_pct - prev_pct) / (cum_pct - prev_pct)
                    result[target_pct] = prev_price + ratio * (price - prev_price)
                break
            prev_price = price
            prev_pct = cum_pct

        # If we didn't find it, use last price
        if target_pct not in result:
            result[target_pct] = cdf[-1][0]

    return result


def calculate_tension_field(
    bids: List[OrderLevel],
    asks: List[OrderLevel],
) -> TensionField:
    """
    Calculate the full tension field between bid and ask distributions.

    Args:
        bids: Bid order levels
        asks: Ask order levels

    Returns:
        TensionField with quantile lines and shape analysis
    """
    percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    # For bids: sort highest first (accumulate from top of book)
    sorted_bids = sorted(bids, key=lambda x: x.price, reverse=True)
    # For asks: sort lowest first (accumulate from bottom of book)
    sorted_asks = sorted(asks, key=lambda x: x.price)

    bid_quantiles = calculate_quantiles(sorted_bids, percentiles)
    ask_quantiles = calculate_quantiles(sorted_asks, percentiles)

    lines: List[QuantileLine] = []

    for pct in percentiles:
        bid_price = bid_quantiles.get(pct)
        # Mirror the percentile: 10% bid -> 90% ask, 20% bid -> 80% ask, etc.
        ask_price = ask_quantiles.get(100 - pct)

        if bid_price is not None and ask_price is not None:
            lines.append(QuantileLine(
                percentile=pct,
                bid_price=bid_price,
                ask_price=ask_price,
            ))

    bid_total = sum(level.quantity for level in bids)
    ask_total = sum(level.quantity for level in asks)

    # Calculate convergence ratio
    convergence_ratio = 1.0
    shape = "parallel"

    if len(lines) >= 5:
        outer_spread = (
            (lines[-1].ask_price - lines[-1].bid_price) +
            (lines[0].ask_price - lines[0].bid_price)
        ) / 2
        inner_idx = len(lines) // 2
        inner_spread = lines[inner_idx].ask_price - lines[inner_idx].bid_price

        if inner_spread > 0:
            convergence_ratio = outer_spread / inner_spread

        if convergence_ratio > 1.1:
            shape = "hourglass"
        elif convergence_ratio < 0.9:
            shape = "diamond"

    return TensionField(
        lines=lines,
        bid_total=bid_total,
        ask_total=ask_total,
        convergence_ratio=convergence_ratio,
        shape=shape,
    )


def create_tension_snapshot(
    tick: int,
    timestamp: datetime,
    bids: List[OrderLevel],
    asks: List[OrderLevel],
) -> TensionSnapshot:
    """
    Create a tension snapshot from current market state.

    Args:
        tick: Frame/tick number
        timestamp: Snapshot timestamp
        bids: Bid order levels
        asks: Ask order levels

    Returns:
        TensionSnapshot with COM, spread, slope, and dominance
    """
    bid_com = calculate_com(bids, "bid")
    ask_com = calculate_com(asks, "ask")

    best_bid = bids[0].price if bids else 0.0
    best_ask = asks[0].price if asks else 0.0
    spread = max(0.0001, best_ask - best_bid)

    # Calculate slope angle
    price_diff = ask_com.price - bid_com.price
    normalized = price_diff / spread
    slope_angle = math.degrees(math.atan2(normalized, 1.0))

    # Calculate bid dominance
    total = bid_com.total_quantity + ask_com.total_quantity
    bid_dominance = bid_com.total_quantity / total if total > 0 else 0.5

    return TensionSnapshot(
        tick=tick,
        timestamp=timestamp,
        bid_com=bid_com,
        ask_com=ask_com,
        spread=spread,
        slope_angle=slope_angle,
        bid_dominance=bid_dominance,
    )


def calculate_trade_density(trades: List[Trade]) -> List[TradeDensity]:
    """
    Calculate trade density using unique prices.

    Counts occurrences at each unique price and normalizes.

    Args:
        trades: List of trades

    Returns:
        List of TradeDensity sorted by price
    """
    if not trades:
        return []

    # Count occurrences of each unique price
    count_map: Dict[float, int] = {}
    for trade in trades:
        count_map[trade.price] = count_map.get(trade.price, 0) + 1

    # Convert to list and sort by price
    unique_prices = sorted(count_map.items(), key=lambda x: x[0])

    # Normalize counts to sum to 1
    total_count = len(trades)

    return [
        TradeDensity(
            price=price,
            density=count / total_count,
            count=count,
        )
        for price, count in unique_prices
    ]


def calculate_vwap(trades: List[Trade]) -> Optional[float]:
    """
    Calculate Volume Weighted Average Price.

    VWAP = Σ(price × quantity) / Σ(quantity)

    Args:
        trades: List of trades

    Returns:
        VWAP or None if no trades
    """
    if not trades:
        return None

    total_value = 0.0
    total_volume = 0.0

    for trade in trades:
        total_value += trade.price * trade.quantity
        total_volume += trade.quantity

    return total_value / total_volume if total_volume > 0 else None


def aggregate_candles(
    trades: List[Trade],
    interval_seconds: int = 30,
) -> List[Candle]:
    """
    Aggregate trades into OHLCV candles.

    Args:
        trades: List of trades sorted by timestamp
        interval_seconds: Candle interval in seconds

    Returns:
        List of Candle objects
    """
    if not trades:
        return []

    candles: List[Candle] = []
    current_candle: Optional[Candle] = None
    current_bucket: Optional[float] = None

    for trade in sorted(trades, key=lambda t: t.timestamp):
        # Calculate bucket start time
        ts = trade.timestamp.timestamp()
        bucket = (ts // interval_seconds) * interval_seconds

        if current_bucket is None or bucket != current_bucket:
            # Save previous candle
            if current_candle is not None:
                candles.append(current_candle)

            # Start new candle
            current_bucket = bucket
            current_candle = Candle(
                timestamp=datetime.fromtimestamp(bucket, tz=trade.timestamp.tzinfo),
                open=trade.price,
                high=trade.price,
                low=trade.price,
                close=trade.price,
                volume=trade.quantity,
                trade_count=1,
            )
        else:
            # Update current candle
            if current_candle is not None:
                current_candle.high = max(current_candle.high, trade.price)
                current_candle.low = min(current_candle.low, trade.price)
                current_candle.close = trade.price
                current_candle.volume += trade.quantity
                current_candle.trade_count += 1

    # Don't forget the last candle
    if current_candle is not None:
        candles.append(current_candle)

    return candles


def calculate_trend_state(
    candle: Candle,
    prev_center: Optional[float] = None,
    prev_upper_mid: Optional[float] = None,
    prev_lower_mid: Optional[float] = None,
    prev_dynamic_support: Optional[float] = None,
    prev_dynamic_resistance: Optional[float] = None,
    prev_trend_support: Optional[float] = None,
    prev_trend_resistance: Optional[float] = None,
) -> TrendState:
    """
    Calculate trend state from candle OHLC data.

    Based on ThinkScript trend analysis logic.

    Args:
        candle: Current candle
        prev_*: Previous state values for continuity

    Returns:
        TrendState with support/resistance and pressure indicators
    """
    o, h, l, c = candle.open, candle.high, candle.low, candle.close

    # Candle midpoint calculations
    body_mid = (o + c) / 2  # Center of candle body
    range_mid = (h + l) / 2  # Center of candle range
    center = (body_mid + range_mid) / 2  # Overall price center

    # Momentum: change in center from previous candle
    momentum = center - prev_center if prev_center is not None else 0.0

    # Midpoint extremes
    upper_mid = max(body_mid, range_mid)
    lower_mid = min(body_mid, range_mid)

    # Dynamic support: updates when lower_mid rises
    if prev_lower_mid is None or prev_dynamic_support is None:
        dynamic_support = lower_mid
    else:
        dynamic_support = lower_mid if lower_mid > prev_lower_mid else prev_dynamic_support

    # Dynamic resistance: updates when upper_mid falls
    if prev_upper_mid is None or prev_dynamic_resistance is None:
        dynamic_resistance = upper_mid
    else:
        dynamic_resistance = upper_mid if upper_mid < prev_upper_mid else prev_dynamic_resistance

    # Pressure calculations
    bull_pressure = center - dynamic_support + momentum
    bear_pressure = center - dynamic_resistance + momentum

    # Decompose into positive/negative components
    bull_positive = bull_pressure if bull_pressure > 0 else 0.0
    bull_negative = bull_pressure if bull_pressure < 0 else 0.0
    bear_positive = bear_pressure if bear_pressure > 0 else 0.0
    bear_negative = bear_pressure if bear_pressure < 0 else 0.0

    # Overall trend strength
    bull_strength = bull_positive + bear_positive
    bear_strength = bull_negative + bear_negative

    # Trend support: updates to candle low during pure uptrend
    if bull_strength > 0 and bear_strength == 0:
        trend_support = l
    else:
        trend_support = prev_trend_support if prev_trend_support is not None else l

    # Trend resistance: updates to candle high during pure downtrend
    if bear_strength < 0 and bull_strength == 0:
        trend_resistance = h
    else:
        trend_resistance = prev_trend_resistance if prev_trend_resistance is not None else h

    return TrendState(
        body_mid=body_mid,
        range_mid=range_mid,
        center=center,
        momentum=momentum,
        upper_mid=upper_mid,
        lower_mid=lower_mid,
        dynamic_support=dynamic_support,
        dynamic_resistance=dynamic_resistance,
        bull_pressure=bull_pressure,
        bear_pressure=bear_pressure,
        bull_positive=bull_positive,
        bull_negative=bull_negative,
        bear_positive=bear_positive,
        bear_negative=bear_negative,
        bull_strength=bull_strength,
        bear_strength=bear_strength,
        trend_support=trend_support,
        trend_resistance=trend_resistance,
    )


def calculate_trend_series(candles: List[Candle]) -> List[TrendState]:
    """
    Calculate trend state series from a list of candles.

    Maintains state continuity across candles.

    Args:
        candles: List of candles sorted by timestamp

    Returns:
        List of TrendState for each candle
    """
    if not candles:
        return []

    states: List[TrendState] = []

    prev_center = None
    prev_upper_mid = None
    prev_lower_mid = None
    prev_dynamic_support = None
    prev_dynamic_resistance = None
    prev_trend_support = None
    prev_trend_resistance = None

    for candle in candles:
        state = calculate_trend_state(
            candle,
            prev_center=prev_center,
            prev_upper_mid=prev_upper_mid,
            prev_lower_mid=prev_lower_mid,
            prev_dynamic_support=prev_dynamic_support,
            prev_dynamic_resistance=prev_dynamic_resistance,
            prev_trend_support=prev_trend_support,
            prev_trend_resistance=prev_trend_resistance,
        )
        states.append(state)

        # Update previous values for next iteration
        prev_center = state.center
        prev_upper_mid = state.upper_mid
        prev_lower_mid = state.lower_mid
        prev_dynamic_support = state.dynamic_support
        prev_dynamic_resistance = state.dynamic_resistance
        prev_trend_support = state.trend_support
        prev_trend_resistance = state.trend_resistance

    return states


# ============================================================
# COM Intersection Calculations
# ============================================================

def calculate_com_intersection(
    bid_com_price: float,
    ask_com_price: float,
    target_price: float,
) -> Optional[float]:
    """
    Find where a horizontal line at target_price intersects the BID-ASK COM line.

    The COM line connects BID COM (left) to ASK COM (right) on the tension field.
    This calculates the normalized X position (0-1) where a price level crosses it.

    Args:
        bid_com_price: BID center of mass price
        ask_com_price: ASK center of mass price
        target_price: Price to find intersection for (SUP, RES, or LAST)

    Returns:
        Normalized position (0 = at bid COM, 1 = at ask COM)
        None if target_price is outside the COM range
    """
    if ask_com_price == bid_com_price:
        return 0.5

    # Check if target is within the COM price range
    min_price = min(bid_com_price, ask_com_price)
    max_price = max(bid_com_price, ask_com_price)

    if target_price < min_price or target_price > max_price:
        return None

    # Linear interpolation: find t where price = bid_com + t * (ask_com - bid_com)
    t = (target_price - bid_com_price) / (ask_com_price - bid_com_price)
    return t


@dataclass
class ComIntersections:
    """Intersection positions of key prices with the COM line."""
    support_position: Optional[float] = None  # 0-1, where SUP crosses COM line
    resistance_position: Optional[float] = None  # 0-1, where RES crosses COM line
    last_price_position: Optional[float] = None  # 0-1, where LAST crosses COM line


def calculate_all_intersections(
    bid_com_price: float,
    ask_com_price: float,
    trend_support: Optional[float],
    trend_resistance: Optional[float],
    last_trade_price: Optional[float],
) -> ComIntersections:
    """
    Calculate all COM line intersection positions.

    These positions indicate where key price levels sit along the bid-ask tension:
    - Position near 0 (left): price is near bid side (bullish for support)
    - Position near 1 (right): price is near ask side (bearish for resistance)

    Args:
        bid_com_price: BID center of mass price
        ask_com_price: ASK center of mass price
        trend_support: Current support level
        trend_resistance: Current resistance level
        last_trade_price: Most recent trade price

    Returns:
        ComIntersections with normalized positions (0-1) for each
    """
    return ComIntersections(
        support_position=calculate_com_intersection(
            bid_com_price, ask_com_price, trend_support
        ) if trend_support else None,
        resistance_position=calculate_com_intersection(
            bid_com_price, ask_com_price, trend_resistance
        ) if trend_resistance else None,
        last_price_position=calculate_com_intersection(
            bid_com_price, ask_com_price, last_trade_price
        ) if last_trade_price else None,
    )


# ============================================================
# Convenience functions for parsing raw data
# ============================================================

def parse_order_book(
    bids: List[List[float]],
    asks: List[List[float]],
) -> Tuple[List[OrderLevel], List[OrderLevel]]:
    """
    Parse raw order book data into OrderLevel objects.

    Args:
        bids: [[price, qty], ...] - highest first
        asks: [[price, qty], ...] - lowest first

    Returns:
        Tuple of (bid_levels, ask_levels)
    """
    bid_levels = [OrderLevel(price=b[0], quantity=b[1]) for b in bids]
    ask_levels = [OrderLevel(price=a[0], quantity=a[1]) for a in asks]
    return bid_levels, ask_levels


def parse_trades(trade_dicts: List[dict]) -> List[Trade]:
    """
    Parse raw trade data into Trade objects.

    Args:
        trade_dicts: List of trade dicts with price, quantity, time, is_buyer_maker

    Returns:
        List of Trade objects
    """
    trades = []
    for t in trade_dicts:
        timestamp = t.get("time") or t.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

        trades.append(Trade(
            price=float(t["price"]),
            quantity=float(t["quantity"]),
            timestamp=timestamp,
            is_buyer_maker=t.get("is_buyer_maker", False),
            trade_id=t.get("trade_id"),
        ))
    return trades
