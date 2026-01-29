"""
Real-time S/R Penetration Signal Detector.

Detects when support/resistance levels enter the bid-ask COM tension field,
generating trading signals in real-time during market data streaming.
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import AsyncGenerator, Callable, Dict, List, Optional
from uuid import UUID

from ..calculations import (
    Candle,
    CenterOfMass,
    OrderLevel,
    Trade,
    TrendState,
    aggregate_candles,
    calculate_com,
    calculate_trend_state,
    parse_order_book,
)
from ..models.signals import SignalType, PenetrationSignal
from ..protocols import DepthUpdate, TradeUpdate

logger = logging.getLogger(__name__)


@dataclass
class SignalState:
    """
    Maintains state for signal detection on a single symbol.

    Tracks rolling windows of trades and candles for trend analysis,
    plus previous COM positions for detecting threshold crossings.
    """

    symbol: str
    recording_id: Optional[UUID] = None

    # Rolling trade window for candle building (last 5 min)
    trades: deque = field(default_factory=lambda: deque(maxlen=10000))

    # Candle state
    current_candle: Optional[Candle] = None
    candles: deque = field(default_factory=lambda: deque(maxlen=100))
    candle_interval_seconds: int = 30

    # Trend state from candles
    trend_state: Optional[TrendState] = None
    prev_center: Optional[float] = None
    prev_upper_mid: Optional[float] = None
    prev_lower_mid: Optional[float] = None
    prev_dynamic_support: Optional[float] = None
    prev_dynamic_resistance: Optional[float] = None
    prev_trend_support: Optional[float] = None
    prev_trend_resistance: Optional[float] = None

    # COM state
    bid_com: Optional[CenterOfMass] = None
    ask_com: Optional[CenterOfMass] = None

    # Previous signal state (for detecting changes)
    prev_support_in_range: bool = False
    prev_resistance_in_range: bool = False
    prev_price_position: Optional[float] = None

    # EMA smoothed ratio for confluence detection
    smoothed_ratio: Optional[float] = None
    ema_period: int = 30

    # Last trade price
    last_trade_price: Optional[float] = None


@dataclass
class DetectedSignal:
    """A detected trading signal ready to be persisted or streamed."""

    timestamp: datetime
    symbol: str
    signal_type: SignalType
    support_position: Optional[float]
    resistance_position: Optional[float]
    last_price_position: Optional[float]
    bid_com_price: float
    ask_com_price: float
    trend_support: Optional[float]
    trend_resistance: Optional[float]
    last_trade_price: Optional[float]
    bid_active_qty: float
    ask_active_qty: float
    bulls_bears_ratio: float
    support_in_range: bool
    resistance_in_range: bool
    confidence: float
    recording_id: Optional[UUID] = None

    def to_penetration_signal(self) -> PenetrationSignal:
        """Convert to database model."""
        return PenetrationSignal(
            time=self.timestamp,
            symbol=self.symbol,
            signal_type=self.signal_type,
            support_position=self.support_position,
            resistance_position=self.resistance_position,
            last_price_position=self.last_price_position,
            bid_com_price=self.bid_com_price,
            ask_com_price=self.ask_com_price,
            trend_support=self.trend_support,
            trend_resistance=self.trend_resistance,
            last_trade_price=self.last_trade_price,
            bid_active_qty=self.bid_active_qty,
            ask_active_qty=self.ask_active_qty,
            bulls_bears_ratio=self.bulls_bears_ratio,
            support_in_range=self.support_in_range,
            resistance_in_range=self.resistance_in_range,
            confidence=self.confidence,
            recording_id=self.recording_id,
        )


class SignalDetector:
    """
    Detects S/R penetration signals in real-time from market data streams.

    Maintains state for each symbol and generates signals when:
    1. Resistance enters the COM tension field (bullish)
    2. Support enters the COM tension field (bearish)
    3. Penetration signal + bulls/bears ratio confluence (strong signals)
    4. Price crosses from one side of COM to the other (breakouts)
    """

    def __init__(
        self,
        candle_interval_seconds: int = 30,
        ema_period: int = 30,
        on_signal: Optional[Callable[[DetectedSignal], None]] = None,
    ):
        """
        Initialize the signal detector.

        Args:
            candle_interval_seconds: Interval for building candles
            ema_period: EMA period for smoothing bulls/bears ratio
            on_signal: Optional callback invoked for each detected signal
        """
        self.candle_interval_seconds = candle_interval_seconds
        self.ema_period = ema_period
        self.on_signal = on_signal

        # State per symbol
        self._states: Dict[str, SignalState] = {}

        # Signal buffer for batch persistence
        self._signal_buffer: List[DetectedSignal] = []
        self._buffer_lock = asyncio.Lock()

    def get_or_create_state(
        self, symbol: str, recording_id: Optional[UUID] = None
    ) -> SignalState:
        """Get or create state for a symbol."""
        if symbol not in self._states:
            self._states[symbol] = SignalState(
                symbol=symbol,
                recording_id=recording_id,
                candle_interval_seconds=self.candle_interval_seconds,
                ema_period=self.ema_period,
            )
        return self._states[symbol]

    def process_depth(
        self, symbol: str, update: DepthUpdate, recording_id: Optional[UUID] = None
    ) -> List[DetectedSignal]:
        """
        Process an order book depth update and detect signals.

        Args:
            symbol: Trading pair symbol
            update: Depth update from exchange
            recording_id: Optional recording ID if from recorded data

        Returns:
            List of detected signals (may be empty)
        """
        state = self.get_or_create_state(symbol, recording_id)
        signals = []

        # Parse order book
        bid_levels = [OrderLevel(price=b[0], quantity=b[1]) for b in update.bids[:20]]
        ask_levels = [OrderLevel(price=a[0], quantity=a[1]) for a in update.asks[:20]]

        if not bid_levels or not ask_levels:
            return signals

        # Calculate COM
        bid_com = calculate_com(bid_levels, "bid")
        ask_com = calculate_com(ask_levels, "ask")

        state.bid_com = bid_com
        state.ask_com = ask_com

        # Calculate bulls/bears ratio
        total_qty = bid_com.active_quantity + ask_com.active_quantity
        if total_qty > 0:
            ratio = (bid_com.active_quantity - ask_com.active_quantity) / total_qty
        else:
            ratio = 0.0

        # EMA smooth the ratio
        if state.smoothed_ratio is None:
            state.smoothed_ratio = ratio
        else:
            alpha = 2 / (self.ema_period + 1)
            state.smoothed_ratio = alpha * ratio + (1 - alpha) * state.smoothed_ratio

        # Check for signals if we have trend state
        if state.trend_state is not None:
            detected = self._detect_signals(state, update.timestamp)
            signals.extend(detected)

        return signals

    def process_trade(
        self, symbol: str, trade: TradeUpdate, recording_id: Optional[UUID] = None
    ) -> List[DetectedSignal]:
        """
        Process a trade update and update candle/trend state.

        Args:
            symbol: Trading pair symbol
            trade: Trade update from exchange
            recording_id: Optional recording ID if from recorded data

        Returns:
            List of detected signals (may be empty, usually after candle close)
        """
        state = self.get_or_create_state(symbol, recording_id)
        signals = []

        # Update last trade price
        state.last_trade_price = trade.price

        # Add to trade window
        trade_obj = Trade(
            price=trade.price,
            quantity=trade.quantity,
            timestamp=trade.timestamp,
            is_buyer_maker=trade.is_buyer_maker,
            trade_id=trade.trade_id,
        )
        state.trades.append(trade_obj)

        # Calculate candle bucket
        ts = trade.timestamp.timestamp()
        bucket_start = (ts // self.candle_interval_seconds) * self.candle_interval_seconds

        if state.current_candle is None:
            # Start first candle
            state.current_candle = Candle(
                timestamp=datetime.fromtimestamp(bucket_start, tz=timezone.utc),
                open=trade.price,
                high=trade.price,
                low=trade.price,
                close=trade.price,
                volume=trade.quantity,
                trade_count=1,
            )
        elif state.current_candle.timestamp.timestamp() != bucket_start:
            # Candle closed, update trend state
            self._close_candle(state)

            # Start new candle
            state.current_candle = Candle(
                timestamp=datetime.fromtimestamp(bucket_start, tz=timezone.utc),
                open=trade.price,
                high=trade.price,
                low=trade.price,
                close=trade.price,
                volume=trade.quantity,
                trade_count=1,
            )

            # Check for signals after trend update
            if state.bid_com is not None and state.ask_com is not None:
                detected = self._detect_signals(state, trade.timestamp)
                signals.extend(detected)
        else:
            # Update current candle
            state.current_candle.high = max(state.current_candle.high, trade.price)
            state.current_candle.low = min(state.current_candle.low, trade.price)
            state.current_candle.close = trade.price
            state.current_candle.volume += trade.quantity
            state.current_candle.trade_count += 1

        return signals

    def _close_candle(self, state: SignalState):
        """Close current candle and update trend state."""
        if state.current_candle is None:
            return

        # Add to candle history
        state.candles.append(state.current_candle)

        # Calculate new trend state
        state.trend_state = calculate_trend_state(
            state.current_candle,
            prev_center=state.prev_center,
            prev_upper_mid=state.prev_upper_mid,
            prev_lower_mid=state.prev_lower_mid,
            prev_dynamic_support=state.prev_dynamic_support,
            prev_dynamic_resistance=state.prev_dynamic_resistance,
            prev_trend_support=state.prev_trend_support,
            prev_trend_resistance=state.prev_trend_resistance,
        )

        # Update prev values for next candle
        state.prev_center = state.trend_state.center
        state.prev_upper_mid = state.trend_state.upper_mid
        state.prev_lower_mid = state.trend_state.lower_mid
        state.prev_dynamic_support = state.trend_state.dynamic_support
        state.prev_dynamic_resistance = state.trend_state.dynamic_resistance
        state.prev_trend_support = state.trend_state.trend_support
        state.prev_trend_resistance = state.trend_state.trend_resistance

    def _calculate_com_intersection(
        self, bid_com_price: float, ask_com_price: float, target_price: float
    ) -> tuple[Optional[float], bool]:
        """
        Calculate where target_price intersects the BID-ASK COM line.

        Returns:
            (position, is_in_range): position is 0-1, is_in_range if within COM span
        """
        if ask_com_price == bid_com_price:
            return 0.5, True

        t = (target_price - bid_com_price) / (ask_com_price - bid_com_price)

        if 0 <= t <= 1:
            return t, True
        else:
            return max(0, min(1, t)), False

    def _detect_signals(
        self, state: SignalState, timestamp: datetime
    ) -> List[DetectedSignal]:
        """
        Detect signals based on current state.

        Returns list of detected signals.
        """
        signals = []

        if state.bid_com is None or state.ask_com is None or state.trend_state is None:
            return signals

        bid_com_price = state.bid_com.price
        ask_com_price = state.ask_com.price
        trend = state.trend_state

        # Calculate intersection positions
        support_pos, support_in_range = self._calculate_com_intersection(
            bid_com_price, ask_com_price, trend.trend_support
        )
        resistance_pos, resistance_in_range = self._calculate_com_intersection(
            bid_com_price, ask_com_price, trend.trend_resistance
        )
        last_price_pos = None
        if state.last_trade_price is not None:
            last_price_pos, _ = self._calculate_com_intersection(
                bid_com_price, ask_com_price, state.last_trade_price
            )

        # Calculate bulls/bears ratio
        total_qty = state.bid_com.active_quantity + state.ask_com.active_quantity
        if total_qty > 0:
            ratio = (state.bid_com.active_quantity - state.ask_com.active_quantity) / total_qty
        else:
            ratio = 0.0

        # Base signal data
        base_data = dict(
            timestamp=timestamp,
            symbol=state.symbol,
            support_position=support_pos,
            resistance_position=resistance_pos,
            last_price_position=last_price_pos,
            bid_com_price=bid_com_price,
            ask_com_price=ask_com_price,
            trend_support=trend.trend_support,
            trend_resistance=trend.trend_resistance,
            last_trade_price=state.last_trade_price,
            bid_active_qty=state.bid_com.active_quantity,
            ask_active_qty=state.ask_com.active_quantity,
            bulls_bears_ratio=ratio,
            support_in_range=support_in_range,
            resistance_in_range=resistance_in_range,
            recording_id=state.recording_id,
        )

        # Detect penetration signals (state changes)
        # Bullish: resistance just entered range (wasn't before, is now)
        if resistance_in_range and not state.prev_resistance_in_range and not support_in_range:
            # Check for confluence with ratio
            smoothed = state.smoothed_ratio or 0
            if smoothed > 0:
                signal_type = SignalType.STRONG_BULLISH
                confidence = min(1.0, 0.6 + abs(smoothed) * 0.4)
            else:
                signal_type = SignalType.BULLISH
                confidence = 0.5 + (1 - resistance_pos) * 0.3  # Higher if further from ask

            signals.append(DetectedSignal(
                signal_type=signal_type,
                confidence=confidence,
                **base_data,
            ))

        # Bearish: support just entered range (wasn't before, is now)
        if support_in_range and not state.prev_support_in_range and not resistance_in_range:
            smoothed = state.smoothed_ratio or 0
            if smoothed < 0:
                signal_type = SignalType.STRONG_BEARISH
                confidence = min(1.0, 0.6 + abs(smoothed) * 0.4)
            else:
                signal_type = SignalType.BEARISH
                confidence = 0.5 + support_pos * 0.3  # Higher if further from bid

            signals.append(DetectedSignal(
                signal_type=signal_type,
                confidence=confidence,
                **base_data,
            ))

        # Breakout detection: price crossed from one side to the other
        if last_price_pos is not None and state.prev_price_position is not None:
            prev = state.prev_price_position
            curr = last_price_pos

            # Crossed from bid side (< 0.3) to ask side (> 0.7)
            if prev < 0.3 and curr > 0.7:
                signals.append(DetectedSignal(
                    signal_type=SignalType.BREAKOUT_UP,
                    confidence=0.8,
                    **base_data,
                ))

            # Crossed from ask side (> 0.7) to bid side (< 0.3)
            elif prev > 0.7 and curr < 0.3:
                signals.append(DetectedSignal(
                    signal_type=SignalType.BREAKOUT_DOWN,
                    confidence=0.8,
                    **base_data,
                ))

        # Update previous state
        state.prev_support_in_range = support_in_range
        state.prev_resistance_in_range = resistance_in_range
        state.prev_price_position = last_price_pos

        # Invoke callback for each signal
        for signal in signals:
            if self.on_signal:
                try:
                    self.on_signal(signal)
                except Exception as e:
                    logger.error(f"Error in signal callback: {e}")

        return signals

    async def buffer_signal(self, signal: DetectedSignal):
        """Add signal to buffer for batch persistence."""
        async with self._buffer_lock:
            self._signal_buffer.append(signal)

    async def flush_signals(self) -> List[DetectedSignal]:
        """Flush and return buffered signals."""
        async with self._buffer_lock:
            signals = self._signal_buffer
            self._signal_buffer = []
            return signals

    def reset_state(self, symbol: Optional[str] = None):
        """Reset detector state for a symbol or all symbols."""
        if symbol:
            self._states.pop(symbol, None)
        else:
            self._states.clear()
