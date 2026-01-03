"""
Order Book Tug-of-War: Adult Trading Game

Visualizes the battle between buyers and sellers through center-of-mass analysis
and pattern recognition. The order book becomes a physics simulation where:

- Each side (bid/ask) has a center of mass based on volume-weighted price
- A "tension line" connects the two centers of mass
- Historical tension lines fade over time
- When a similar slope pattern repeats, those lines brighten (pattern recognition)

THE BATTLEFIELD:
    ┌─────────────────────────────────────────────────────────────┐
    │  ASK SIDE (sellers)                                         │
    │  ████████████████████ $101.50 (1,234)                       │
    │  ██████████████ $101.40 (567)                               │
    │  ████████ $101.30 (123)              ● ASK COM              │
    │  ──────────────────── SPREAD ────────────────────────────── │
    │                          ● BID COM                          │
    │            ████████████ $101.20 (234)                       │
    │  ██████████████████ $101.10 (789)                           │
    │  ██████████████████████ $101.00 (1,567)                     │
    │  BID SIDE (buyers)                                          │
    ├─────────────────────────────────────────────────────────────┤
    │  TENSION HISTORY                                            │
    │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ▲ (current)                 │
    │  ═══════════════════════════      (PATTERN MATCH!)          │
    │  ────────────────────             (recent)                  │
    │  ·················                (fading)                  │
    └─────────────────────────────────────────────────────────────┘

Usage:
    PYTHONPATH=src python -m structural_rorschach.order_book_tug
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from datetime import datetime
from enum import Enum
import math
import time
import random

# Local imports
try:
    from .binance_live import BinanceLiveStream, OrderBookSnapshot
except ImportError:
    try:
        from binance_live import BinanceLiveStream, OrderBookSnapshot
    except ImportError:
        BinanceLiveStream = None
        OrderBookSnapshot = None


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class CenterOfMass:
    """Center of mass for one side of the order book."""
    price: float           # The COM price level
    total_quantity: float  # Sum of quantities used in calculation
    side: str              # "bid" or "ask"
    num_levels: int = 0    # How many price levels contributed

    def __repr__(self) -> str:
        return f"COM({self.side}: ${self.price:.2f}, qty={self.total_quantity:.2f})"


@dataclass
class QuantileLine:
    """A line connecting matching quantiles between bid and ask distributions."""
    percentile: int        # 10, 20, 30, ..., 90
    bid_price: float       # Price at this percentile on bid side
    ask_price: float       # Price at this percentile on ask side

    @property
    def spread(self) -> float:
        """Price difference between ask and bid at this quantile."""
        return self.ask_price - self.bid_price

    def __repr__(self) -> str:
        return f"Q{self.percentile}(bid=${self.bid_price:.2f}, ask=${self.ask_price:.2f})"


@dataclass
class TensionField:
    """
    The full tension field between bid and ask distributions.

    Treats both sides as probability distributions and maps quantiles:
    - 10% bid → 10% ask
    - 20% bid → 20% ask
    - ... etc ...

    This shows how the "shape" of one distribution maps to the other,
    revealing whether tension is uniform or concentrated at certain levels.
    """
    lines: List[QuantileLine]  # Lines at 10%, 20%, ..., 90%
    bid_total: float           # Total bid volume (for normalization info)
    ask_total: float           # Total ask volume

    @property
    def median_line(self) -> Optional[QuantileLine]:
        """The 50% line (equivalent to COM)."""
        for line in self.lines:
            if line.percentile == 50:
                return line
        return None

    @property
    def convergence_ratio(self) -> float:
        """
        Ratio of outer spread to inner spread.

        > 1.0 = lines converge toward center (hourglass shape)
        < 1.0 = lines diverge from center (diamond shape)
        = 1.0 = parallel lines (uniform tension)
        """
        if len(self.lines) < 5:
            return 1.0

        outer_spread = (self.lines[-1].spread + self.lines[0].spread) / 2
        inner_spread = self.lines[len(self.lines) // 2].spread

        if inner_spread <= 0:
            return 1.0
        return outer_spread / inner_spread


@dataclass
class Trade:
    """A single executed trade."""
    price: float
    quantity: float
    timestamp: datetime
    is_buyer_maker: bool = False  # True if buyer was the maker (sell aggressor)


class TradeDistribution:
    """
    Tracks the distribution of actual executed trades.

    This shows WHERE the market is actually transacting, not just
    where orders are waiting. The density at each price level shows
    how frequently that price has been the "meeting point."
    """

    def __init__(self, max_trades: int = 100, num_bins: int = 20):
        self.max_trades = max_trades
        self.num_bins = num_bins
        self.trades: List[Trade] = []

    def add_trade(self, price: float, quantity: float, is_buyer_maker: bool = False):
        """Record a new trade."""
        trade = Trade(
            price=price,
            quantity=quantity,
            timestamp=datetime.now(),
            is_buyer_maker=is_buyer_maker,
        )
        self.trades.append(trade)

        # Trim to max size
        if len(self.trades) > self.max_trades:
            self.trades.pop(0)

    def get_density(self, price_range: Tuple[float, float]) -> List[Tuple[float, float]]:
        """
        Calculate trade density across price bins.

        Returns list of (bin_center_price, density) tuples.
        Density is normalized so max = 1.0.
        """
        if not self.trades or price_range[1] <= price_range[0]:
            return []

        min_price, max_price = price_range
        bin_size = (max_price - min_price) / self.num_bins

        # Count volume in each bin
        bins = [0.0] * self.num_bins

        for trade in self.trades:
            if min_price <= trade.price <= max_price:
                bin_idx = int((trade.price - min_price) / bin_size)
                bin_idx = min(bin_idx, self.num_bins - 1)
                bins[bin_idx] += trade.quantity

        # Normalize
        max_vol = max(bins) if bins else 1.0
        if max_vol <= 0:
            max_vol = 1.0

        result = []
        for i, vol in enumerate(bins):
            bin_center = min_price + (i + 0.5) * bin_size
            density = vol / max_vol
            result.append((bin_center, density))

        return result

    def get_vwap(self) -> Optional[float]:
        """Get volume-weighted average price of recent trades."""
        if not self.trades:
            return None

        total_value = sum(t.price * t.quantity for t in self.trades)
        total_volume = sum(t.quantity for t in self.trades)

        if total_volume <= 0:
            return None
        return total_value / total_volume

    def get_price_range(self) -> Optional[Tuple[float, float]]:
        """Get min/max prices from recent trades."""
        if not self.trades:
            return None

        prices = [t.price for t in self.trades]
        return (min(prices), max(prices))

    @property
    def buy_volume(self) -> float:
        """Volume where buyer was aggressor (taker)."""
        return sum(t.quantity for t in self.trades if not t.is_buyer_maker)

    @property
    def sell_volume(self) -> float:
        """Volume where seller was aggressor (taker)."""
        return sum(t.quantity for t in self.trades if t.is_buyer_maker)

    @property
    def buy_sell_ratio(self) -> float:
        """Ratio of buy to sell aggression (> 1 = more buying pressure)."""
        if self.sell_volume <= 0:
            return 2.0 if self.buy_volume > 0 else 1.0
        return self.buy_volume / self.sell_volume


@dataclass
class TensionSnapshot:
    """A moment-in-time snapshot of bid/ask tension."""
    tick: int
    timestamp: datetime
    bid_com: CenterOfMass
    ask_com: CenterOfMass
    spread: float          # Best ask - best bid

    @property
    def slope_angle(self) -> float:
        """
        Angle of the line from bid_com to ask_com (degrees).

        Normalized by spread to make angles comparable across different price levels.
        Positive angle = ask COM is higher relative to spread
        """
        if self.spread <= 0:
            return 0.0
        price_diff = self.ask_com.price - self.bid_com.price
        normalized = price_diff / self.spread
        return math.degrees(math.atan2(normalized, 1.0))

    @property
    def tension_magnitude(self) -> float:
        """Strength of the tension (distance between COMs relative to spread)."""
        if self.spread <= 0:
            return 0.0
        return abs(self.ask_com.price - self.bid_com.price) / self.spread

    @property
    def bid_dominance(self) -> float:
        """Ratio showing which side is stronger (0-1, 0.5=balanced)."""
        total = self.bid_com.total_quantity + self.ask_com.total_quantity
        if total <= 0:
            return 0.5
        return self.bid_com.total_quantity / total

    @property
    def mid_price(self) -> float:
        """Middle point between bid and ask COM."""
        return (self.bid_com.price + self.ask_com.price) / 2


@dataclass
class PatternMatch:
    """A detected pattern similarity between current and historical tension."""
    historical_tick: int
    similarity_score: float  # 0.0 to 1.0
    angle_difference: float  # Degrees difference in slope

    @property
    def intensity_boost(self) -> float:
        """How much to boost the historical line's intensity."""
        return self.similarity_score ** 2  # Square for sharper contrast


# ============================================================================
# CENTER OF MASS CALCULATOR
# ============================================================================

class COMCalculator:
    """Calculate center of mass for bid and ask sides of order book."""

    def calc_bid_com(self, bids: List[Tuple[float, float]]) -> CenterOfMass:
        """
        Calculate bid side center of mass.

        COM = Sum(price * quantity) / Sum(quantity)

        Args:
            bids: List of (price, quantity) tuples, highest price first
        """
        if not bids:
            return CenterOfMass(price=0.0, total_quantity=0.0, side="bid")

        total_qty = 0.0
        weighted_sum = 0.0

        for price, qty in bids:
            price_f = float(price)
            qty_f = float(qty)
            weighted_sum += price_f * qty_f
            total_qty += qty_f

        if total_qty <= 0:
            return CenterOfMass(price=float(bids[0][0]), total_quantity=0.0, side="bid")

        com_price = weighted_sum / total_qty
        return CenterOfMass(
            price=com_price,
            total_quantity=total_qty,
            side="bid",
            num_levels=len(bids)
        )

    def calc_ask_com(self, asks: List[Tuple[float, float]]) -> CenterOfMass:
        """
        Calculate ask side center of mass.

        Args:
            asks: List of (price, quantity) tuples, lowest price first
        """
        if not asks:
            return CenterOfMass(price=0.0, total_quantity=0.0, side="ask")

        total_qty = 0.0
        weighted_sum = 0.0

        for price, qty in asks:
            price_f = float(price)
            qty_f = float(qty)
            weighted_sum += price_f * qty_f
            total_qty += qty_f

        if total_qty <= 0:
            return CenterOfMass(price=float(asks[0][0]), total_quantity=0.0, side="ask")

        com_price = weighted_sum / total_qty
        return CenterOfMass(
            price=com_price,
            total_quantity=total_qty,
            side="ask",
            num_levels=len(asks)
        )


# ============================================================================
# QUANTILE MAPPER (CDF-based distribution mapping)
# ============================================================================

class QuantileMapper:
    """
    Maps order book sides as probability distributions using CDF.

    The key insight: treat volume at each price level as probability mass.
    Normalize so total = 1.0, compute CDF, then find quantile prices.

    For bids: CDF goes from highest price (0%) to lowest price (100%)
              because we accumulate from the "top" of the bid book
    For asks: CDF goes from lowest price (0%) to highest price (100%)
              because we accumulate from the "bottom" of the ask book

    This means:
    - 10% bid = price where 10% of bid volume is at or above this price
    - 10% ask = price where 10% of ask volume is at or below this price
    """

    def __init__(self, percentiles: List[int] = None):
        self.percentiles = percentiles or [10, 20, 30, 40, 50, 60, 70, 80, 90]

    def calc_tension_field(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]]
    ) -> TensionField:
        """
        Calculate the full tension field between bid and ask distributions.

        Returns quantile lines connecting matching percentiles.
        """
        bid_quantiles = self._calc_quantile_prices(bids, side="bid")
        ask_quantiles = self._calc_quantile_prices(asks, side="ask")

        lines = []
        for pct in self.percentiles:
            if pct in bid_quantiles and pct in ask_quantiles:
                lines.append(QuantileLine(
                    percentile=pct,
                    bid_price=bid_quantiles[pct],
                    ask_price=ask_quantiles[pct],
                ))

        bid_total = sum(float(q) for _, q in bids) if bids else 0.0
        ask_total = sum(float(q) for _, q in asks) if asks else 0.0

        return TensionField(
            lines=lines,
            bid_total=bid_total,
            ask_total=ask_total,
        )

    def _calc_quantile_prices(
        self,
        levels: List[Tuple[float, float]],
        side: str
    ) -> Dict[int, float]:
        """
        Calculate prices at each percentile for one side.

        For bids: sorted highest first, CDF accumulates downward
        For asks: sorted lowest first, CDF accumulates upward
        """
        if not levels:
            return {}

        # Convert to float and sort appropriately
        sorted_levels = [(float(p), float(q)) for p, q in levels]

        if side == "bid":
            # Bids: highest price first (already sorted this way from exchange)
            sorted_levels.sort(key=lambda x: x[0], reverse=True)
        else:
            # Asks: lowest price first (already sorted this way from exchange)
            sorted_levels.sort(key=lambda x: x[0])

        # Calculate total volume
        total = sum(q for _, q in sorted_levels)
        if total <= 0:
            return {}

        # Build CDF
        cumulative = 0.0
        cdf_points = []  # List of (price, cumulative_percentage)

        for price, qty in sorted_levels:
            cumulative += qty
            pct = (cumulative / total) * 100
            cdf_points.append((price, pct))

        # Find prices at target percentiles using linear interpolation
        quantiles = {}
        for target_pct in self.percentiles:
            price = self._interpolate_quantile(cdf_points, target_pct)
            if price is not None:
                quantiles[target_pct] = price

        return quantiles

    def _interpolate_quantile(
        self,
        cdf_points: List[Tuple[float, float]],
        target_pct: float
    ) -> Optional[float]:
        """
        Find the price at a given percentile using linear interpolation.

        cdf_points: List of (price, cumulative_pct) sorted by accumulation order
        """
        if not cdf_points:
            return None

        # Find bracketing points
        prev_price, prev_pct = cdf_points[0][0], 0.0

        for price, pct in cdf_points:
            if pct >= target_pct:
                # Interpolate between prev and current
                if pct == prev_pct:
                    return price
                ratio = (target_pct - prev_pct) / (pct - prev_pct)
                return prev_price + ratio * (price - prev_price)
            prev_price, prev_pct = price, pct

        # Target is beyond 100%, return last price
        return cdf_points[-1][0]


# ============================================================================
# TENSION HISTORY
# ============================================================================

class TensionHistory:
    """Manages the historical record of tension snapshots."""

    def __init__(self, max_history: int = 200):
        self.snapshots: List[TensionSnapshot] = []
        self.max_history = max_history

    def add(self, snapshot: TensionSnapshot):
        """Add a new snapshot, maintaining max_history limit."""
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.max_history:
            self.snapshots.pop(0)

    def get_recent(self, n: int) -> List[TensionSnapshot]:
        """Get the N most recent snapshots."""
        return self.snapshots[-n:] if n < len(self.snapshots) else self.snapshots[:]

    def calculate_opacity(self, tick: int, current_tick: int) -> float:
        """
        Calculate base opacity for a historical line.

        Older lines fade: opacity = 1.0 - (age / max_history)
        """
        age = current_tick - tick
        if age <= 0:
            return 1.0
        base_opacity = max(0.05, 1.0 - (age / self.max_history))
        return base_opacity

    def clear(self):
        """Clear all history."""
        self.snapshots.clear()


# ============================================================================
# SLOPE PATTERN MATCHER
# ============================================================================

class SlopePatternMatcher:
    """
    Detect when current slope pattern matches historical patterns.

    The "slope" is the angle of the line connecting bid_com to ask_com.
    When a similar angle repeats, it indicates a recognizable market structure.
    """

    def __init__(
        self,
        angle_tolerance: float = 5.0,     # Degrees
        min_match_score: float = 0.6,     # Minimum similarity threshold
        min_age: int = 5,                 # Minimum ticks old to be considered
    ):
        self.angle_tolerance = angle_tolerance
        self.min_match_score = min_match_score
        self.min_age = min_age

    def find_matches(
        self,
        current: TensionSnapshot,
        history: List[TensionSnapshot]
    ) -> List[PatternMatch]:
        """
        Find historical snapshots with similar slope angles.

        Returns matches sorted by similarity (best first).
        """
        matches = []

        for h in history:
            # Skip very recent snapshots
            age = current.tick - h.tick
            if age < self.min_age:
                continue

            score = self.calc_similarity_score(current, h)
            if score >= self.min_match_score:
                angle_diff = abs(current.slope_angle - h.slope_angle)
                matches.append(PatternMatch(
                    historical_tick=h.tick,
                    similarity_score=score,
                    angle_difference=angle_diff
                ))

        # Sort by similarity (best first)
        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return matches

    def calc_similarity_score(
        self,
        current: TensionSnapshot,
        historical: TensionSnapshot
    ) -> float:
        """
        Calculate how similar two tension snapshots are.

        Factors:
        - Angle similarity (primary): How close are the slopes?
        - Magnitude similarity: How similar is the tension strength?
        - Balance similarity: Is bid/ask dominance similar?

        Weighted formula:
            score = 0.6 * angle_sim + 0.2 * magnitude_sim + 0.2 * balance_sim
        """
        # Angle similarity (0-1)
        angle_diff = abs(current.slope_angle - historical.slope_angle)
        angle_sim = max(0.0, 1.0 - (angle_diff / 45.0))  # 45 degrees = very different

        # Magnitude similarity (0-1)
        mag_diff = abs(current.tension_magnitude - historical.tension_magnitude)
        mag_sim = max(0.0, 1.0 - mag_diff)

        # Balance similarity (0-1)
        balance_diff = abs(current.bid_dominance - historical.bid_dominance)
        balance_sim = max(0.0, 1.0 - (balance_diff * 2))  # Scale: 0.5 diff = 0 sim

        # Weighted combination
        score = 0.6 * angle_sim + 0.2 * mag_sim + 0.2 * balance_sim
        return score


# ============================================================================
# TUG RENDERER
# ============================================================================

class TugRenderer:
    """
    Rich ASCII/curses renderer for the tug-of-war visualization.

    Uses Unicode block characters for histograms and varying line thickness
    for pattern matching intensity.
    """

    # Unicode characters for histogram bars (full to empty)
    BLOCKS = ['█', '▉', '▊', '▋', '▌', '▍', '▎', '▏', ' ']

    # Line characters by intensity (faintest to boldest)
    LINES = ['·', '╌', '─', '═', '━']

    # Markers
    COM_MARKER = '●'
    ARROW_UP = '▲'
    ARROW_DOWN = '▼'
    ARROW_LEFT = '◀'
    ARROW_RIGHT = '▶'

    def __init__(self, width: int = 70, histogram_width: int = 25):
        self.width = width
        self.histogram_width = histogram_width

    def render_histogram_bar(
        self,
        quantity: float,
        max_qty: float,
        width: int,
        align_right: bool = False
    ) -> str:
        """
        Render a single histogram bar using Unicode blocks.

        Args:
            quantity: The value to represent
            max_qty: Maximum value for scaling
            width: Total width in characters
            align_right: If True, bar extends from right to left
        """
        if max_qty <= 0 or quantity <= 0:
            return ' ' * width

        fill_ratio = min(1.0, quantity / max_qty)
        full_blocks = int(fill_ratio * width)
        fractional = (fill_ratio * width) - full_blocks

        # Select fractional block character (0-7 maps to blocks 0-7)
        frac_idx = min(7, int((1 - fractional) * 8)) if fractional > 0 else 8
        fractional_char = self.BLOCKS[frac_idx] if full_blocks < width else ''

        bar = self.BLOCKS[0] * full_blocks
        if fractional_char and len(bar) < width:
            bar += fractional_char

        # Pad to width
        bar = bar[:width]
        bar += ' ' * (width - len(bar))

        if align_right:
            bar = bar[::-1]  # Reverse for right alignment

        return bar

    def get_line_char(self, intensity: float) -> str:
        """Get line character based on intensity (0-1)."""
        if intensity >= 0.9:
            return self.LINES[4]  # ━ (bold)
        elif intensity >= 0.7:
            return self.LINES[3]  # ═
        elif intensity >= 0.4:
            return self.LINES[2]  # ─
        elif intensity >= 0.15:
            return self.LINES[1]  # ╌
        else:
            return self.LINES[0]  # ·

    def render_tension_line(
        self,
        snapshot: TensionSnapshot,
        width: int,
        opacity: float,
        pattern_boost: float = 0.0
    ) -> str:
        """
        Render a single tension line with opacity and pattern boost.

        The line position reflects bid_dominance (more bids = line shifts left).
        """
        effective_intensity = min(1.0, opacity + pattern_boost)
        line_char = self.get_line_char(effective_intensity)

        # Position based on bid dominance (0.5 = center)
        center = width // 2
        offset = int((snapshot.bid_dominance - 0.5) * width * 0.8)

        # Build line with COM markers
        line = [' '] * width

        # Draw the main tension line (centered portion)
        line_start = max(0, center - 15 + offset)
        line_end = min(width, center + 15 + offset)

        for i in range(line_start, line_end):
            line[i] = line_char

        # Mark bid COM position (left side of line)
        bid_pos = max(0, line_start)
        if bid_pos < width:
            line[bid_pos] = self.COM_MARKER

        # Mark ask COM position (right side of line)
        ask_pos = min(width - 1, line_end - 1)
        if ask_pos >= 0:
            line[ask_pos] = self.COM_MARKER

        return ''.join(line)

    def render_tension_field(
        self,
        field: TensionField,
        width: int,
        price_range: Tuple[float, float],  # (min_price, max_price)
    ) -> List[str]:
        """
        Render the tension field showing all quantile lines.

        This shows how the bid and ask distributions map to each other,
        with lines connecting matching percentiles.

        Visual layout:
            BID SIDE          |  SPREAD  |          ASK SIDE
            ────────90%───────┼──────────┼───────90%────────
            ──────80%─────────┼──────────┼─────80%──────────
            ────70%───────────┼──────────┼───70%────────────
            ...               |          |               ...
        """
        lines = []

        if not field.lines or price_range[1] <= price_range[0]:
            return ["  No tension field data"]

        min_price, max_price = price_range
        price_span = max_price - min_price

        # Calculate column widths
        col_width = (width - 10) // 2  # Leave space for center marker

        def price_to_col(price: float, col_w: int) -> int:
            """Map price to column position."""
            if price_span <= 0:
                return col_w // 2
            normalized = (price - min_price) / price_span
            return int(normalized * (col_w - 1))

        # Header
        lines.append("  TENSION FIELD (quantile mapping: bid distribution → ask distribution)")
        lines.append("")

        # Render each quantile line
        for qline in field.lines:
            # Calculate positions
            bid_col = price_to_col(qline.bid_price, col_width)
            ask_col = price_to_col(qline.ask_price, col_width)

            # Build the line
            row = [' '] * width

            # Left side (bid): marker at bid_col, line extends right
            bid_start = 2
            bid_end = bid_start + col_width

            # Draw bid side line from bid_col to center
            for i in range(bid_start + bid_col, bid_end):
                row[i] = '─'

            # Bid percentile marker
            if bid_start + bid_col < width:
                row[bid_start + bid_col] = '○'

            # Center divider
            center = width // 2
            row[center - 1] = '│'
            row[center] = '│'
            row[center + 1] = '│'

            # Right side (ask): line extends from center to ask_col
            ask_start = center + 2
            ask_end = ask_start + col_width

            # Draw ask side line from center to ask_col
            for i in range(ask_start, min(ask_start + ask_col + 1, ask_end)):
                row[i] = '─'

            # Ask percentile marker
            if ask_start + ask_col < width:
                row[ask_start + ask_col] = '●'

            # Percentile label
            label = f"{qline.percentile:2d}%"
            if qline.percentile == 50:
                label = "50% (median)"
                row[bid_start + bid_col] = '◆'  # Special marker for median
                row[min(ask_start + ask_col, width - 1)] = '◆'

            lines.append(''.join(row) + f"  {label}")

        # Summary stats
        lines.append("")
        conv = field.convergence_ratio
        if conv > 1.1:
            shape = "HOURGLASS (lines converge at center)"
        elif conv < 0.9:
            shape = "DIAMOND (lines diverge at center)"
        else:
            shape = "PARALLEL (uniform tension)"

        lines.append(f"  Shape: {shape}  Convergence: {conv:.2f}")
        lines.append(f"  Bid volume: {field.bid_total:,.0f}  Ask volume: {field.ask_total:,.0f}")

        return lines

    def render_full_display(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        current_tension: Optional[TensionSnapshot],
        history: TensionHistory,
        pattern_matches: List[PatternMatch],
        current_tick: int,
        symbol: str = "BTCUSDT"
    ) -> str:
        """Render the complete display."""
        lines = []

        # Build pattern match lookup for quick intensity boost
        match_lookup: Dict[int, float] = {}
        for m in pattern_matches:
            if m.historical_tick not in match_lookup:
                match_lookup[m.historical_tick] = m.intensity_boost
            else:
                match_lookup[m.historical_tick] = max(
                    match_lookup[m.historical_tick],
                    m.intensity_boost
                )

        # Header
        bid_dom = current_tension.bid_dominance if current_tension else 0.5
        dom_pct = int(bid_dom * 100)
        dom_label = "BULLS" if bid_dom > 0.5 else "BEARS" if bid_dom < 0.5 else "BALANCED"

        lines.append(f"═══ ORDER BOOK TUG OF WAR ═══  {symbol}  {dom_label}: {dom_pct}%")
        lines.append("─" * self.width)

        # Order book section
        lines.extend(self._render_order_book(bids, asks, current_tension))

        lines.append("─" * self.width)

        # Tension history section
        lines.append("  TENSION HISTORY (pattern matches highlighted)")
        lines.append("")

        # Render recent tension lines (most recent first)
        num_to_show = 15
        recent = history.get_recent(num_to_show)

        for idx, snapshot in enumerate(reversed(recent)):
            # Calculate opacity based on position in displayed list (not total history)
            # First line (most recent) = 1.0, last line = 0.1
            position_opacity = 1.0 - (idx / num_to_show) * 0.9

            # Also factor in age for pattern-matched old lines
            age_opacity = history.calculate_opacity(snapshot.tick, current_tick)

            # Use minimum to ensure old lines still fade even if shown
            base_opacity = min(position_opacity, age_opacity)

            boost = match_lookup.get(snapshot.tick, 0.0)

            # Add indicator for current or pattern match
            is_current = (idx == 0)  # First in reversed list is most recent
            if is_current:
                indicator = f" {self.ARROW_UP} NOW"
            elif boost > 0.3:
                indicator = f" ★ PATTERN ({boost:.0%})"
            else:
                indicator = ""

            line = self.render_tension_line(snapshot, self.width - 15, base_opacity, boost)
            lines.append(f"  {line}{indicator}")

        lines.append("")
        lines.append("─" * self.width)

        # Stats
        if current_tension:
            angle = current_tension.slope_angle
            mag = current_tension.tension_magnitude
            num_matches = len(pattern_matches)
            lines.append(f"  Angle: {angle:+.1f}°  Tension: {mag:.2f}  Patterns: {num_matches}")

        lines.append("  [Q] Quit  [R] Reset  [SPACE] Pause")

        return '\n'.join(lines)

    def _render_order_book(
        self,
        bids: List[Tuple[float, float]],
        asks: List[Tuple[float, float]],
        tension: Optional[TensionSnapshot]
    ) -> List[str]:
        """Render the order book histogram section."""
        lines = []

        # Find max quantity for scaling
        all_qtys = [float(q) for _, q in bids] + [float(q) for _, q in asks]
        max_qty = max(all_qtys) if all_qtys else 1.0

        # Render asks (sellers) - top of book, lowest ask first
        lines.append("  ASK SIDE (sellers pushing down)")

        # Show top 5 ask levels (reverse order - highest first for display)
        ask_display = list(reversed(asks[:5])) if asks else []
        for price, qty in ask_display:
            bar = self.render_histogram_bar(float(qty), max_qty, self.histogram_width, align_right=True)
            qty_str = f"{float(qty):,.0f}"
            price_str = f"${float(price):,.2f}"

            # Check if this level is near ask COM
            is_com = tension and abs(float(price) - tension.ask_com.price) < tension.spread * 0.3
            com_mark = f" {self.COM_MARKER} ASK COM" if is_com else ""

            lines.append(f"  {bar} {price_str} ({qty_str}){com_mark}")

        # Spread line
        if tension:
            spread_str = f"SPREAD: ${tension.spread:.4f}"
        else:
            spread_str = "SPREAD"
        pad = (self.width - len(spread_str) - 4) // 2
        lines.append(f"  {'─' * pad} {spread_str} {'─' * pad}")

        # Render bids (buyers) - bottom of book, highest bid first
        bid_display = bids[:5] if bids else []
        for price, qty in bid_display:
            bar = self.render_histogram_bar(float(qty), max_qty, self.histogram_width)
            qty_str = f"{float(qty):,.0f}"
            price_str = f"${float(price):,.2f}"

            # Check if this level is near bid COM
            is_com = tension and abs(float(price) - tension.bid_com.price) < tension.spread * 0.3
            com_mark = f" {self.COM_MARKER} BID COM" if is_com else ""

            lines.append(f"  {bar} {price_str} ({qty_str}){com_mark}")

        lines.append("  BID SIDE (buyers pushing up)")

        return lines


# ============================================================================
# SIMULATED ORDER BOOK (for demo/testing)
# ============================================================================

class SimulatedOrderBook:
    """
    Generates realistic-ish order book data for testing without Binance.

    Simulates market regimes that shift over time, creating recognizable
    patterns that the pattern matcher can detect.
    """

    def __init__(
        self,
        base_price: float = 100.0,
        num_levels: int = 10,
        tick_size: float = 0.01,
    ):
        self.base_price = base_price
        self.num_levels = num_levels
        self.tick_size = tick_size

        self.mid_price = base_price
        self.bid_bias = 0.0  # -1 to 1, affects which side has more volume

        # Market regime (creates recurring patterns)
        self.regime = 0  # 0=balanced, 1=bid_heavy, 2=ask_heavy, 3=volatile
        self.regime_ticks = 0
        self.regime_duration = random.randint(15, 40)

        self._tick = 0

    def generate_snapshot(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], float]:
        """
        Generate a simulated order book snapshot.

        Returns: (bids, asks, spread)
        """
        self._tick += 1
        self.regime_ticks += 1

        # Switch regime periodically (creates repeating patterns)
        if self.regime_ticks >= self.regime_duration:
            old_regime = self.regime
            # Tend to cycle through regimes, but with some randomness
            if random.random() < 0.6:
                self.regime = (self.regime + 1) % 4
            else:
                self.regime = random.randint(0, 3)
            self.regime_ticks = 0
            self.regime_duration = random.randint(15, 40)

        # Apply regime effects
        if self.regime == 0:  # Balanced
            target_bias = 0.0
            volatility = 1.0
        elif self.regime == 1:  # Bid heavy (bulls)
            target_bias = 0.6
            volatility = 0.8
        elif self.regime == 2:  # Ask heavy (bears)
            target_bias = -0.6
            volatility = 0.8
        else:  # Volatile
            target_bias = random.uniform(-0.8, 0.8)
            volatility = 2.0

        # Move bias toward target with some noise
        self.bid_bias += (target_bias - self.bid_bias) * 0.15 + random.gauss(0, 0.08 * volatility)
        self.bid_bias = max(-1.0, min(1.0, self.bid_bias))

        # Random walk the mid price
        self.mid_price += random.gauss(0, self.tick_size * 2 * volatility)

        # Generate spread (wider in volatile regime)
        base_spread = self.tick_size * random.uniform(1, 3) * (1.5 if self.regime == 3 else 1.0)

        # Generate bid levels
        bids = []
        bid_price = self.mid_price - base_spread / 2
        for i in range(self.num_levels):
            price = bid_price - i * self.tick_size
            # Volume decreases with distance, bid bias adds volume
            base_vol = max(10, 100 - i * 8 + random.gauss(0, 15 * volatility))
            bias_adj = base_vol * (1 + self.bid_bias * 0.6)
            volume = max(1, bias_adj)
            bids.append((round(price, 4), round(volume, 2)))

        # Generate ask levels
        asks = []
        ask_price = self.mid_price + base_spread / 2
        for i in range(self.num_levels):
            price = ask_price + i * self.tick_size
            # Volume decreases with distance, ask bias (negative bid_bias) adds volume
            base_vol = max(10, 100 - i * 8 + random.gauss(0, 15 * volatility))
            bias_adj = base_vol * (1 - self.bid_bias * 0.6)
            volume = max(1, bias_adj)
            asks.append((round(price, 4), round(volume, 2)))

        spread = asks[0][0] - bids[0][0] if bids and asks else 0.0

        return bids, asks, spread


# ============================================================================
# MAIN GAME CLASS
# ============================================================================

class OrderBookTug:
    """
    Order Book Tug-of-War game.

    Visualizes the battle between buyers and sellers through:
    - Center of mass calculation
    - Tension line connecting bid/ask COMs
    - Pattern recognition highlighting repeated tension slopes
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        max_history: int = 200,
        angle_tolerance: float = 5.0,
        use_live: bool = True,
        tld: str = "us",
    ):
        self.symbol = symbol
        self.use_live = use_live and BinanceLiveStream is not None
        self.tld = tld

        # Components
        self.com_calculator = COMCalculator()
        self.quantile_mapper = QuantileMapper()  # NEW: CDF-based distribution mapping
        self.history = TensionHistory(max_history=max_history)
        self.pattern_matcher = SlopePatternMatcher(angle_tolerance=angle_tolerance)
        self.renderer = TugRenderer()

        # State
        self.tick = 0
        self.current_tension: Optional[TensionSnapshot] = None
        self.current_field: Optional[TensionField] = None  # NEW: Full tension field
        self.pattern_matches: List[PatternMatch] = []
        self.paused = False
        self.show_field: bool = True  # Toggle tension field display

        # Current order book data
        self.current_bids: List[Tuple[float, float]] = []
        self.current_asks: List[Tuple[float, float]] = []

        # Binance stream or simulator
        self._stream: Optional[BinanceLiveStream] = None
        self._simulator: Optional[SimulatedOrderBook] = None

    def connect(self) -> bool:
        """Connect to data source (live Binance or simulator)."""
        if self.use_live:
            try:
                self._stream = BinanceLiveStream(
                    symbol=self.symbol,
                    tld=self.tld,
                )
                self._stream.start()
                return True
            except Exception as e:
                print(f"Could not connect to Binance: {e}")
                print("Falling back to simulated data...")
                self.use_live = False

        # Fall back to simulator
        self._simulator = SimulatedOrderBook(base_price=100.0)
        return True

    def disconnect(self):
        """Disconnect from data source."""
        if self._stream:
            self._stream.stop()
            self._stream = None

    def update(self) -> bool:
        """
        Fetch latest data and update state.

        Returns True if update was successful.
        """
        if self.paused:
            return True

        # Get order book data
        if self._stream:
            book = self._stream.get_order_book()
            if book:
                self.current_bids = book.bids
                self.current_asks = book.asks
                spread = book.spread
            else:
                return False  # No data yet
        elif self._simulator:
            self.current_bids, self.current_asks, spread = self._simulator.generate_snapshot()
        else:
            return False

        # Calculate centers of mass
        bid_com = self.com_calculator.calc_bid_com(self.current_bids)
        ask_com = self.com_calculator.calc_ask_com(self.current_asks)

        # Create tension snapshot
        if spread <= 0 and self.current_bids and self.current_asks:
            spread = float(self.current_asks[0][0]) - float(self.current_bids[0][0])

        snapshot = TensionSnapshot(
            tick=self.tick,
            timestamp=datetime.now(),
            bid_com=bid_com,
            ask_com=ask_com,
            spread=max(0.0001, spread),  # Avoid division by zero
        )

        self.current_tension = snapshot

        # Calculate full tension field (quantile mapping)
        self.current_field = self.quantile_mapper.calc_tension_field(
            self.current_bids,
            self.current_asks
        )

        # Find pattern matches with history
        self.pattern_matches = self.pattern_matcher.find_matches(
            snapshot,
            self.history.snapshots
        )

        # Add to history
        self.history.add(snapshot)
        self.tick += 1

        return True

    def render(self) -> str:
        """Render the current state."""
        lines = []

        # Main display
        main_display = self.renderer.render_full_display(
            bids=self.current_bids,
            asks=self.current_asks,
            current_tension=self.current_tension,
            history=self.history,
            pattern_matches=self.pattern_matches,
            current_tick=self.tick,
            symbol=self.symbol,
        )
        lines.append(main_display)

        # Add tension field if enabled
        if self.show_field and self.current_field and self.current_bids and self.current_asks:
            lines.append("")
            lines.append("─" * 70)

            # Get price range for rendering
            all_prices = [float(p) for p, _ in self.current_bids] + [float(p) for p, _ in self.current_asks]
            if all_prices:
                price_range = (min(all_prices), max(all_prices))
                field_lines = self.renderer.render_tension_field(
                    self.current_field,
                    width=70,
                    price_range=price_range,
                )
                lines.extend(field_lines)

            lines.append("")
            lines.append("  [F] Toggle field view  [Q] Quit  [R] Reset  [SPACE] Pause")

        return '\n'.join(lines)

    def reset(self):
        """Reset game state."""
        self.tick = 0
        self.history.clear()
        self.current_tension = None
        self.current_field = None
        self.pattern_matches = []
        self.paused = False


# ============================================================================
# INTERACTIVE TERMINAL MODE
# ============================================================================

def clear_screen():
    """Clear the terminal screen."""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


def run_interactive(symbol: str = "BTCUSDT", use_live: bool = True):
    """
    Run the game in interactive terminal mode.
    """
    import sys
    import select

    # Terminal handling for raw keyboard input
    try:
        import termios
        import tty
        HAS_TERMIOS = True
    except ImportError:
        HAS_TERMIOS = False

    print("═══ ORDER BOOK TUG OF WAR ═══")
    print(f"Symbol: {symbol}")
    print(f"Mode: {'LIVE' if use_live else 'SIMULATED'}")
    print()
    print("Connecting...")

    game = OrderBookTug(symbol=symbol, use_live=use_live)
    if not game.connect():
        print("Failed to connect!")
        return

    print("Connected! Starting in 2 seconds...")
    time.sleep(2)

    # Game loop timing
    last_update = time.time()
    update_interval = 0.5  # Update every 500ms

    # Set up terminal for raw input
    if HAS_TERMIOS:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

    try:
        running = True
        while running:
            current_time = time.time()

            # Check for input (non-blocking)
            if HAS_TERMIOS and select.select([sys.stdin], [], [], 0.0)[0]:
                ch = sys.stdin.read(1).upper()

                if ch == 'Q' or ch == '\x03':  # Q or Ctrl+C
                    running = False
                elif ch == 'R':
                    game.reset()
                elif ch == ' ':
                    game.paused = not game.paused
                elif ch == 'F':
                    game.show_field = not game.show_field

            # Update at fixed interval
            if current_time - last_update >= update_interval:
                last_update = current_time

                game.update()

                # Render
                clear_screen()
                print(game.render())

                if game.paused:
                    print("\n  *** PAUSED - Press SPACE to resume ***")

            time.sleep(0.01)  # Small sleep to prevent CPU spinning

    except KeyboardInterrupt:
        pass

    finally:
        if HAS_TERMIOS:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        game.disconnect()

    print("\nThanks for playing Order Book Tug of War!")


def demo():
    """Run a quick demo with simulated data."""
    print("═══ ORDER BOOK TUG OF WAR - DEMO ═══")
    print()

    game = OrderBookTug(use_live=False)
    game.connect()

    # Run a few ticks
    for i in range(20):
        game.update()

    # Show final state
    print(game.render())
    print()
    print("This was a demo with simulated data.")
    print("Run with --live flag for real Binance data.")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Order Book Tug of War")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair (default: BTCUSDT)")
    parser.add_argument("--live", action="store_true", help="Use live Binance data")
    parser.add_argument("--demo", action="store_true", help="Run quick demo")

    args = parser.parse_args()

    if args.demo:
        demo()
    else:
        run_interactive(symbol=args.symbol, use_live=args.live)


if __name__ == "__main__":
    main()
