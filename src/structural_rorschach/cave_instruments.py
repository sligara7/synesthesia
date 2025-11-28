"""
Cave Trader Instrument Panel

Two simple 2D gauges like aircraft instruments:
1. Vertical Gauge: Floor (price+support) ↔ Ceiling (volume+resistance)
2. Horizontal Gauge: Bids (left) ↔ Asks (right)

Design philosophy: Pilots don't render 3D aerodynamics - they read gauges.
The complexity is in the DATA, not the DISPLAY.

Player Controls Reference:
                    BUY (RIGHT)
                        │
        ┌───────────────┼───────────────┐
        │   UP+LEFT     │   UP+RIGHT    │
        │  Reduce long  │   Add long    │
 SHORT ─┼───────────────┼───────────────┼─ LONG
        │  DOWN+LEFT    │  DOWN+RIGHT   │
        │  Add short    │  Cover short  │
        └───────────────┼───────────────┘
                        │
                    SELL (LEFT)

Force Field Reference:
         CEILING: Volume + Resistance
                    ▼
    BIDS ◄────── ROCKET ──────► ASKS
   (left)                      (right)
                    ▲
         FLOOR: Price + Support
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


class ThreatLevel(Enum):
    """How close danger is."""
    CLEAR = 0      # Safe distance
    CAUTION = 1    # Getting close
    WARNING = 2    # Danger zone
    CRITICAL = 3   # Impact imminent


@dataclass
class VerticalReading:
    """Reading from the altitude gauge (floor/ceiling)."""
    # Position 0.0 = on floor, 1.0 = on ceiling
    altitude: float

    # Distance to obstacles (0.0 = touching, 1.0 = far)
    floor_clearance: float
    ceiling_clearance: float

    # Support/Resistance nearby
    support_below: bool = False
    support_distance: float = 1.0  # How many bars until support
    resistance_above: bool = False
    resistance_distance: float = 1.0

    # Threat assessment
    floor_threat: ThreatLevel = ThreatLevel.CLEAR
    ceiling_threat: ThreatLevel = ThreatLevel.CLEAR


@dataclass
class HorizontalReading:
    """Reading from the lateral gauge (bids/asks)."""
    # Position -1.0 = at bid wall, +1.0 = at ask wall, 0.0 = center
    lateral_position: float

    # Wall distances (0.0 = touching, 1.0 = far)
    bid_wall_distance: float
    ask_wall_distance: float

    # Order book depth affects wall "hardness"
    bid_depth: float  # 0-1, how thick the bid wall is
    ask_depth: float  # 0-1, how thick the ask wall is

    # Threat assessment
    bid_threat: ThreatLevel = ThreatLevel.CLEAR
    ask_threat: ThreatLevel = ThreatLevel.CLEAR


@dataclass
class PositionReading:
    """Current trading position status."""
    # Position: -1.0 = max short, +1.0 = max long, 0.0 = flat
    position: float

    # P&L as percentage of account
    unrealized_pnl: float
    realized_pnl: float

    # Risk indicators
    stop_distance: Optional[float] = None  # Distance to stop-loss
    target_distance: Optional[float] = None  # Distance to take-profit


@dataclass
class InstrumentPanel:
    """Complete instrument readings for one moment."""
    vertical: VerticalReading
    horizontal: HorizontalReading
    position: PositionReading

    # Current bar info
    bar_index: int = 0
    bar_progress: float = 0.0  # 0.0 = bar start, 1.0 = bar end

    # Audio cue hints (for the sound system)
    audio_hints: List[str] = field(default_factory=list)


class InstrumentCalculator:
    """
    Calculates instrument readings from raw market data.

    This is where the complexity lives - translating market structure
    into simple gauge readings.
    """

    def __init__(
        self,
        altitude_range: Tuple[float, float] = (0.0, 100.0),
        threat_thresholds: Tuple[float, float, float] = (0.3, 0.15, 0.05)
    ):
        """
        Args:
            altitude_range: Min/max for normalizing price to altitude
            threat_thresholds: Distances for CAUTION, WARNING, CRITICAL
        """
        self.altitude_range = altitude_range
        self.caution_dist, self.warning_dist, self.critical_dist = threat_thresholds

    def _assess_threat(self, clearance: float) -> ThreatLevel:
        """Convert clearance distance to threat level."""
        if clearance <= self.critical_dist:
            return ThreatLevel.CRITICAL
        elif clearance <= self.warning_dist:
            return ThreatLevel.WARNING
        elif clearance <= self.caution_dist:
            return ThreatLevel.CAUTION
        return ThreatLevel.CLEAR

    def calculate_vertical(
        self,
        price: float,
        floor_height: float,
        ceiling_height: float,
        support_levels: Optional[List[Tuple[float, float]]] = None,
        resistance_levels: Optional[List[Tuple[float, float]]] = None
    ) -> VerticalReading:
        """
        Calculate vertical gauge reading.

        Args:
            price: Current price (rocket altitude)
            floor_height: Floor level from price structure
            ceiling_height: Ceiling level from volume
            support_levels: List of (price, bars_away) for supports below
            resistance_levels: List of (price, bars_away) for resistances above
        """
        # Normalize altitude to 0-1 range
        total_height = ceiling_height - floor_height
        if total_height <= 0:
            total_height = 1.0  # Avoid division by zero

        altitude = (price - floor_height) / total_height
        altitude = max(0.0, min(1.0, altitude))

        # Calculate clearances
        floor_clearance = altitude
        ceiling_clearance = 1.0 - altitude

        # Check for nearby support/resistance
        support_below = False
        support_distance = 1.0
        if support_levels:
            for level, bars_away in support_levels:
                if level < price:
                    support_below = True
                    support_distance = min(support_distance, bars_away / 10.0)  # Normalize

        resistance_above = False
        resistance_distance = 1.0
        if resistance_levels:
            for level, bars_away in resistance_levels:
                if level > price:
                    resistance_above = True
                    resistance_distance = min(resistance_distance, bars_away / 10.0)

        return VerticalReading(
            altitude=altitude,
            floor_clearance=floor_clearance,
            ceiling_clearance=ceiling_clearance,
            support_below=support_below,
            support_distance=support_distance,
            resistance_above=resistance_above,
            resistance_distance=resistance_distance,
            floor_threat=self._assess_threat(floor_clearance),
            ceiling_threat=self._assess_threat(ceiling_clearance)
        )

    def calculate_horizontal(
        self,
        bid_depth: float,
        ask_depth: float,
        position_bias: float = 0.0
    ) -> HorizontalReading:
        """
        Calculate horizontal gauge reading.

        Args:
            bid_depth: Normalized bid depth (0-1, 1 = thick wall)
            ask_depth: Normalized ask depth (0-1, 1 = thick wall)
            position_bias: Current position affecting perceived lateral position
        """
        # Wall distance inversely related to depth
        # Thin walls = far (easy to push through)
        # Thick walls = close (hard to push through)
        bid_wall_distance = 1.0 - bid_depth
        ask_wall_distance = 1.0 - ask_depth

        # Lateral position based on relative depths
        # If bids thick and asks thin, we're pushed toward asks
        total_pressure = bid_depth + ask_depth
        if total_pressure > 0:
            lateral_position = (bid_depth - ask_depth) / total_pressure
        else:
            lateral_position = 0.0

        # Position bias affects perceived position
        lateral_position = lateral_position * 0.7 + position_bias * 0.3
        lateral_position = max(-1.0, min(1.0, lateral_position))

        return HorizontalReading(
            lateral_position=lateral_position,
            bid_wall_distance=bid_wall_distance,
            ask_wall_distance=ask_wall_distance,
            bid_depth=bid_depth,
            ask_depth=ask_depth,
            bid_threat=self._assess_threat(bid_wall_distance),
            ask_threat=self._assess_threat(ask_wall_distance)
        )

    def calculate_position(
        self,
        position_size: float,
        max_position: float,
        entry_price: float,
        current_price: float,
        account_value: float,
        stop_price: Optional[float] = None,
        target_price: Optional[float] = None
    ) -> PositionReading:
        """Calculate position gauge reading."""
        # Normalize position to -1 to +1
        if max_position > 0:
            position = position_size / max_position
        else:
            position = 0.0
        position = max(-1.0, min(1.0, position))

        # Calculate P&L
        if position_size != 0 and entry_price > 0:
            pnl_dollars = (current_price - entry_price) * position_size
            unrealized_pnl = pnl_dollars / account_value if account_value > 0 else 0.0
        else:
            unrealized_pnl = 0.0

        # Distance to stop/target
        stop_distance = None
        if stop_price and current_price > 0:
            stop_distance = abs(current_price - stop_price) / current_price

        target_distance = None
        if target_price and current_price > 0:
            target_distance = abs(target_price - current_price) / current_price

        return PositionReading(
            position=position,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=0.0,  # Would need account history
            stop_distance=stop_distance,
            target_distance=target_distance
        )


class ASCIIRenderer:
    """
    Renders instrument panel as ASCII art.

    This is intentionally simple - like early flight instruments.
    The goal is CLARITY, not beauty.
    """

    def __init__(self, width: int = 50, height: int = 20):
        self.width = width
        self.height = height

    def render_vertical_gauge(self, reading: VerticalReading, width: int = 20) -> List[str]:
        """
        Render the altitude gauge (floor/ceiling).

        Example output:
        ╔══════════════════╗
        ║▓▓▓▓▓▓▓▓ CEILING ▓║  <- Volume + Resistance
        ║░░░░░░░░░░░░░░░░░░║
        ║       ═══        ║  <- Resistance level
        ║                  ║
        ║        ◆         ║  <- Rocket
        ║                  ║
        ║       ═══        ║  <- Support level
        ║░░░░░░░░░░░░░░░░░░║
        ║▓▓▓▓▓▓▓▓ FLOOR ▓▓▓║  <- Price + Support
        ╚══════════════════╝
        """
        lines = []
        inner_width = width - 4
        gauge_height = 10

        # Top border
        lines.append("╔" + "═" * (width - 2) + "╗")

        # Ceiling zone (top 2 rows)
        ceiling_char = "▓" if reading.ceiling_threat != ThreatLevel.CLEAR else "░"
        ceiling_label = "CEILING"
        pad = (inner_width - len(ceiling_label)) // 2
        lines.append("║" + ceiling_char * pad + ceiling_label + ceiling_char * (inner_width - pad - len(ceiling_label)) + "║")

        # Calculate rocket position in gauge (0 = bottom, gauge_height-1 = top)
        rocket_row = int(reading.altitude * (gauge_height - 3)) + 1

        # Main gauge area
        for row in range(gauge_height - 2):
            row_from_bottom = gauge_height - 3 - row

            if row_from_bottom == rocket_row:
                # Rocket row
                rocket_pos = inner_width // 2
                content = " " * (rocket_pos - 1) + "◆" + " " * (inner_width - rocket_pos)
            elif reading.resistance_above and row > gauge_height // 2 - 2:
                # Resistance level indicator
                if row == gauge_height // 2 - 1:
                    sr_pos = inner_width // 2 - 2
                    content = " " * sr_pos + "═══" + " " * (inner_width - sr_pos - 3)
                else:
                    content = " " * inner_width
            elif reading.support_below and row < gauge_height // 2:
                # Support level indicator
                if row == gauge_height // 2 - 3:
                    sr_pos = inner_width // 2 - 2
                    content = " " * sr_pos + "═══" + " " * (inner_width - sr_pos - 3)
                else:
                    content = " " * inner_width
            else:
                content = " " * inner_width

            lines.append("║ " + content + " ║")

        # Floor zone (bottom 2 rows)
        floor_char = "▓" if reading.floor_threat != ThreatLevel.CLEAR else "░"
        floor_label = "FLOOR"
        pad = (inner_width - len(floor_label)) // 2
        lines.append("║" + floor_char * pad + floor_label + floor_char * (inner_width - pad - len(floor_label)) + "║")

        # Bottom border
        lines.append("╚" + "═" * (width - 2) + "╝")

        return lines

    def render_horizontal_gauge(self, reading: HorizontalReading, width: int = 40) -> List[str]:
        """
        Render the lateral gauge (bids/asks).

        Example output:
        ┌──────────────────────────────────┐
        │BID▓▓▓░░░░░░◆░░░░░░░▓▓ASK        │
        │███         ◆           ███      │
        └──────────────────────────────────┘
        """
        lines = []
        inner_width = width - 2

        # Top border
        lines.append("┌" + "─" * (width - 2) + "┐")

        # Bid wall visualization (3 chars each)
        bid_chars = int(reading.bid_depth * 3)
        ask_chars = int(reading.ask_depth * 3)

        bid_section = "▓" * bid_chars + "░" * (3 - bid_chars)
        ask_section = "░" * (3 - ask_chars) + "▓" * ask_chars

        # Middle section with rocket
        middle_width = inner_width - 12  # BID(3) + bid_wall(3) + ask_wall(3) + ASK(3)
        center = middle_width // 2
        rocket_offset = int(reading.lateral_position * (center - 1))
        rocket_pos = max(0, min(middle_width - 1, center + rocket_offset))

        middle = " " * rocket_pos + "◆" + " " * (middle_width - rocket_pos - 1)

        main_line = f"BID{bid_section}{middle}{ask_section}ASK"
        lines.append("│" + main_line[:inner_width].ljust(inner_width) + "│")

        # Depth visualization (bar chart style)
        bid_bar = "█" * int(reading.bid_depth * 4)
        ask_bar = "█" * int(reading.ask_depth * 4)
        depth_line = f"{bid_bar:>4}{'◆':^{inner_width - 8}}{ask_bar:<4}"
        lines.append("│" + depth_line[:inner_width].ljust(inner_width) + "│")

        # Bottom border
        lines.append("└" + "─" * (width - 2) + "┘")

        return lines

    def render_position_gauge(self, reading: PositionReading, width: int = 30) -> List[str]:
        """
        Render position and P&L gauge.

        Example output:
        ┌────────────────────────────┐
        │S▓▓░░░░░│░░░░▓▓L  P&L:+2.5%│
        │STOP:1.2%    TARGET:5.0%   │
        └────────────────────────────┘
        """
        lines = []
        inner_width = width - 2

        lines.append("┌" + "─" * (width - 2) + "┐")

        # Position bar (more compact)
        bar_width = 8
        pos_offset = int(reading.position * bar_width)

        short_fill = max(0, -pos_offset)
        long_fill = max(0, pos_offset)

        bar = "▓" * short_fill + "░" * (bar_width - short_fill) + "│" + "░" * (bar_width - long_fill) + "▓" * long_fill

        pnl_str = f"P&L:{reading.unrealized_pnl:+.1%}"
        pos_line = f"S{bar}L {pnl_str}"
        lines.append("│" + pos_line[:inner_width].ljust(inner_width) + "│")

        # Stop/Target line
        stop_str = f"STOP:{reading.stop_distance:.1%}" if reading.stop_distance else ""
        target_str = f"TGT:{reading.target_distance:.1%}" if reading.target_distance else ""
        info_line = f"{stop_str}  {target_str}".strip()
        lines.append("│" + info_line[:inner_width].ljust(inner_width) + "│")

        lines.append("└" + "─" * (width - 2) + "┘")

        return lines

    def render_full_panel(self, panel: InstrumentPanel) -> str:
        """
        Render complete instrument panel.

        Layout:
        ┌─────────────────────────────────────────────────────────┐
        │              CAVE TRADER INSTRUMENTS                    │
        ├──────────────────────┬──────────────────────────────────┤
        │   ALTITUDE GAUGE     │        LATERAL GAUGE             │
        │   ╔═══════════════╗  │  ┌────────────────────────────┐  │
        │   ║    CEILING    ║  │  │BIDS ▓▓░░░░◆░░░░░▓▓ ASKS   │  │
        │   ║               ║  │  │████      ◆        ████     │  │
        │   ║      ◆        ║  │  └────────────────────────────┘  │
        │   ║               ║  │                                  │
        │   ║     FLOOR     ║  │  ┌──────────────────────────────┐│
        │   ╚═══════════════╝  │  │SHORT ░░░░░│░░▓▓▓ LONG       ││
        │                      │  │P&L: +2.5%  STOP: 1.2%       ││
        │   Bar: 42 [████░░]   │  └──────────────────────────────┘│
        └──────────────────────┴──────────────────────────────────┘
        """
        vertical = self.render_vertical_gauge(panel.vertical, width=22)
        horizontal = self.render_horizontal_gauge(panel.horizontal, width=34)
        position = self.render_position_gauge(panel.position, width=34)

        # Build combined output
        lines = []
        lines.append("╔" + "═" * 58 + "╗")
        lines.append("║" + "CAVE TRADER INSTRUMENTS".center(58) + "║")
        lines.append("╠" + "═" * 24 + "╦" + "═" * 33 + "╣")

        # Side by side rendering
        max_rows = max(len(vertical), len(horizontal) + len(position) + 1)

        for i in range(max_rows):
            left = vertical[i] if i < len(vertical) else " " * 22

            if i < len(horizontal):
                right = horizontal[i]
            elif i == len(horizontal):
                right = " " * 34
            elif i - len(horizontal) - 1 < len(position):
                right = position[i - len(horizontal) - 1]
            else:
                right = " " * 34

            lines.append(f"║ {left} ║ {right}║")

        # Bar progress
        progress = int(panel.bar_progress * 6)
        progress_bar = "█" * progress + "░" * (6 - progress)
        bar_info = f"Bar: {panel.bar_index} [{progress_bar}]"
        lines.append("╠" + "═" * 24 + "╩" + "═" * 33 + "╣")
        lines.append("║ " + bar_info.ljust(57) + "║")
        lines.append("╚" + "═" * 58 + "╝")

        return "\n".join(lines)

    def render_compact(self, panel: InstrumentPanel) -> str:
        """
        Ultra-compact single-line status.

        Example: ALT:0.65▲ LAT:-0.2← POS:+0.3L P&L:+1.2%
        """
        # Altitude with direction
        alt_arrow = "▲" if panel.vertical.altitude > 0.5 else "▼"

        # Lateral with direction
        lat_arrow = "→" if panel.horizontal.lateral_position > 0 else "←"

        # Position
        pos_char = "L" if panel.position.position > 0 else "S" if panel.position.position < 0 else "F"

        return (
            f"ALT:{panel.vertical.altitude:.2f}{alt_arrow} "
            f"LAT:{panel.horizontal.lateral_position:+.1f}{lat_arrow} "
            f"POS:{panel.position.position:+.1f}{pos_char} "
            f"P&L:{panel.position.unrealized_pnl:+.1%}"
        )


def demo():
    """Demonstrate the instrument panel."""
    print("Cave Trader Instrument Panel Demo")
    print("=" * 60)

    # Create calculator
    calc = InstrumentCalculator()

    # Simulate some readings
    vertical = calc.calculate_vertical(
        price=105.0,
        floor_height=100.0,
        ceiling_height=115.0,
        support_levels=[(102.0, 3)],
        resistance_levels=[(110.0, 5)]
    )

    horizontal = calc.calculate_horizontal(
        bid_depth=0.7,
        ask_depth=0.3,
        position_bias=0.2
    )

    position = calc.calculate_position(
        position_size=100,
        max_position=500,
        entry_price=103.0,
        current_price=105.0,
        account_value=10000.0,
        stop_price=101.0,
        target_price=110.0
    )

    panel = InstrumentPanel(
        vertical=vertical,
        horizontal=horizontal,
        position=position,
        bar_index=42,
        bar_progress=0.7
    )

    # Render
    renderer = ASCIIRenderer()
    print(renderer.render_full_panel(panel))
    print()
    print("Compact:", renderer.render_compact(panel))

    # Show threat levels
    print()
    print("Threat Assessment:")
    print(f"  Floor: {vertical.floor_threat.name}")
    print(f"  Ceiling: {vertical.ceiling_threat.name}")
    print(f"  Bid Wall: {horizontal.bid_threat.name}")
    print(f"  Ask Wall: {horizontal.ask_threat.name}")


if __name__ == "__main__":
    demo()
