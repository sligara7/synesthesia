"""
Egg Trader Levels - Progressive Difficulty Trading Game

A leveled trading game where each level introduces new concepts:

LEVEL 1: GREEN EGGS ONLY (Long positions)
    ┌─────────────────────────────────────────────────┐
    │  LEFT SIDE              RIGHT SIDE              │
    │  ══════════              ══════════             │
    │  ● Sources              ● Sinks                 │
    │  (buy longs)            (sell longs)            │
    │                                                 │
    │  Pick up green eggs on left (open long)         │
    │  Deposit green eggs on right (close long)       │
    │  Profit = exit_price - entry_price              │
    └─────────────────────────────────────────────────┘

LEVEL 2: RED EGGS TOO (Short positions)
    ┌─────────────────────────────────────────────────┐
    │  LEFT SIDE              RIGHT SIDE              │
    │  ══════════              ══════════             │
    │  ● Sources              ○ Sources               │
    │  ○ Sinks                ● Sinks                 │
    │                                                 │
    │  Green: buy left (open) → sell right (close)    │
    │  Red: buy right (open) → sell left (close)      │
    └─────────────────────────────────────────────────┘

LEVEL 3: POSITION REVERSAL (Close + open in one move)
    ┌─────────────────────────────────────────────────┐
    │  LEFT SIDE              RIGHT SIDE              │
    │  ══════════              ══════════             │
    │  ● Sources              ○ Sources               │
    │  ○ Sinks                ● Sinks                 │
    │                                                 │
    │  Can close position AND open opposite in one    │
    │  trip. Example: deposit 3● + pickup 2○ = -2 net │
    └─────────────────────────────────────────────────┘

LEVEL 4: FUTURES (Timed positions)
    - Eggs with expiration timers
    - Must close before expiry or auto-settle

LEVEL 5: MULTIPLE PAIRS
    - Purple eggs = BTC/ETH
    - Orange eggs = BTC/XRP
    - Arbitrage opportunities

LEVEL 6: ADVANCED ORDERS
    - Stop-loss (auto-close at price)
    - Take-profit
    - Trailing stops
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import random


class EggColor(Enum):
    """Egg colors representing position types."""
    GREEN = "green"   # Long position (profit when price goes up)
    RED = "red"       # Short position (profit when price goes down)
    PURPLE = "purple" # BTC/ETH pair (Level 5)
    ORANGE = "orange" # BTC/XRP pair (Level 5)


class ZoneType(Enum):
    """Types of zones on the trading floor."""
    SOURCE = "source"  # Where you pick up eggs (open position)
    SINK = "sink"      # Where you deposit eggs (close position)


@dataclass
class EggZone:
    """A zone where eggs can be picked up or deposited."""
    color: EggColor
    zone_type: ZoneType
    price_level: int      # Which price row (0-9)
    quantity: int = 0     # Current eggs available/capacity
    max_quantity: int = 5


@dataclass
class CarriedPosition:
    """Eggs you're currently carrying (open position)."""
    color: EggColor
    quantity: int
    entry_price: float
    entry_tick: int


@dataclass
class CompletedTrade:
    """A closed trade with P&L."""
    color: EggColor
    quantity: int
    entry_price: float
    exit_price: float
    pnl: float
    tick: int
    bonus_type: Optional[str] = None  # "breakout", "bounce", or None
    bonus_multiplier: float = 1.0


class LevelType(Enum):
    """Type of support/resistance level."""
    SUPPORT = "support"      # Price bounced UP from here (swing low)
    RESISTANCE = "resistance"  # Price bounced DOWN from here (swing high)


@dataclass
class PriceLevel:
    """A detected support or resistance level."""
    level_type: LevelType
    price: float
    strength: int = 1       # How many times price touched this level
    last_touch_tick: int = 0
    broken: bool = False    # True if price broke through


@dataclass
class PriceBar:
    """A single price bar for history tracking."""
    tick: int
    price: float
    high: float
    low: float


@dataclass
class TradingFloor:
    """The trading floor layout for a level."""
    left_zones: List[EggZone]   # Zones on left side
    right_zones: List[EggZone]  # Zones on right side
    num_levels: int = 10
    base_price: float = 100.0
    price_step: float = 1.0


class DifficultyLevel(Enum):
    """Game difficulty levels."""
    LEVEL_1 = 1    # Green eggs only (long positions)
    LEVEL_1B = 15  # Red eggs only (short positions) - teaches shorting alone
    LEVEL_2 = 2    # Green + Red on opposite sides
    LEVEL_3 = 3    # Position reversal
    LEVEL_4 = 4    # Futures
    LEVEL_5 = 5    # Multiple pairs
    LEVEL_6 = 6    # Advanced orders


def create_floor_for_level(level: DifficultyLevel, num_levels: int = 10) -> TradingFloor:
    """Create a trading floor layout appropriate for the difficulty level."""
    left_zones = []
    right_zones = []

    if level == DifficultyLevel.LEVEL_1:
        # Level 1: Green sources on left, green sinks on right
        # LONG ONLY: Pick up green on left (buy), deposit on right (sell)
        # Want price to GO UP between buy and sell
        for i in range(num_levels):
            left_zones.append(EggZone(
                color=EggColor.GREEN,
                zone_type=ZoneType.SOURCE,
                price_level=i,
                quantity=random.randint(2, 5),
            ))
            right_zones.append(EggZone(
                color=EggColor.GREEN,
                zone_type=ZoneType.SINK,
                price_level=i,
                quantity=0,  # Sinks start empty (capacity)
            ))

    elif level == DifficultyLevel.LEVEL_1B:
        # Level 1B: Red sources on right, red sinks on left
        # SHORT ONLY: Pick up red on right (sell/short), deposit on left (buy back)
        # Want price to GO DOWN between short and buy-back
        for i in range(num_levels):
            right_zones.append(EggZone(
                color=EggColor.RED,
                zone_type=ZoneType.SOURCE,
                price_level=i,
                quantity=random.randint(2, 5),
            ))
            left_zones.append(EggZone(
                color=EggColor.RED,
                zone_type=ZoneType.SINK,
                price_level=i,
                quantity=0,
            ))

    elif level == DifficultyLevel.LEVEL_2:
        # Level 2: Green sources left, red sources right
        #          Green sinks right, red sinks left
        for i in range(num_levels):
            # Left side: green sources, red sinks
            left_zones.append(EggZone(
                color=EggColor.GREEN,
                zone_type=ZoneType.SOURCE,
                price_level=i,
                quantity=random.randint(2, 5),
            ))
            left_zones.append(EggZone(
                color=EggColor.RED,
                zone_type=ZoneType.SINK,
                price_level=i,
                quantity=0,
            ))
            # Right side: red sources, green sinks
            right_zones.append(EggZone(
                color=EggColor.RED,
                zone_type=ZoneType.SOURCE,
                price_level=i,
                quantity=random.randint(2, 5),
            ))
            right_zones.append(EggZone(
                color=EggColor.GREEN,
                zone_type=ZoneType.SINK,
                price_level=i,
                quantity=0,
            ))

    elif level.value >= DifficultyLevel.LEVEL_3.value:
        # Level 3+: Both sources and sinks on both sides
        for i in range(num_levels):
            # Left side: green sources + sinks, red sources + sinks
            left_zones.append(EggZone(
                color=EggColor.GREEN,
                zone_type=ZoneType.SOURCE,
                price_level=i,
                quantity=random.randint(2, 5),
            ))
            left_zones.append(EggZone(
                color=EggColor.GREEN,
                zone_type=ZoneType.SINK,
                price_level=i,
                quantity=0,
            ))
            left_zones.append(EggZone(
                color=EggColor.RED,
                zone_type=ZoneType.SOURCE,
                price_level=i,
                quantity=random.randint(1, 3),
            ))
            left_zones.append(EggZone(
                color=EggColor.RED,
                zone_type=ZoneType.SINK,
                price_level=i,
                quantity=0,
            ))
            # Right side: same layout
            right_zones.append(EggZone(
                color=EggColor.GREEN,
                zone_type=ZoneType.SOURCE,
                price_level=i,
                quantity=random.randint(1, 3),
            ))
            right_zones.append(EggZone(
                color=EggColor.GREEN,
                zone_type=ZoneType.SINK,
                price_level=i,
                quantity=0,
            ))
            right_zones.append(EggZone(
                color=EggColor.RED,
                zone_type=ZoneType.SOURCE,
                price_level=i,
                quantity=random.randint(2, 5),
            ))
            right_zones.append(EggZone(
                color=EggColor.RED,
                zone_type=ZoneType.SINK,
                price_level=i,
                quantity=0,
            ))

    return TradingFloor(
        left_zones=left_zones,
        right_zones=right_zones,
        num_levels=num_levels,
    )


class Side(Enum):
    """Which side of the floor."""
    LEFT = "left"
    RIGHT = "right"


class EggTraderLevels:
    """
    The leveled egg trading game.

    Player navigates a 2D grid:
    - Vertical axis = price levels ($100-$109)
    - Horizontal axis = left side vs right side

    Each side has egg zones (sources to pick up, sinks to deposit).
    """

    def __init__(
        self,
        level: DifficultyLevel = DifficultyLevel.LEVEL_1,
        num_levels: int = 10,
        base_price: float = 100.0,
    ):
        self.level = level
        self.num_levels = num_levels
        self.base_price = base_price

        # Create floor layout
        self.floor = create_floor_for_level(level, num_levels)

        # Price state
        self.price_index = num_levels // 2  # Current market price

        # Player state
        self.player_price_level = num_levels // 2
        self.player_side = Side.LEFT
        self.player_zone_index = 0  # Which zone within the side

        # Positions
        self.carrying: Optional[CarriedPosition] = None
        self.trades: List[CompletedTrade] = []
        self.total_pnl: float = 0.0
        self.cash: float = 10000.0

        # Game state
        self.tick = 0
        self.game_over = False

        # Price history for support/resistance detection
        self.price_history: List[PriceBar] = []
        self.max_history = 50  # Keep last 50 bars

        # Support and resistance levels
        self.support_levels: List[PriceLevel] = []
        self.resistance_levels: List[PriceLevel] = []

        # Bonus tracking
        self.last_bonus_message: Optional[str] = None
        self.bonus_message_tick: int = 0

        # Constants for S/R detection
        self.sr_tolerance = 0.5  # Price within 0.5 of level counts as "at" level
        self.breakout_bonus = 2.0  # 2x multiplier for breakout trades
        self.bounce_bonus = 1.5   # 1.5x multiplier for bounce trades

        # Initialize with some historical S/R levels for better gameplay
        self._init_sr_levels()

    def _init_sr_levels(self):
        """Initialize with some obvious S/R levels for better gameplay."""
        # Add support near the bottom and resistance near the top
        # These represent "historical" levels the player can trade around

        # Support at lower prices
        self.support_levels.append(PriceLevel(
            level_type=LevelType.SUPPORT,
            price=self.base_price + 2,  # $102
            strength=2,
            last_touch_tick=0,
        ))

        # Resistance at higher prices
        self.resistance_levels.append(PriceLevel(
            level_type=LevelType.RESISTANCE,
            price=self.base_price + 7,  # $107
            strength=2,
            last_touch_tick=0,
        ))

    def get_price_at_level(self, level: int) -> float:
        """Get the price at a given level index."""
        return self.base_price + level * self.floor.price_step

    @property
    def current_market_price(self) -> float:
        """Current market price."""
        return self.get_price_at_level(self.price_index)

    @property
    def player_price(self) -> float:
        """Price at player's current level."""
        return self.get_price_at_level(self.player_price_level)

    def get_zones_at_level(self, side: Side, price_level: int) -> List[EggZone]:
        """Get all zones on a side at a price level."""
        zones = self.floor.left_zones if side == Side.LEFT else self.floor.right_zones
        return [z for z in zones if z.price_level == price_level]

    def get_current_zone(self) -> Optional[EggZone]:
        """Get the zone the player is currently at."""
        zones = self.get_zones_at_level(self.player_side, self.player_price_level)
        if self.player_zone_index < len(zones):
            return zones[self.player_zone_index]
        return None

    # === Player Movement ===

    def input_up(self):
        """Move up one price level."""
        if self.player_price_level < self.num_levels - 1:
            self.player_price_level += 1
            self.player_zone_index = 0

    def input_down(self):
        """Move down one price level."""
        if self.player_price_level > 0:
            self.player_price_level -= 1
            self.player_zone_index = 0

    def input_left(self):
        """Move to left side."""
        self.player_side = Side.LEFT
        self.player_zone_index = 0

    def input_right(self):
        """Move to right side."""
        self.player_side = Side.RIGHT
        self.player_zone_index = 0

    def input_cycle_zone(self):
        """Cycle through zones at current position (TAB key)."""
        zones = self.get_zones_at_level(self.player_side, self.player_price_level)
        if zones:
            self.player_zone_index = (self.player_zone_index + 1) % len(zones)

    # === Trading Actions ===

    def input_pickup(self, quantity: int = 1) -> bool:
        """Pick up eggs from current zone (open position)."""
        if self.carrying is not None:
            return False  # Already carrying

        zone = self.get_current_zone()
        if zone is None or zone.zone_type != ZoneType.SOURCE:
            return False  # Not at a source

        qty = min(quantity, zone.quantity)
        if qty <= 0:
            return False  # No eggs available

        # Take eggs
        zone.quantity -= qty
        self.carrying = CarriedPosition(
            color=zone.color,
            quantity=qty,
            entry_price=self.player_price,
            entry_tick=self.tick,
        )
        return True

    def input_deposit(self) -> bool:
        """Deposit eggs at current zone (close position)."""
        if self.carrying is None:
            return False

        zone = self.get_current_zone()
        if zone is None or zone.zone_type != ZoneType.SINK:
            return False  # Not at a sink

        if zone.color != self.carrying.color:
            return False  # Wrong color sink

        # Calculate P&L
        exit_price = self.player_price
        entry_price = self.carrying.entry_price
        qty = self.carrying.quantity

        if self.carrying.color == EggColor.GREEN:
            # Long: profit if exit > entry
            base_pnl = (exit_price - entry_price) * qty
        else:
            # Short: profit if exit < entry
            base_pnl = (entry_price - exit_price) * qty

        # Calculate bonus for support/resistance plays
        multiplier, bonus_type = self._calculate_trade_bonus(
            entry_price, exit_price, self.carrying.color
        )

        # Apply bonus (only to profitable trades)
        if base_pnl > 0 and multiplier > 1.0:
            pnl = base_pnl * multiplier
            if bonus_type == "breakout":
                self.last_bonus_message = f"BREAKOUT BONUS! {multiplier}x = +${pnl:.2f}"
            elif bonus_type == "bounce":
                self.last_bonus_message = f"BOUNCE BONUS! {multiplier}x = +${pnl:.2f}"
            self.bonus_message_tick = self.tick
        else:
            pnl = base_pnl
            bonus_type = None
            multiplier = 1.0

        # Record trade
        trade = CompletedTrade(
            color=self.carrying.color,
            quantity=qty,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            tick=self.tick,
            bonus_type=bonus_type,
            bonus_multiplier=multiplier,
        )
        self.trades.append(trade)
        self.total_pnl += pnl
        self.cash += pnl

        self.carrying = None
        return True

    def input_reverse(self, new_quantity: int = 0) -> bool:
        """
        Level 3+: Close current position AND open opposite.

        Example: Carrying 3 green, at red source with 5 available.
        - Deposit 3 green (close long)
        - Pick up 2 red (open short)
        - Net position: -2 (short)
        """
        if self.level.value < DifficultyLevel.LEVEL_3.value:
            return False  # Not available at this level

        if self.carrying is None:
            return False

        # Find sink for current position
        zones = self.get_zones_at_level(self.player_side, self.player_price_level)
        sink = None
        source = None
        opposite_color = EggColor.RED if self.carrying.color == EggColor.GREEN else EggColor.GREEN

        for z in zones:
            if z.zone_type == ZoneType.SINK and z.color == self.carrying.color:
                sink = z
            if z.zone_type == ZoneType.SOURCE and z.color == opposite_color:
                source = z

        if sink is None or source is None:
            return False  # Need both sink and source

        if new_quantity > 0 and source.quantity < new_quantity:
            return False  # Not enough opposite eggs

        # Close current position
        exit_price = self.player_price
        entry_price = self.carrying.entry_price
        qty = self.carrying.quantity

        if self.carrying.color == EggColor.GREEN:
            pnl = (exit_price - entry_price) * qty
        else:
            pnl = (entry_price - exit_price) * qty

        trade = CompletedTrade(
            color=self.carrying.color,
            quantity=qty,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            tick=self.tick,
        )
        self.trades.append(trade)
        self.total_pnl += pnl
        self.cash += pnl

        # Open opposite position
        if new_quantity > 0:
            source.quantity -= new_quantity
            self.carrying = CarriedPosition(
                color=opposite_color,
                quantity=new_quantity,
                entry_price=self.player_price,
                entry_tick=self.tick,
            )
        else:
            self.carrying = None

        return True

    # === Game Loop ===

    def tick_update(self):
        """Advance game by one tick."""
        self.tick += 1

        # Store previous price for breakout detection
        prev_price = self.current_market_price

        # Price movement with momentum and mean reversion
        # More volatile to create S/R levels
        if random.random() < 0.3:
            # 30% chance of larger move (momentum/breakout)
            move = random.choice([-2, -1, 1, 2])
        else:
            # 70% chance of small move or stay
            move = random.choice([-1, 0, 0, 0, 1])

        # Slight mean reversion to middle (keeps price in range)
        middle = self.num_levels // 2
        if self.price_index > middle + 2:
            move -= 0.5  # Bias down if too high
        elif self.price_index < middle - 2:
            move += 0.5  # Bias up if too low

        move = int(round(move))
        self.price_index = max(0, min(self.num_levels - 1, self.price_index + move))

        new_price = self.current_market_price

        # Record price bar
        bar = PriceBar(
            tick=self.tick,
            price=new_price,
            high=max(prev_price, new_price),
            low=min(prev_price, new_price),
        )
        self.price_history.append(bar)
        if len(self.price_history) > self.max_history:
            self.price_history.pop(0)

        # Detect support/resistance levels
        self._detect_support_resistance()

        # Check for breakouts
        self._check_breakouts(prev_price, new_price)

        # Replenish sources occasionally
        self._replenish_sources()

    def _detect_support_resistance(self):
        """
        Detect support and resistance levels from price history.

        A swing LOW (support) is when price goes down, then reverses up.
        A swing HIGH (resistance) is when price goes up, then reverses down.
        """
        if len(self.price_history) < 5:
            return

        # Look at recent price action
        recent = self.price_history[-5:]

        # Detect swing low (support): price made a low then bounced
        # Pattern: down, down, LOW, up, up
        if len(recent) >= 3:
            for i in range(1, len(recent) - 1):
                prev_bar = recent[i - 1]
                curr_bar = recent[i]
                next_bar = recent[i + 1]

                # Swing low: lower than neighbors
                if curr_bar.low < prev_bar.low and curr_bar.low < next_bar.low:
                    self._add_support_level(curr_bar.low, curr_bar.tick)

                # Swing high: higher than neighbors
                if curr_bar.high > prev_bar.high and curr_bar.high > next_bar.high:
                    self._add_resistance_level(curr_bar.high, curr_bar.tick)

    def _add_support_level(self, price: float, tick: int):
        """Add or strengthen a support level."""
        # Check if we already have a support near this price
        for level in self.support_levels:
            if abs(level.price - price) <= self.sr_tolerance:
                level.strength += 1
                level.last_touch_tick = tick
                return

        # New support level
        self.support_levels.append(PriceLevel(
            level_type=LevelType.SUPPORT,
            price=price,
            strength=1,
            last_touch_tick=tick,
        ))

        # Keep only the strongest/most recent levels
        self.support_levels = sorted(
            self.support_levels,
            key=lambda x: (x.strength, -x.last_touch_tick),
            reverse=True
        )[:5]

    def _add_resistance_level(self, price: float, tick: int):
        """Add or strengthen a resistance level."""
        for level in self.resistance_levels:
            if abs(level.price - price) <= self.sr_tolerance:
                level.strength += 1
                level.last_touch_tick = tick
                return

        self.resistance_levels.append(PriceLevel(
            level_type=LevelType.RESISTANCE,
            price=price,
            strength=1,
            last_touch_tick=tick,
        ))

        self.resistance_levels = sorted(
            self.resistance_levels,
            key=lambda x: (x.strength, -x.last_touch_tick),
            reverse=True
        )[:5]

    def _check_breakouts(self, prev_price: float, new_price: float):
        """
        Check if price broke through support or resistance.
        Mark the level as broken for visual feedback.
        """
        # Check resistance breakouts (price went UP through resistance)
        for level in self.resistance_levels:
            if not level.broken and prev_price <= level.price < new_price:
                level.broken = True
                self.last_bonus_message = f"BREAKOUT! Price broke resistance at ${level.price:.0f}!"
                self.bonus_message_tick = self.tick

        # Check support breakouts (price went DOWN through support)
        for level in self.support_levels:
            if not level.broken and prev_price >= level.price > new_price:
                level.broken = True
                self.last_bonus_message = f"BREAKDOWN! Price broke support at ${level.price:.0f}!"
                self.bonus_message_tick = self.tick

    def get_nearby_levels(self, price: float) -> Tuple[Optional[PriceLevel], Optional[PriceLevel]]:
        """
        Get the nearest support below and resistance above current price.
        Returns (nearest_support, nearest_resistance).
        """
        nearest_support = None
        nearest_resistance = None

        for level in self.support_levels:
            if not level.broken and level.price < price:
                if nearest_support is None or level.price > nearest_support.price:
                    nearest_support = level

        for level in self.resistance_levels:
            if not level.broken and level.price > price:
                if nearest_resistance is None or level.price < nearest_resistance.price:
                    nearest_resistance = level

        return nearest_support, nearest_resistance

    def _calculate_trade_bonus(self, entry_price: float, exit_price: float, color: EggColor) -> Tuple[float, Optional[str]]:
        """
        Calculate bonus multiplier based on support/resistance.

        Returns (multiplier, bonus_type) where bonus_type is:
        - "breakout": Rode momentum through S/R level
        - "bounce": Correctly faded at S/R level
        - None: Normal trade
        """
        # Check if trade crossed a resistance level (for longs)
        if color == EggColor.GREEN:  # Long position
            for level in self.resistance_levels:
                # Breakout: bought below resistance, sold above it
                if entry_price < level.price <= exit_price:
                    return self.breakout_bonus, "breakout"
                # Bounce: sold near resistance (within tolerance)
                if abs(exit_price - level.price) <= self.sr_tolerance and exit_price > entry_price:
                    return self.bounce_bonus, "bounce"

            for level in self.support_levels:
                # Bounce: bought near support and it held
                if abs(entry_price - level.price) <= self.sr_tolerance and exit_price > entry_price:
                    return self.bounce_bonus, "bounce"

        else:  # Short position (RED)
            for level in self.support_levels:
                # Breakout: shorted above support, covered below it
                if entry_price > level.price >= exit_price:
                    return self.breakout_bonus, "breakout"
                # Bounce: covered near support (within tolerance)
                if abs(exit_price - level.price) <= self.sr_tolerance and exit_price < entry_price:
                    return self.bounce_bonus, "bounce"

            for level in self.resistance_levels:
                # Bounce: shorted near resistance and it held
                if abs(entry_price - level.price) <= self.sr_tolerance and exit_price < entry_price:
                    return self.bounce_bonus, "bounce"

        return 1.0, None

    def _replenish_sources(self):
        """Add eggs back to sources."""
        all_zones = self.floor.left_zones + self.floor.right_zones
        for zone in all_zones:
            if zone.zone_type == ZoneType.SOURCE and zone.quantity < zone.max_quantity:
                if random.random() < 0.2:
                    zone.quantity += 1

    # === Rendering ===

    def render(self) -> str:
        """Render the trading floor."""
        lines = []

        # Header
        pnl_str = f"+${self.total_pnl:.2f}" if self.total_pnl >= 0 else f"-${abs(self.total_pnl):.2f}"
        level_names = {
            DifficultyLevel.LEVEL_1: "LEVEL 1",
            DifficultyLevel.LEVEL_1B: "LEVEL 1.5",
            DifficultyLevel.LEVEL_2: "LEVEL 2",
            DifficultyLevel.LEVEL_3: "LEVEL 3",
        }
        level_name = level_names.get(self.level, f"LEVEL {self.level.value}")
        lines.append(f"═══ EGG TRADER {level_name} ═══  P&L: {pnl_str}  Cash: ${self.cash:.2f}")
        lines.append(f"Trades: {len(self.trades)}  |  Tick: {self.tick}")
        lines.append("")

        # Level description
        if self.level == DifficultyLevel.LEVEL_1:
            lines.append("GREEN EGGS: Buy low on LEFT, sell high on RIGHT")
            lines.append("            Want price to GO UP! ↑↑↑")
        elif self.level == DifficultyLevel.LEVEL_1B:
            lines.append("RED EGGS: Sell high on RIGHT, buy back low on LEFT")
            lines.append("          Want price to GO DOWN! ↓↓↓")
        elif self.level == DifficultyLevel.LEVEL_2:
            lines.append("GREEN=long (want ↑)  RED=short (want ↓)")
            lines.append("Green: left→right    Red: right→left")
        elif self.level == DifficultyLevel.LEVEL_3:
            lines.append("REVERSAL: Close + open opposite in one move!")
        lines.append("")

        # Column headers
        lines.append("  PRICE │    LEFT SIDE     │    RIGHT SIDE    │ S/R")
        lines.append("  ──────┼──────────────────┼──────────────────┼────")

        # Build S/R lookup for quick access
        support_prices = {int(l.price): l for l in self.support_levels if not l.broken}
        resistance_prices = {int(l.price): l for l in self.resistance_levels if not l.broken}

        # Render each price level (top to bottom = high to low)
        for i in range(self.num_levels - 1, -1, -1):
            price = self.get_price_at_level(i)
            price_int = int(price)
            is_market = (i == self.price_index)
            is_player = (i == self.player_price_level)

            # Get zones at this level
            left_zones = self.get_zones_at_level(Side.LEFT, i)
            right_zones = self.get_zones_at_level(Side.RIGHT, i)

            # Format left side
            left_str = self._format_zones(left_zones, is_player and self.player_side == Side.LEFT)

            # Format right side
            right_str = self._format_zones(right_zones, is_player and self.player_side == Side.RIGHT)

            # Support/Resistance indicator
            sr_indicator = "    "
            if price_int in resistance_prices:
                level = resistance_prices[price_int]
                sr_indicator = f"R{'×' * min(level.strength, 3)}"[:4]
            elif price_int in support_prices:
                level = support_prices[price_int]
                sr_indicator = f"S{'×' * min(level.strength, 3)}"[:4]

            # Price marker
            if is_market:
                price_str = f"►${price:5.0f}"
            else:
                price_str = f" ${price:5.0f}"

            # Assemble row
            if is_market:
                row = f"{price_str}│═{left_str:16}═│═{right_str:16}═│{sr_indicator}"
            else:
                row = f"{price_str} │ {left_str:16} │ {right_str:16} │{sr_indicator}"

            lines.append(row)

        lines.append("  ──────┴──────────────────┴──────────────────┴────")
        lines.append("")

        # Player status
        side_name = "LEFT" if self.player_side == Side.LEFT else "RIGHT"
        lines.append(f"You are at ${self.player_price:.0f} on {side_name} side")

        zone = self.get_current_zone()
        if zone:
            color_sym = "●" if zone.color == EggColor.GREEN else "○"
            zone_desc = "SOURCE" if zone.zone_type == ZoneType.SOURCE else "SINK"
            lines.append(f"At: {zone.color.value.upper()} {zone_desc} ({zone.quantity}{color_sym} available)")

        if self.carrying:
            color_sym = "●" if self.carrying.color == EggColor.GREEN else "○"
            pos_type = "LONG" if self.carrying.color == EggColor.GREEN else "SHORT"
            lines.append(f"CARRYING: {self.carrying.quantity}{color_sym} {pos_type} from ${self.carrying.entry_price:.0f}")

            # P&L preview
            if self.carrying.color == EggColor.GREEN:
                preview = (self.current_market_price - self.carrying.entry_price) * self.carrying.quantity
            else:
                preview = (self.carrying.entry_price - self.current_market_price) * self.carrying.quantity
            pnl_str = f"+${preview:.2f}" if preview >= 0 else f"-${abs(preview):.2f}"
            lines.append(f"Unrealized P&L: {pnl_str}")

        # Show nearby support/resistance levels
        nearest_support, nearest_resistance = self.get_nearby_levels(self.current_market_price)
        sr_info = []
        if nearest_support:
            dist = self.current_market_price - nearest_support.price
            sr_info.append(f"Support: ${nearest_support.price:.0f} ({dist:.0f} below)")
        if nearest_resistance:
            dist = nearest_resistance.price - self.current_market_price
            sr_info.append(f"Resistance: ${nearest_resistance.price:.0f} ({dist:.0f} above)")
        if sr_info:
            lines.append("")
            lines.append("  ".join(sr_info))

        # Show bonus message if recent
        if self.last_bonus_message and (self.tick - self.bonus_message_tick) < 5:
            lines.append("")
            lines.append(f"*** {self.last_bonus_message} ***")

        lines.append("")
        lines.append("CONTROLS:")
        lines.append("  [↑/↓] Change price    [←/→] Switch sides    [TAB] Cycle zones")
        lines.append("  [1-5] Pick up eggs    [D] Deposit eggs      [R] Reverse (L3+)")
        lines.append("  [Q] Quit              [SPACE] Advance time")
        lines.append("")
        lines.append("S/R: S=Support (bounce up), R=Resistance (bounce down)")
        lines.append("     × = strength (more × = stronger level)")
        lines.append("     BREAKOUT = 2x bonus!  BOUNCE = 1.5x bonus!")

        return "\n".join(lines)

    def _format_zones(self, zones: List[EggZone], player_here: bool) -> str:
        """Format zones for display."""
        if not zones:
            return ""

        parts = []
        for idx, zone in enumerate(zones):
            sym = "●" if zone.color == EggColor.GREEN else "○"
            if zone.zone_type == ZoneType.SOURCE:
                # Show eggs available
                eggs = sym * min(zone.quantity, 5)
            else:
                # Sink - show as brackets
                eggs = f"[{sym}]"

            # Highlight if player is at this zone
            if player_here and idx == self.player_zone_index:
                parts.append(f"◆{eggs}")
            else:
                parts.append(eggs)

        return " ".join(parts)

    def reset(self):
        """Reset game state."""
        self.floor = create_floor_for_level(self.level, self.num_levels)
        self.price_index = self.num_levels // 2
        self.player_price_level = self.num_levels // 2
        self.player_side = Side.LEFT
        self.player_zone_index = 0
        self.carrying = None
        self.trades = []
        self.total_pnl = 0.0
        self.cash = 10000.0
        self.tick = 0
        self.game_over = False

        # Reset S/R tracking
        self.price_history = []
        self.support_levels = []
        self.resistance_levels = []
        self.last_bonus_message = None
        self.bonus_message_tick = 0


def demo():
    """Demo the leveled egg trader."""
    print("=" * 60)
    print("EGG TRADER LEVELS DEMO")
    print("=" * 60)
    print()

    # Level 1
    print("LEVEL 1: Green Eggs Only")
    print("-" * 40)
    game = EggTraderLevels(level=DifficultyLevel.LEVEL_1)
    print(game.render())
    print()

    # Level 2
    print("LEVEL 2: Green + Red Eggs")
    print("-" * 40)
    game = EggTraderLevels(level=DifficultyLevel.LEVEL_2)
    print(game.render())
    print()

    # Level 3
    print("LEVEL 3: Position Reversal")
    print("-" * 40)
    game = EggTraderLevels(level=DifficultyLevel.LEVEL_3)
    print(game.render())


if __name__ == "__main__":
    demo()
