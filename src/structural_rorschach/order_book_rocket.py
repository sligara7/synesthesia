"""
Order Book Rocket - Fly Through the Spread

A rocket flies between the BID and ASK walls of an order book.
The walls are literal liquidity - the depth of orders at each price level.

REAL-TIME MODE:
    Connect to Binance WebSocket for live order book data.
    Each price level shows as a horizontal bar (histogram style):

    ASK SIDE (sellers - TOP):
        $101.50 ████████████████  (1,234 BTC)
        $101.40 ██████████        (567 BTC)
        $101.30 ████              (123 BTC)  ← Best Ask
        ─────────────────────────────────────
        $101.20 ██████            (234 BTC)  ← Best Bid
        $101.10 ████████████      (789 BTC)
        $101.00 ██████████████████ (1,567 BTC)
    BID SIDE (buyers - BOTTOM)

THE METAPHOR:
    ════════════════════════════════════════════════════
    ▓▓▓▓▓▓▓▓  ASK WALL (sellers above)  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    ▓▓▓▓▓▓    Thick = many sellers = hard to push up
    ▓▓▓▓      Thin = few sellers = price can break out
    ▓▓
                   ◆ ROCKET
                   Flying through the spread

    ░░
    ░░░░      Thin = few buyers = price can drop
    ░░░░░░    Thick = many buyers = support
    ░░░░░░░░  BID WALL (buyers below)  ░░░░░░░░░░░░░░░░
    ════════════════════════════════════════════════════

GAMEPLAY:
    1. Fly to the ASK wall → "PICKUP" passengers (open LONG position)
       - You're picking up sellers who want to sell TO you
       - You BUY from them at their asking price

    2. Fly to the BID wall → "DROPOFF" passengers (close LONG position)
       - You're dropping off to buyers who want to buy FROM you
       - You SELL to them at their bidding price

    3. Profit = dropoff_price - pickup_price (for longs)
       - Pick up HIGH on ask side, drop off LOW on bid = LOSS
       - Pick up LOW on ask side, drop off HIGH on bid = PROFIT

    Wait... that's backwards for a long! Let me reconsider...

REVISED - The "Passenger" Metaphor:
    - Passengers on ASK side = people wanting to SELL their coins
    - Passengers on BID side = people wanting to BUY coins

    LONG Trade (bet price goes UP):
        1. Go to BID side, pickup passenger (buyer gives you money for coin)
           Actually no...

    Let me use a cleaner metaphor:

FINAL METAPHOR - Cargo Shuttle:
    - ASK wall = Source of LONG opportunities (buy here)
    - BID wall = Destination for LONG positions (sell here)

    LONG Position:
        1. Pickup at ASK wall (buy from sellers) - lower is better
        2. Dropoff at BID wall (sell to buyers) - higher is better
        3. Profit if BID dropoff > ASK pickup

    SHORT Position (maybe Level 2?):
        1. Pickup at BID wall (borrow to sell to buyers)
        2. Dropoff at ASK wall (buy back from sellers)
        3. Profit if ASK dropoff < BID pickup

    For Level 1, just do LONGs:
        - Fly to BOTTOM (asks)  → PICKUP cargo
        - Fly to TOP (bids) → DROPOFF cargo
        - Wall thickness = order book depth (liquidity)
        - Thick walls = easy to trade but price sticky
        - Thin walls = price can move but hard to fill orders

WALL VISUALIZATION (hybrid gradient + thick levels):
    Each price level shows:
    - Base gradient: proportional to depth
    - Highlighted "thick" levels: where large orders sit

    ▓▓▓▓▓▓▓▓  ← Thick level (large order)
    ▓▓▓▓      ← Normal depth
    ▓▓        ← Thin (low liquidity)
    ▓▓▓       ← Normal
    ▓▓▓▓▓▓▓▓▓▓ ← VERY thick level (whale order!)

CONTROLS:
    ↑/↓ = Move up/down (direct, no momentum - Nokia style)
    SPACE = Pickup or Dropoff (context-sensitive)

TIME FLOW:
    - Auto-scroll: order book updates automatically
    - Carrying cargo increases urgency (must exit before conditions change)
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import random


class GameState(Enum):
    """Current state of the game."""
    PLAYING = "playing"
    GAME_OVER = "game_over"
    PAUSED = "paused"


class CargoType(Enum):
    """Type of cargo (position) being carried."""
    LONG = "long"    # Bought from ask, sell to bid
    SHORT = "short"  # Borrowed to sell to bid, buy back from ask


@dataclass
class OrderLevel:
    """A single price level in the order book."""
    price: float
    bid_depth: float  # Quantity of buy orders at this price
    ask_depth: float  # Quantity of sell orders at this price
    is_thick_bid: bool = False  # Large bid order here
    is_thick_ask: bool = False  # Large ask order here

    @property
    def total_depth(self) -> float:
        return self.bid_depth + self.ask_depth

    @property
    def bid_pressure(self) -> float:
        """Normalized bid depth (0-1)."""
        return min(1.0, self.bid_depth / 100)  # Assuming max depth of 100

    @property
    def ask_pressure(self) -> float:
        """Normalized ask depth (0-1)."""
        return min(1.0, self.ask_depth / 100)


@dataclass
class OrderBook:
    """
    The order book represented as cave walls.

    Structure:
        - levels: Price levels from low to high
        - Each level has bid depth (bottom wall) and ask depth (top wall)
        - The "spread" is where you fly (between best bid and best ask)
    """
    levels: List[OrderLevel]
    best_bid_idx: int = 0  # Index of highest bid
    best_ask_idx: int = 0  # Index of lowest ask

    @property
    def spread(self) -> float:
        """The bid-ask spread in price units."""
        if self.best_bid_idx < len(self.levels) and self.best_ask_idx < len(self.levels):
            return self.levels[self.best_ask_idx].price - self.levels[self.best_bid_idx].price
        return 0.0

    @property
    def mid_price(self) -> float:
        """Middle price between best bid and ask."""
        if self.best_bid_idx < len(self.levels) and self.best_ask_idx < len(self.levels):
            bid = self.levels[self.best_bid_idx].price
            ask = self.levels[self.best_ask_idx].price
            return (bid + ask) / 2
        return 0.0


@dataclass
class Cargo:
    """Cargo being carried (an open position)."""
    cargo_type: CargoType
    quantity: int
    entry_price: float
    entry_tick: int


@dataclass
class CompletedTrip:
    """A completed cargo delivery (closed position)."""
    cargo_type: CargoType
    quantity: int
    entry_price: float
    exit_price: float
    pnl: float
    entry_tick: int
    exit_tick: int


@dataclass
class Rocket:
    """The player's rocket."""
    y: float = 0.5  # Vertical position (0=bottom/bid, 1=top/ask)
    health: float = 1.0


class OrderBookRocket:
    """
    Order Book Rocket Game.

    Fly through the order book spread, picking up and dropping off cargo.
    """

    def __init__(
        self,
        num_levels: int = 20,
        base_price: float = 100.0,
        price_step: float = 0.10,
        step_size: float = 0.05,  # Movement per input
        scroll_speed: float = 1.0,  # How fast order book updates
        cargo_urgency: float = 1.5,  # Scroll multiplier when carrying
    ):
        self.num_levels = num_levels
        self.base_price = base_price
        self.price_step = price_step
        self.step_size = step_size
        self.scroll_speed = scroll_speed
        self.cargo_urgency = cargo_urgency

        # Game state
        self.state = GameState.PLAYING
        self.tick = 0
        self.ticks_since_scroll = 0
        self.scroll_interval = 10  # Ticks between order book updates

        # Rocket
        self.rocket = Rocket()

        # Order book
        self.order_book = self._generate_order_book()

        # Trading state
        self.cargo: Optional[Cargo] = None
        self.completed_trips: List[CompletedTrip] = []
        self.total_pnl: float = 0.0
        self.cash: float = 10000.0

        # Level/difficulty
        self.level = 1  # Start with longs only
        self.max_cargo = 5

        # Messages
        self.message: Optional[str] = None
        self.message_tick: int = 0

    def _generate_order_book(self) -> OrderBook:
        """Generate initial order book with realistic structure."""
        levels = []

        # Find the middle - this is where the spread will be
        mid_idx = self.num_levels // 2

        for i in range(self.num_levels):
            price = self.base_price + i * self.price_step

            # Below mid = bids (buyers), above mid = asks (sellers)
            # Near the spread, depth is lower (spread exists because less liquidity)
            distance_from_mid = abs(i - mid_idx)

            if i < mid_idx:
                # Bid side (buyers below)
                # More depth further from spread
                base_depth = 15 + distance_from_mid * 3
                bid_depth = base_depth + random.uniform(-5, 5)
                ask_depth = 0  # No asks below the spread

                # Randomly mark thick levels (whale orders) - less frequent at start
                is_thick_bid = random.random() < 0.08  # 8% chance of thick level
                if is_thick_bid:
                    bid_depth *= 2.0  # Whale order is 2x normal

            elif i > mid_idx:
                # Ask side (sellers above)
                base_depth = 15 + distance_from_mid * 3
                bid_depth = 0
                ask_depth = base_depth + random.uniform(-5, 5)

                is_thick_ask = random.random() < 0.08
                if is_thick_ask:
                    ask_depth *= 2.0
            else:
                # At the spread - minimal depth
                bid_depth = random.uniform(5, 10)
                ask_depth = random.uniform(5, 10)
                is_thick_bid = False
                is_thick_ask = False

            levels.append(OrderLevel(
                price=price,
                bid_depth=max(0, bid_depth),
                ask_depth=max(0, ask_depth),
                is_thick_bid=is_thick_bid if i < mid_idx else False,
                is_thick_ask=is_thick_ask if i > mid_idx else False,
            ))

        # Best bid is highest price with bids, best ask is lowest price with asks
        best_bid_idx = mid_idx - 1  # Just below spread
        best_ask_idx = mid_idx + 1  # Just above spread

        return OrderBook(
            levels=levels,
            best_bid_idx=best_bid_idx,
            best_ask_idx=best_ask_idx,
        )

    def _update_order_book(self):
        """Update the order book (simulate market activity)."""
        # Randomly adjust depths
        for level in self.order_book.levels:
            # Bid side changes
            if level.bid_depth > 0:
                change = random.uniform(-5, 5)
                level.bid_depth = max(5, level.bid_depth + change)

                # Small chance thick level appears/disappears
                if random.random() < 0.05:
                    level.is_thick_bid = not level.is_thick_bid
                    if level.is_thick_bid:
                        level.bid_depth *= 2
                    else:
                        level.bid_depth /= 2

            # Ask side changes
            if level.ask_depth > 0:
                change = random.uniform(-5, 5)
                level.ask_depth = max(5, level.ask_depth + change)

                if random.random() < 0.05:
                    level.is_thick_ask = not level.is_thick_ask
                    if level.is_thick_ask:
                        level.ask_depth *= 2
                    else:
                        level.ask_depth /= 2

        # Occasionally shift the spread (price movement)
        if random.random() < 0.3:
            shift = random.choice([-1, 0, 0, 1])  # Slight bias to stay
            new_bid = self.order_book.best_bid_idx + shift
            new_ask = self.order_book.best_ask_idx + shift

            # Keep spread valid
            if 0 < new_bid < new_ask < self.num_levels - 1:
                self.order_book.best_bid_idx = new_bid
                self.order_book.best_ask_idx = new_ask

                # Update level depths to reflect new spread location
                for i, level in enumerate(self.order_book.levels):
                    if i <= new_bid:
                        level.ask_depth = 0
                        if level.bid_depth == 0:
                            level.bid_depth = random.uniform(20, 40)
                    elif i >= new_ask:
                        level.bid_depth = 0
                        if level.ask_depth == 0:
                            level.ask_depth = random.uniform(20, 40)

    # === Input Methods ===

    def input_up(self):
        """Move rocket up toward ASK wall."""
        if self.state == GameState.PLAYING:
            self.rocket.y = min(1.0, self.rocket.y + self.step_size)

    def input_down(self):
        """Move rocket down toward BID wall."""
        if self.state == GameState.PLAYING:
            self.rocket.y = max(0.0, self.rocket.y - self.step_size)

    def input_action(self) -> bool:
        """
        Pickup or dropoff cargo based on position.

        - Near ASK wall (top): pickup cargo (open long)
        - Near BID wall (bottom): dropoff cargo (close long)

        Returns True if action succeeded.
        """
        if self.state != GameState.PLAYING:
            return False

        # Determine which wall we're near
        # Use same zone calculation as collision detection
        bid_level = self.order_book.levels[self.order_book.best_bid_idx]
        ask_level = self.order_book.levels[self.order_book.best_ask_idx]
        max_depth = 150
        display_height = 12

        ask_wall_rows = max(2, int((ask_level.ask_depth / max_depth) * 4))
        bid_wall_rows = max(2, int((bid_level.bid_depth / max_depth) * 4))

        ask_zone = ask_wall_rows / display_height
        bid_zone = bid_wall_rows / display_height

        # "Near" means in the action zone - close to wall but not crashing
        # Collision happens at (1.0 - ask_zone - 0.10), so action zone is before that
        margin = 0.10  # Same as collision margin
        near_ask = self.rocket.y > (1.0 - ask_zone - margin - 0.15)  # Action zone is 15% before crash zone
        near_bid = self.rocket.y < (bid_zone + margin + 0.15)  # Same for bid

        if near_ask and self.cargo is None:
            # At ASK wall without cargo → PICKUP (open long)
            return self._pickup_cargo()
        elif near_bid and self.cargo is not None:
            # At BID wall with cargo → DROPOFF (close long)
            return self._dropoff_cargo()
        elif near_bid and self.cargo is None and self.level >= 2:
            # Level 2+: Can SHORT from bid side
            return self._pickup_short()
        elif near_ask and self.cargo is not None and self.cargo.cargo_type == CargoType.SHORT:
            # Closing a short at ask side
            return self._dropoff_cargo()
        else:
            self.message = "Can't do that here!"
            self.message_tick = self.tick
            return False

    def _pickup_cargo(self) -> bool:
        """Pick up LONG cargo from ASK wall."""
        # Get current ask price
        ask_level = self.order_book.levels[self.order_book.best_ask_idx]

        # Check if enough depth to fill
        quantity = min(self.max_cargo, int(ask_level.ask_depth / 10))
        if quantity <= 0:
            self.message = "Not enough liquidity!"
            self.message_tick = self.tick
            return False

        # Execute
        entry_price = ask_level.price
        self.cargo = Cargo(
            cargo_type=CargoType.LONG,
            quantity=quantity,
            entry_price=entry_price,
            entry_tick=self.tick,
        )

        # Consume some ask depth
        ask_level.ask_depth -= quantity * 10

        self.message = f"LONG {quantity}x @ ${entry_price:.2f}"
        self.message_tick = self.tick
        return True

    def _pickup_short(self) -> bool:
        """Pick up SHORT cargo from BID wall (Level 2+)."""
        bid_level = self.order_book.levels[self.order_book.best_bid_idx]

        quantity = min(self.max_cargo, int(bid_level.bid_depth / 10))
        if quantity <= 0:
            self.message = "Not enough liquidity!"
            self.message_tick = self.tick
            return False

        entry_price = bid_level.price
        self.cargo = Cargo(
            cargo_type=CargoType.SHORT,
            quantity=quantity,
            entry_price=entry_price,
            entry_tick=self.tick,
        )

        bid_level.bid_depth -= quantity * 10

        self.message = f"SHORT {quantity}x @ ${entry_price:.2f}"
        self.message_tick = self.tick
        return True

    def _dropoff_cargo(self) -> bool:
        """Drop off cargo to close position."""
        if self.cargo is None:
            return False

        if self.cargo.cargo_type == CargoType.LONG:
            # Closing long: sell to bid
            level = self.order_book.levels[self.order_book.best_bid_idx]
            exit_price = level.price
            pnl = (exit_price - self.cargo.entry_price) * self.cargo.quantity
        else:
            # Closing short: buy from ask
            level = self.order_book.levels[self.order_book.best_ask_idx]
            exit_price = level.price
            pnl = (self.cargo.entry_price - exit_price) * self.cargo.quantity

        # Record trip
        trip = CompletedTrip(
            cargo_type=self.cargo.cargo_type,
            quantity=self.cargo.quantity,
            entry_price=self.cargo.entry_price,
            exit_price=exit_price,
            pnl=pnl,
            entry_tick=self.cargo.entry_tick,
            exit_tick=self.tick,
        )
        self.completed_trips.append(trip)
        self.total_pnl += pnl
        self.cash += pnl

        # Format message
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        pos_type = "LONG" if self.cargo.cargo_type == CargoType.LONG else "SHORT"
        self.message = f"Closed {pos_type} @ ${exit_price:.2f} = {pnl_str}"
        self.message_tick = self.tick

        self.cargo = None
        return True

    # === Game Loop ===

    def tick_update(self) -> bool:
        """
        Advance game by one tick.
        Returns True if still playing.
        """
        if self.state != GameState.PLAYING:
            return False

        self.tick += 1
        self.ticks_since_scroll += 1

        # Determine scroll rate (faster when carrying cargo)
        scroll_rate = self.scroll_interval
        if self.cargo is not None:
            scroll_rate = int(scroll_rate / self.cargo_urgency)

        # Update order book periodically
        if self.ticks_since_scroll >= scroll_rate:
            self._update_order_book()
            self.ticks_since_scroll = 0

        # Check for collisions with walls
        if not self._check_position():
            self.state = GameState.GAME_OVER
            self.message = "CRASHED into the wall!"
            self.message_tick = self.tick
            return False

        return True

    def _check_position(self) -> bool:
        """
        Check if rocket position is valid (not inside a wall).
        Returns True if OK, False if crashed.

        The walls are horizontal bars:
        - BID wall at bottom (buyers)
        - ASK wall at top (sellers)

        We use the same calculation as the rendering to determine wall zones.
        """
        # Get wall depths
        bid_level = self.order_book.levels[self.order_book.best_bid_idx]
        ask_level = self.order_book.levels[self.order_book.best_ask_idx]
        max_depth = 150

        # Calculate wall zones as fractions of total height
        # These match the display calculation (wall_rows out of display_height)
        display_height = 12  # Same as render uses (20 - 8)
        ask_wall_rows = max(2, int((ask_level.ask_depth / max_depth) * 4))
        bid_wall_rows = max(2, int((bid_level.bid_depth / max_depth) * 4))

        # Convert to y-coordinate thresholds
        # Ask wall is at top (y=1), bid wall is at bottom (y=0)
        ask_zone = ask_wall_rows / display_height  # How far down ask wall extends
        bid_zone = bid_wall_rows / display_height  # How far up bid wall extends

        # Rocket y=1 is at top (near ask), y=0 is at bottom (near bid)
        # Collision if rocket y > (1 - ask_zone) or y < bid_zone
        # Use a generous margin - walls are visual boundaries, not instant death
        margin = 0.10  # Generous margin for playability

        if self.rocket.y < bid_zone + margin:
            return False  # Hit bid wall
        if self.rocket.y > (1.0 - ask_zone - margin):
            return False  # Hit ask wall

        return True

    def reset(self):
        """Reset game state."""
        self.state = GameState.PLAYING
        self.tick = 0
        self.ticks_since_scroll = 0
        self.rocket = Rocket()
        self.order_book = self._generate_order_book()
        self.cargo = None
        self.completed_trips = []
        self.total_pnl = 0.0
        self.cash = 10000.0
        self.message = None
        self.message_tick = 0

    # === Rendering ===

    def render(self, height: int = 20, width: int = 60) -> str:
        """
        Render the order book rocket game.

        The display shows:
        - ASK wall at TOP (sellers) - horizontal bar, depth shown as thickness
        - Open space in middle (the spread where you fly)
        - BID wall at BOTTOM (buyers) - horizontal bar, depth shown as thickness
        - Rocket position
        - Thick levels highlighted with different characters
        """
        lines = []

        # Header
        pnl_str = f"+${self.total_pnl:.2f}" if self.total_pnl >= 0 else f"-${abs(self.total_pnl):.2f}"
        lines.append(f"═══ ORDER BOOK ROCKET ═══  P&L: {pnl_str}  Trips: {len(self.completed_trips)}")

        # Spread info
        spread = self.order_book.spread
        mid = self.order_book.mid_price
        lines.append(f"Mid: ${mid:.2f}  Spread: ${spread:.2f}  Tick: {self.tick}")
        lines.append("")

        # Get wall info for collision display
        bid_level = self.order_book.levels[self.order_book.best_bid_idx]
        ask_level = self.order_book.levels[self.order_book.best_ask_idx]
        max_depth = 150

        # Wall heights (number of rows each wall takes)
        ask_wall_rows = max(2, int((ask_level.ask_depth / max_depth) * 4))
        bid_wall_rows = max(2, int((bid_level.bid_depth / max_depth) * 4))

        display_height = height - 8  # Leave room for headers/footers
        open_space_rows = display_height - ask_wall_rows - bid_wall_rows

        # Calculate rocket row within the open space
        # rocket.y=0 should be just above bid wall, rocket.y=1 just below ask wall
        # But we need to map to actual rows
        rocket_y_in_open = self.rocket.y  # 0=bottom, 1=top
        rocket_row_in_open = int((1 - rocket_y_in_open) * (open_space_rows - 1))
        rocket_row_in_open = max(0, min(open_space_rows - 1, rocket_row_in_open))
        rocket_actual_row = ask_wall_rows + rocket_row_in_open

        # Top border
        lines.append("╔" + "═" * (width - 2) + "╗")

        for row in range(display_height):
            row_chars = ["║"]

            # Determine what zone this row is in
            if row < ask_wall_rows:
                # ASK WALL (sellers at top)
                # Show depth bar - thicker = more sellers
                depth_pct = ask_level.ask_depth / max_depth
                bar_width = int(depth_pct * (width - 4))
                bar_width = max(4, min(width - 6, bar_width))

                # Center the bar
                left_pad = (width - 4 - bar_width) // 2
                right_pad = width - 4 - bar_width - left_pad

                if row == 0:
                    # Top of ask wall - show price
                    label = f" ASK ${ask_level.price:.2f} "
                    bar_char = "▓" if ask_level.is_thick_ask else "░"
                    fill = bar_char * (bar_width - len(label))
                    row_chars.append(" " * left_pad + label + fill + " " * right_pad)
                else:
                    bar_char = "▓" if ask_level.is_thick_ask else "░"
                    row_chars.append(" " * left_pad + bar_char * bar_width + " " * right_pad)

            elif row >= display_height - bid_wall_rows:
                # BID WALL (buyers at bottom)
                depth_pct = bid_level.bid_depth / max_depth
                bar_width = int(depth_pct * (width - 4))
                bar_width = max(4, min(width - 6, bar_width))

                left_pad = (width - 4 - bar_width) // 2
                right_pad = width - 4 - bar_width - left_pad

                if row == display_height - 1:
                    # Bottom of bid wall - show price
                    label = f" BID ${bid_level.price:.2f} "
                    bar_char = "▓" if bid_level.is_thick_bid else "░"
                    fill = bar_char * (bar_width - len(label))
                    row_chars.append(" " * left_pad + label + fill + " " * right_pad)
                else:
                    bar_char = "▓" if bid_level.is_thick_bid else "░"
                    row_chars.append(" " * left_pad + bar_char * bar_width + " " * right_pad)

            else:
                # OPEN SPACE (the spread - where you fly)
                middle = list(" " * (width - 4))

                # Place rocket
                if row == rocket_actual_row:
                    rocket_col = (width - 4) // 2
                    if self.cargo is not None:
                        if self.cargo.cargo_type == CargoType.LONG:
                            middle[rocket_col] = "▲"
                        else:
                            middle[rocket_col] = "▼"
                    else:
                        middle[rocket_col] = "◆"

                # Pickup/dropoff zone indicators
                row_in_open = row - ask_wall_rows
                if row_in_open <= 1:
                    # Near ask wall - PICKUP zone for longs
                    middle[1] = "["
                    middle[2] = "P"
                    middle[3] = "I"
                    middle[4] = "C"
                    middle[5] = "K"
                    middle[6] = "U"
                    middle[7] = "P"
                    middle[8] = "]"
                elif row_in_open >= open_space_rows - 2:
                    # Near bid wall - DROPOFF zone for longs
                    middle[1] = "["
                    middle[2] = "D"
                    middle[3] = "R"
                    middle[4] = "O"
                    middle[5] = "P"
                    middle[6] = "]"

                row_chars.append("".join(middle))

            row_chars.append("║")
            lines.append("".join(row_chars))

        # Bottom border
        lines.append("╚" + "═" * (width - 2) + "╝")

        # Legend
        lines.append("")
        lines.append("░░ = BID wall (buyers)    ▓▓ = Thick level (whale order)")
        lines.append("░░ = ASK wall (sellers)   ◆ = Rocket  ▲ = Carrying LONG  ▼ = Carrying SHORT")
        lines.append("")

        # Cargo status
        if self.cargo is not None:
            ptype = "LONG" if self.cargo.cargo_type == CargoType.LONG else "SHORT"
            unrealized = self._calculate_unrealized_pnl()
            unr_str = f"+${unrealized:.2f}" if unrealized >= 0 else f"-${abs(unrealized):.2f}"
            lines.append(f"CARRYING: {self.cargo.quantity}x {ptype} @ ${self.cargo.entry_price:.2f}  Unrealized: {unr_str}")
        else:
            lines.append("No cargo - fly to ASK wall (top) to PICKUP")

        # Position hints
        if self.rocket.y > 0.7:
            lines.append(">>> Near ASK wall - Press SPACE to PICKUP cargo <<<")
        elif self.rocket.y < 0.3:
            if self.cargo is not None:
                lines.append(">>> Near BID wall - Press SPACE to DROPOFF cargo <<<")
            else:
                lines.append(">>> Near BID wall - fly UP to pick up cargo first <<<")

        # Message
        if self.message and (self.tick - self.message_tick) < 10:
            lines.append("")
            lines.append(f"*** {self.message} ***")

        # Controls
        lines.append("")
        lines.append("CONTROLS: [↑/↓] Move  [SPACE] Pickup/Dropoff  [R] Reset  [Q] Quit")

        if self.state == GameState.GAME_OVER:
            lines.append("")
            lines.append("═══ GAME OVER ═══ Press R to restart")

        return "\n".join(lines)

    def _calculate_unrealized_pnl(self) -> float:
        """Calculate unrealized P&L on current cargo."""
        if self.cargo is None:
            return 0.0

        if self.cargo.cargo_type == CargoType.LONG:
            # Would sell to bid
            current_price = self.order_book.levels[self.order_book.best_bid_idx].price
            return (current_price - self.cargo.entry_price) * self.cargo.quantity
        else:
            # Would buy from ask
            current_price = self.order_book.levels[self.order_book.best_ask_idx].price
            return (self.cargo.entry_price - current_price) * self.cargo.quantity

    def get_wall_at_y(self, y: float) -> Tuple[float, float]:
        """
        Get wall positions at a given y coordinate.
        Returns (bid_wall_right, ask_wall_left) as x coordinates.
        """
        level_idx = int(y * (self.num_levels - 1))
        level_idx = max(0, min(self.num_levels - 1, level_idx))
        level = self.order_book.levels[level_idx]

        max_depth = 100
        bid_wall = level.bid_depth / max_depth * 0.3  # Grows from left (0)
        ask_wall = 1.0 - (level.ask_depth / max_depth * 0.3)  # Grows from right (1)

        return (bid_wall, ask_wall)


# =============================================================================
# LIVE ORDER BOOK ROCKET - Real-time Binance data with histogram walls
# =============================================================================

@dataclass
class HistogramLevel:
    """A single level in the order book histogram."""
    price: float
    quantity: float
    is_bid: bool  # True for bid (buy), False for ask (sell)

    @property
    def side_label(self) -> str:
        return "BID" if self.is_bid else "ASK"


class AlienColor(Enum):
    """Color of alien passengers."""
    GREEN = "green"  # Long position - wants to go UP (buy low, sell high)
    RED = "red"      # Short position - wants to go DOWN (sell high, buy low)


@dataclass
class AlienCargo:
    """Aliens being carried."""
    color: AlienColor
    count: int
    pickup_price: float
    pickup_tick: int


class LiveOrderBookRocket:
    """
    Order Book Rocket with real-time Binance data.

    KID-FRIENDLY VERSION:
    - Two walls of aliens facing each other
    - Green aliens at bottom wall - pick them up, drop at top = profit!
    - Red aliens at top wall - pick them up, drop at bottom = profit!
    - Walls show how many aliens are waiting at each level

    TRADING TRANSLATION:
    - Green aliens = LONG position (buy low, sell high)
    - Red aliens = SHORT position (sell high, buy low)
    - Bottom wall = BID side (buyers)
    - Top wall = ASK side (sellers)
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        num_levels: int = 10,  # How many price levels to show on each side
        step_size: float = 0.05,
        kid_mode: bool = True,  # Simplified labels for kids
        tld: str = "us",  # Binance TLD: "us" for binance.us, "com" for binance.com
        bar_interval_seconds: int = 10,  # Bar interval for MagicIndicator (default 10 seconds)
    ):
        self.symbol = symbol
        self.num_levels = num_levels
        self.step_size = step_size
        self.kid_mode = kid_mode
        self.tld = tld
        self.bar_interval_seconds = bar_interval_seconds

        # Game state
        self.state = GameState.PLAYING
        self.tick = 0

        # Rocket position (0 = bottom wall, 1 = top wall)
        self.rocket_y = 0.5  # Start in middle

        # Horizontal position for limit order selection (0 = best price, increases away from spread)
        self.rocket_x = 0  # Index into order book levels (0 = best bid/ask)
        self.limit_order_mode = False  # When True, left/right selects limit price

        # Pending limit orders (waiting to be filled)
        self.pending_limit_order: Optional[dict] = None  # {side, price, count, tick}

        # Order book data (will be populated from Binance)
        self.bids: List[HistogramLevel] = []  # Highest price first
        self.asks: List[HistogramLevel] = []  # Lowest price first
        self.best_bid: float = 0.0
        self.best_ask: float = 0.0
        self.max_quantity: float = 1.0  # For normalization

        # Alien cargo (simplified trading)
        self.aliens: Optional[AlienCargo] = None
        self.completed_trips: List[CompletedTrip] = []
        self.total_pnl: float = 0.0
        self.score: int = 0  # Kid-friendly score

        # Binance stream (lazy init)
        self._stream = None
        self._stream_started = False

        # Message
        self.message: Optional[str] = None
        self.message_tick: int = 0

        # MagicIndicator stop levels history (last 3 values)
        # popstop = triggered when downtrend starts (Low at that moment)
        # dropstop = triggered when uptrend starts (High at that moment)
        self.popstop_history: List[float] = []  # Most recent first
        self.dropstop_history: List[float] = []  # Most recent first
        self._magic_indicator = None
        self._last_bar_time = None

    def connect(self):
        """Connect to Binance WebSocket for live order book data."""
        try:
            from .binance_live import BinanceLiveStream
        except ImportError:
            try:
                from binance_live import BinanceLiveStream
            except ImportError:
                print("binance_live module not available, using simulated data")
                return False

        try:
            self._stream = BinanceLiveStream(
                symbol=self.symbol,
                tld=self.tld,
                on_depth=self._handle_depth_update,
                on_bar=self._handle_bar_update,  # For MagicIndicator stop levels
                bar_interval_seconds=self.bar_interval_seconds,  # Custom bar interval
            )
            self._stream.start()
            self._stream_started = True
            self.message = f"Connected to {self.symbol}"
            self.message_tick = self.tick
            return True
        except Exception as e:
            print(f"Could not connect to Binance: {e}")
            print("Using simulated data instead.")
            return False

    def disconnect(self):
        """Disconnect from Binance WebSocket."""
        if self._stream and self._stream_started:
            self._stream.stop()
            self._stream_started = False

    def _handle_depth_update(self, book):
        """Handle order book depth updates from Binance."""
        # Convert to HistogramLevel format
        self.bids = []
        self.asks = []

        # Bids: highest price first (book.bids is already sorted this way)
        for price_str, qty_str in book.bids[:self.num_levels]:
            self.bids.append(HistogramLevel(
                price=float(price_str),
                quantity=float(qty_str),
                is_bid=True,
            ))

        # Asks: lowest price first (book.asks is already sorted this way)
        for price_str, qty_str in book.asks[:self.num_levels]:
            self.asks.append(HistogramLevel(
                price=float(price_str),
                quantity=float(qty_str),
                is_bid=False,
            ))

        # Update best bid/ask
        if self.bids:
            self.best_bid = self.bids[0].price
        if self.asks:
            self.best_ask = self.asks[0].price

        # Update max quantity for normalization
        all_qtys = [l.quantity for l in self.bids + self.asks]
        if all_qtys:
            self.max_quantity = max(all_qtys)

    def _handle_bar_update(self, bar):
        """Handle kline/candlestick updates for MagicIndicator."""
        import math

        # Initialize MagicIndicator if needed
        if self._magic_indicator is None:
            try:
                # Try to import from petra/omega
                import sys
                sys.path.insert(0, '/home/ajs7/petra/omega')
                from indicators import MagicIndicator
                # Use BT (Bitcoin) tick size or default
                self._magic_indicator = MagicIndicator('BT')
            except ImportError:
                # Create a simple inline version if not available
                self._magic_indicator = self._create_simple_magic_indicator()
                return

        # Only process completed bars (not partial updates)
        # bar has: open, high, low, close, volume, timestamp
        bar_time = getattr(bar, 'timestamp', None) or getattr(bar, 'close_time', None)

        if bar_time and bar_time != self._last_bar_time:
            self._last_bar_time = bar_time

            # Get previous stop values before update
            prev_popstop = self._magic_indicator.state.get('popstop', float('nan'))
            prev_dropstop = self._magic_indicator.state.get('dropstop', float('nan'))

            # Update the indicator with OHLC data
            self._magic_indicator.update(bar.open, bar.high, bar.low, bar.close)

            # Check if popstop changed (new downtrend detected)
            new_popstop = self._magic_indicator.state.get('popstop', float('nan'))
            if not math.isnan(new_popstop) and (math.isnan(prev_popstop) or new_popstop != prev_popstop):
                # Add to history (most recent first)
                self.popstop_history.insert(0, new_popstop)
                # Keep only last 3
                self.popstop_history = self.popstop_history[:3]

            # Check if dropstop changed (new uptrend detected)
            new_dropstop = self._magic_indicator.state.get('dropstop', float('nan'))
            if not math.isnan(new_dropstop) and (math.isnan(prev_dropstop) or new_dropstop != prev_dropstop):
                # Add to history (most recent first)
                self.dropstop_history.insert(0, new_dropstop)
                # Keep only last 3
                self.dropstop_history = self.dropstop_history[:3]

    def _create_simple_magic_indicator(self):
        """Create a simple fallback indicator if MagicIndicator not available."""
        class SimpleMagicFallback:
            def __init__(self):
                self.state = {
                    'popstop': float('nan'),
                    'dropstop': float('nan'),
                    'uptrend': 0,
                    'dntrend': 0,
                    'uptrend_prev': 0,
                    'dntrend_prev': 0,
                }
                self.prev_high = float('nan')
                self.prev_low = float('nan')

            def update(self, o, h, l, c):
                import math
                # Simple trend detection based on higher highs / lower lows
                if not math.isnan(self.prev_high):
                    # Uptrend: making higher highs
                    if h > self.prev_high:
                        self.state['uptrend_prev'] = self.state['uptrend']
                        self.state['uptrend'] = 1
                        if self.state['uptrend_prev'] == 0:
                            self.state['dropstop'] = h
                    else:
                        self.state['uptrend_prev'] = self.state['uptrend']
                        self.state['uptrend'] = 0

                    # Downtrend: making lower lows
                    if l < self.prev_low:
                        self.state['dntrend_prev'] = self.state['dntrend']
                        self.state['dntrend'] = -1
                        if self.state['dntrend_prev'] == 0:
                            self.state['popstop'] = l
                    else:
                        self.state['dntrend_prev'] = self.state['dntrend']
                        self.state['dntrend'] = 0

                self.prev_high = h
                self.prev_low = l

        return SimpleMagicFallback()

    def set_simulated_book(self, mid_price: float = 100.0, spread: float = 0.10):
        """Set up simulated order book for testing without Binance."""
        self.bids = []
        self.asks = []

        # Generate bid levels (below mid, descending prices)
        for i in range(self.num_levels):
            price = mid_price - spread / 2 - i * 0.01
            qty = random.uniform(10, 100) * (1 + i * 0.5)  # More depth further from spread
            self.bids.append(HistogramLevel(price=price, quantity=qty, is_bid=True))

        # Generate ask levels (above mid, ascending prices)
        for i in range(self.num_levels):
            price = mid_price + spread / 2 + i * 0.01
            qty = random.uniform(10, 100) * (1 + i * 0.5)
            self.asks.append(HistogramLevel(price=price, quantity=qty, is_bid=False))

        self.best_bid = self.bids[0].price if self.bids else mid_price - spread / 2
        self.best_ask = self.asks[0].price if self.asks else mid_price + spread / 2

        all_qtys = [l.quantity for l in self.bids + self.asks]
        self.max_quantity = max(all_qtys) if all_qtys else 100

    def input_up(self):
        """Move rocket up toward ASK side."""
        if self.state == GameState.PLAYING:
            self.rocket_y = min(1.0, self.rocket_y + self.step_size)
            # Exit limit order mode when moving vertically
            self.limit_order_mode = False
            self.rocket_x = 0

    def input_down(self):
        """Move rocket down toward BID side."""
        if self.state == GameState.PLAYING:
            self.rocket_y = max(0.0, self.rocket_y - self.step_size)
            # Exit limit order mode when moving vertically
            self.limit_order_mode = False
            self.rocket_x = 0

    def input_left(self):
        """Move left in order book (toward worse prices for limit orders)."""
        if self.state == GameState.PLAYING:
            # Only allow horizontal movement when near a wall
            if self.rocket_y > 0.7 or self.rocket_y < 0.3:
                self.limit_order_mode = True
                max_level = self.num_levels - 1
                self.rocket_x = min(max_level, self.rocket_x + 1)

    def input_right(self):
        """Move right in order book (toward better prices / market order)."""
        if self.state == GameState.PLAYING:
            if self.rocket_y > 0.7 or self.rocket_y < 0.3:
                self.limit_order_mode = True
                self.rocket_x = max(0, self.rocket_x - 1)
                if self.rocket_x == 0:
                    self.limit_order_mode = False  # Back to market order

    def input_action(self) -> bool:
        """
        Pickup or dropoff aliens.

        GREEN aliens: Pick up at BOTTOM, drop off at TOP (long)
        RED aliens: Pick up at TOP, drop off at BOTTOM (short)

        If in limit_order_mode (rocket_x > 0), places a LIMIT order instead of market order.
        """
        if self.state != GameState.PLAYING:
            return False

        near_top = self.rocket_y > 0.7   # Near ask/top wall
        near_bottom = self.rocket_y < 0.3  # Near bid/bottom wall

        if self.aliens is None:
            # PICKUP - no aliens on board (opening a position)
            if near_bottom and self.bids:
                # Pick up GREEN aliens from bottom (open long)
                level_idx = min(self.rocket_x, len(self.bids) - 1)
                level = self.bids[level_idx]
                price = level.price
                count = min(5, max(1, int(level.quantity / 20)))

                if self.limit_order_mode and self.rocket_x > 0:
                    # LIMIT ORDER - place order and wait for fill
                    self.pending_limit_order = {
                        'side': 'long',
                        'price': price,
                        'count': count,
                        'tick': self.tick,
                        'color': AlienColor.GREEN,
                    }
                    if self.kid_mode:
                        self.message = f"Waiting for {count} GREEN @ ${price:.2f}"
                    else:
                        self.message = f"LIMIT LONG {count}x @ ${price:.2f}"
                else:
                    # MARKET ORDER - immediate fill
                    self.aliens = AlienCargo(
                        color=AlienColor.GREEN,
                        count=count,
                        pickup_price=price,
                        pickup_tick=self.tick,
                    )
                    if self.kid_mode:
                        self.message = f"Picked up {count} GREEN aliens!"
                    else:
                        self.message = f"LONG {count}x @ ${price:.2f}"
                self.message_tick = self.tick
                return True

            elif near_top and self.asks:
                # Pick up RED aliens from top (open short)
                level_idx = min(self.rocket_x, len(self.asks) - 1)
                level = self.asks[level_idx]
                price = level.price
                count = min(5, max(1, int(level.quantity / 20)))

                if self.limit_order_mode and self.rocket_x > 0:
                    # LIMIT ORDER
                    self.pending_limit_order = {
                        'side': 'short',
                        'price': price,
                        'count': count,
                        'tick': self.tick,
                        'color': AlienColor.RED,
                    }
                    if self.kid_mode:
                        self.message = f"Waiting for {count} RED @ ${price:.2f}"
                    else:
                        self.message = f"LIMIT SHORT {count}x @ ${price:.2f}"
                else:
                    # MARKET ORDER
                    self.aliens = AlienCargo(
                        color=AlienColor.RED,
                        count=count,
                        pickup_price=price,
                        pickup_tick=self.tick,
                    )
                    if self.kid_mode:
                        self.message = f"Picked up {count} RED aliens!"
                    else:
                        self.message = f"SHORT {count}x @ ${price:.2f}"
                self.message_tick = self.tick
                return True

        else:
            # DROPOFF - have aliens on board (closing a position)
            if self.aliens.color == AlienColor.GREEN and near_top and self.asks:
                # Drop GREEN aliens at top (close long - sell high)
                level_idx = min(self.rocket_x, len(self.asks) - 1)
                level = self.asks[level_idx]
                exit_price = level.price

                if self.limit_order_mode and self.rocket_x > 0:
                    # LIMIT ORDER to close
                    self.pending_limit_order = {
                        'side': 'close_long',
                        'price': exit_price,
                        'count': self.aliens.count,
                        'tick': self.tick,
                        'entry_price': self.aliens.pickup_price,
                    }
                    if self.kid_mode:
                        self.message = f"Limit sell @ ${exit_price:.2f}"
                    else:
                        self.message = f"LIMIT SELL @ ${exit_price:.2f}"
                    self.message_tick = self.tick
                    return True
                else:
                    # MARKET ORDER
                    pnl = (exit_price - self.aliens.pickup_price) * self.aliens.count
                    self._complete_trip(exit_price, pnl)
                    return True

            elif self.aliens.color == AlienColor.RED and near_bottom and self.bids:
                # Drop RED aliens at bottom (close short - buy low)
                level_idx = min(self.rocket_x, len(self.bids) - 1)
                level = self.bids[level_idx]
                exit_price = level.price

                if self.limit_order_mode and self.rocket_x > 0:
                    # LIMIT ORDER to close
                    self.pending_limit_order = {
                        'side': 'close_short',
                        'price': exit_price,
                        'count': self.aliens.count,
                        'tick': self.tick,
                        'entry_price': self.aliens.pickup_price,
                    }
                    if self.kid_mode:
                        self.message = f"Limit buy @ ${exit_price:.2f}"
                    else:
                        self.message = f"LIMIT BUY @ ${exit_price:.2f}"
                    self.message_tick = self.tick
                    return True
                else:
                    # MARKET ORDER
                    pnl = (self.aliens.pickup_price - exit_price) * self.aliens.count
                    self._complete_trip(exit_price, pnl)
                    return True

        if self.kid_mode:
            self.message = "Fly to the other wall!"
        else:
            self.message = "Can't do that here!"
        self.message_tick = self.tick
        return False

    def cancel_limit_order(self):
        """Cancel any pending limit order."""
        if self.pending_limit_order:
            self.pending_limit_order = None
            self.message = "Limit order cancelled"
            self.message_tick = self.tick

    def _complete_trip(self, exit_price: float, pnl: float):
        """Complete an alien delivery trip."""
        trip = CompletedTrip(
            cargo_type=CargoType.LONG if self.aliens.color == AlienColor.GREEN else CargoType.SHORT,
            quantity=self.aliens.count,
            entry_price=self.aliens.pickup_price,
            exit_price=exit_price,
            pnl=pnl,
            entry_tick=self.aliens.pickup_tick,
            exit_tick=self.tick,
        )
        self.completed_trips.append(trip)
        self.total_pnl += pnl

        # Score for kids (1 point per $1 profit, minimum 1 for completing)
        points = max(1, int(pnl))
        if pnl > 0:
            self.score += points

        if self.kid_mode:
            if pnl > 0:
                self.message = f"Delivered! +{points} points!"
            else:
                self.message = f"Delivered! (wrong direction)"
        else:
            pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
            self.message = f"Closed @ ${exit_price:.2f} = {pnl_str}"

        self.message_tick = self.tick
        self.aliens = None

    def tick_update(self) -> bool:
        """Advance game state."""
        if self.state != GameState.PLAYING:
            return False

        self.tick += 1

        # If not connected to live data, simulate some movement
        if not self._stream_started and self.tick % 10 == 0:
            self._simulate_book_update()

        # Check if pending limit order should fill
        self._check_limit_order_fill()

        return True

    def _check_limit_order_fill(self):
        """Check if a pending limit order should be filled."""
        if not self.pending_limit_order:
            return

        order = self.pending_limit_order
        filled = False

        if order['side'] == 'long':
            # Long limit order fills when best ask drops to or below our price
            if self.best_ask <= order['price']:
                self.aliens = AlienCargo(
                    color=order['color'],
                    count=order['count'],
                    pickup_price=order['price'],
                    pickup_tick=self.tick,
                )
                if self.kid_mode:
                    self.message = f"FILLED! Got {order['count']} GREEN @ ${order['price']:.2f}"
                else:
                    self.message = f"LIMIT FILLED: LONG {order['count']}x @ ${order['price']:.2f}"
                filled = True

        elif order['side'] == 'short':
            # Short limit order fills when best bid rises to or above our price
            if self.best_bid >= order['price']:
                self.aliens = AlienCargo(
                    color=order['color'],
                    count=order['count'],
                    pickup_price=order['price'],
                    pickup_tick=self.tick,
                )
                if self.kid_mode:
                    self.message = f"FILLED! Got {order['count']} RED @ ${order['price']:.2f}"
                else:
                    self.message = f"LIMIT FILLED: SHORT {order['count']}x @ ${order['price']:.2f}"
                filled = True

        elif order['side'] == 'close_long':
            # Limit sell fills when best bid rises to or above our price
            if self.best_bid >= order['price']:
                pnl = (order['price'] - order['entry_price']) * order['count']
                self._complete_trip(order['price'], pnl)
                filled = True

        elif order['side'] == 'close_short':
            # Limit buy fills when best ask drops to or below our price
            if self.best_ask <= order['price']:
                pnl = (order['entry_price'] - order['price']) * order['count']
                self._complete_trip(order['price'], pnl)
                filled = True

        if filled:
            self.pending_limit_order = None
            self.message_tick = self.tick

    def _simulate_book_update(self):
        """Simulate order book changes for demo mode."""
        for level in self.bids + self.asks:
            level.quantity += random.uniform(-5, 5)
            level.quantity = max(1, level.quantity)

        # Occasionally shift prices
        if random.random() < 0.2:
            shift = random.choice([-0.01, 0, 0.01])
            for level in self.bids:
                level.price += shift
            for level in self.asks:
                level.price += shift
            if self.bids:
                self.best_bid = self.bids[0].price
            if self.asks:
                self.best_ask = self.asks[0].price

        # Generate simulated bar data for MagicIndicator (every ~10 ticks = 1 "bar")
        self._simulate_bar_update()

    def _simulate_bar_update(self):
        """Generate simulated OHLC bar data for MagicIndicator in demo mode."""
        # Use mid price as the base
        mid_price = (self.best_bid + self.best_ask) / 2 if self.best_bid and self.best_ask else 50000.0

        # Initialize simulated bar state if needed
        if not hasattr(self, '_sim_bar_open'):
            self._sim_bar_open = mid_price
            self._sim_bar_high = mid_price
            self._sim_bar_low = mid_price
            self._sim_bar_close = mid_price
            self._sim_bar_tick_count = 0
            self._sim_bar_interval = 5  # Complete a bar every 5 updates

        # Update current bar with new price
        self._sim_bar_high = max(self._sim_bar_high, mid_price)
        self._sim_bar_low = min(self._sim_bar_low, mid_price)
        self._sim_bar_close = mid_price
        self._sim_bar_tick_count += 1

        # When bar interval reached, emit bar and start new one
        if self._sim_bar_tick_count >= self._sim_bar_interval:
            # Create a simulated bar object
            class SimBar:
                def __init__(self, o, h, l, c, ts):
                    self.open = o
                    self.high = h
                    self.low = l
                    self.close = c
                    self.timestamp = ts
                    self.volume = random.uniform(10, 100)

            bar = SimBar(
                self._sim_bar_open,
                self._sim_bar_high,
                self._sim_bar_low,
                self._sim_bar_close,
                self.tick,  # Use tick count as timestamp
            )

            # Feed to MagicIndicator
            self._handle_bar_update(bar)

            # Start new bar
            self._sim_bar_open = mid_price
            self._sim_bar_high = mid_price
            self._sim_bar_low = mid_price
            self._sim_bar_close = mid_price
            self._sim_bar_tick_count = 0

    def render(self, height: int = 24, width: int = 70) -> str:
        """
        Render the game with order book histograms on TOP and BOTTOM walls.
        Bars extend horizontally toward the center (like stalagmites/stalactites).

        KID MODE:
            ┌──────────────────────────────────────────────────────────────┐
            │  ▓▓▓▓▓▓▓▓     ▓▓▓▓▓     ▓▓▓▓▓▓▓▓▓▓▓▓     ▓▓▓     ▓▓▓▓▓▓▓▓▓  │  <- ASK wall (red aliens)
            │  ▓▓▓▓         ▓▓▓       ▓▓▓▓▓▓▓▓         ▓▓      ▓▓▓▓▓      │
            │  ▓▓           ▓         ▓▓▓▓             ▓       ▓▓▓        │
            └──────────────────────────────────────────────────────────────┘
                                          🚀
            ┌──────────────────────────────────────────────────────────────┐
            │  ░░           ░░        ░░░░░            ░       ░░░        │
            │  ░░░░         ░░░       ░░░░░░░░         ░░      ░░░░░      │
            │  ░░░░░░░░     ░░░░░     ░░░░░░░░░░░░     ░░░     ░░░░░░░░░  │  <- BID wall (green aliens)
            └──────────────────────────────────────────────────────────────┘
        """
        lines = []

        # Header
        if self.kid_mode:
            lines.append(f"═══ ALIEN TAXI ═══  Score: {self.score}  Trips: {len(self.completed_trips)}")
        else:
            pnl_str = f"+${self.total_pnl:.2f}" if self.total_pnl >= 0 else f"-${abs(self.total_pnl):.2f}"
            lines.append(f"═══ ORDER BOOK ROCKET ({self.symbol}) ═══  P&L: {pnl_str}  Trips: {len(self.completed_trips)}")

        # Calculate mid price
        mid_price = (self.best_ask + self.best_bid) / 2 if self.best_ask and self.best_bid else 0

        if not self.kid_mode:
            spread = self.best_ask - self.best_bid if self.best_ask and self.best_bid else 0
            lines.append(f"Mid: ${mid_price:.2f}  Spread: ${spread:.4f}  Tick: {self.tick}")
        lines.append("")

        # Layout constants
        inner_width = width - 4  # Space inside the walls
        num_columns = min(self.num_levels, 12)  # Max columns to show
        col_width = inner_width // num_columns
        max_bar_height = 3  # How many rows each histogram can use

        # Calculate price range for positioning vertical lines
        all_prices = [l.price for l in self.bids + self.asks]
        if all_prices:
            min_price = min(all_prices)
            max_price = max(all_prices)
            price_range = max_price - min_price if max_price > min_price else 1
        else:
            min_price, max_price, price_range = 0, 1, 1

        def price_to_column(price: float) -> int:
            """Convert a price to a column position (0 to inner_width-1)."""
            if price_range == 0:
                return inner_width // 2
            normalized = (price - min_price) / price_range
            return int(normalized * (inner_width - 1))

        # Calculate line positions
        mid_col = price_to_column(mid_price) if mid_price else inner_width // 2
        breakeven_col = price_to_column(self.aliens.pickup_price) if self.aliens else None

        # Calculate popstop/dropstop line positions (last 3 of each)
        # Line thickness: newest = thickest (║), middle = medium (│), oldest = thin (¦)
        popstop_cols = []  # (column, thickness_index) - 0=newest/thickest
        for i, price in enumerate(self.popstop_history[:3]):
            col = price_to_column(price)
            if 0 <= col < inner_width:
                popstop_cols.append((col, i))

        dropstop_cols = []  # (column, thickness_index) - 0=newest/thickest
        for i, price in enumerate(self.dropstop_history[:3]):
            col = price_to_column(price)
            if 0 <= col < inner_width:
                dropstop_cols.append((col, i))

        # Line characters by thickness (newest to oldest)
        # popstop (downtrend start) - use red-ish chars: ╽ ╿ ╵
        # dropstop (uptrend start) - use green-ish chars: ╿ ╽ ╷
        popstop_chars = ['╻', '╷', '·']  # Thick to thin (downtrend = bearish)
        dropstop_chars = ['╹', '╵', '·']  # Thick to thin (uptrend = bullish)

        def add_vertical_lines(row_str: str, is_histogram: bool = False) -> str:
            """Add vertical line markers to a row string."""
            row_list = list(row_str)

            # Draw popstop lines (oldest first so newest overwrites)
            for col, thickness in reversed(popstop_cols):
                if 0 <= col < len(row_list) and row_list[col] == ' ':
                    row_list[col] = popstop_chars[thickness]

            # Draw dropstop lines (oldest first so newest overwrites)
            for col, thickness in reversed(dropstop_cols):
                if 0 <= col < len(row_list) and row_list[col] == ' ':
                    row_list[col] = dropstop_chars[thickness]

            # Current price line (yellow/gold marker)
            if 0 <= mid_col < len(row_list):
                if row_list[mid_col] == ' ':
                    row_list[mid_col] = '│'

            # Breakeven line (shows where you entered)
            if breakeven_col is not None and 0 <= breakeven_col < len(row_list):
                if row_list[breakeven_col] == ' ':
                    row_list[breakeven_col] = '┃'
                elif row_list[breakeven_col] == '│':
                    row_list[breakeven_col] = '╋'  # Intersection

            return ''.join(row_list)

        # TOP WALL (asks = red aliens) - bars hang DOWN from ceiling
        if self.kid_mode:
            lines.append("  RED ALIENS (pick up here, deliver to bottom)")
        else:
            lines.append("  ASKS (sellers)")

        lines.append(f"  ┌{'─' * inner_width}┐")

        # Get ask levels (reversed so best ask is closest to spread)
        asks_to_show = list(reversed(self.asks[:num_columns]))

        # Render ask histogram rows (from ceiling down)
        for row in range(max_bar_height):
            row_chars = []
            for i in range(num_columns):
                if i < len(asks_to_show):
                    level = asks_to_show[i]
                    bar_height = int((level.quantity / self.max_quantity) * max_bar_height) if self.max_quantity > 0 else 0
                    bar_height = max(1, min(max_bar_height, bar_height))
                    # Draw bar from top down
                    if row < bar_height:
                        row_chars.append("▓" * (col_width - 1) + " ")
                    else:
                        row_chars.append(" " * col_width)
                else:
                    row_chars.append(" " * col_width)
            row_content = ''.join(row_chars)[:inner_width]
            row_content = add_vertical_lines(row_content, is_histogram=True)
            lines.append(f"  │{row_content}│")

        lines.append(f"  └{'─' * inner_width}┘")

        # SPREAD section - open space where rocket flies
        spread_rows = 5
        rocket_row = int((1 - self.rocket_y) * (spread_rows - 1))
        rocket_row = max(0, min(spread_rows - 1, rocket_row))

        # Calculate profit direction for arrows
        profit_direction = None  # None, 'up', or 'down'
        if self.aliens:
            if self.aliens.color == AlienColor.GREEN:
                # GREEN = long, profit when price goes UP
                profit_direction = 'up'
            else:
                # RED = short, profit when price goes DOWN
                profit_direction = 'down'

        for row in range(spread_rows):
            if row == rocket_row:
                # Rocket with alien indicator
                if self.aliens:
                    if self.aliens.color == AlienColor.GREEN:
                        rocket_char = "🟢"
                    else:
                        rocket_char = "🔴"
                else:
                    rocket_char = "🚀"

                # Calculate rocket horizontal position based on limit order selection
                if self.limit_order_mode and self.rocket_x > 0:
                    # Offset rocket horizontally to show selected price level
                    # Move left for bid wall selection, right for ask wall selection
                    offset = self.rocket_x * 2  # 2 chars per level
                    if self.rocket_y < 0.3:  # Near bid wall
                        rocket_col = (inner_width // 2) - offset
                    else:  # Near ask wall
                        rocket_col = (inner_width // 2) + offset
                    rocket_col = max(2, min(inner_width - 4, rocket_col))
                else:
                    rocket_col = inner_width // 2 - 1

                left_space = " " * rocket_col
                right_space = " " * (inner_width - rocket_col - 2)
                spread_content = f"{left_space}{rocket_char}{right_space}"
            else:
                spread_content = " " * inner_width

            # Add profit direction arrows on the sides
            if self.aliens and row == spread_rows // 2:
                arrow_list = list(spread_content)
                if profit_direction == 'up':
                    # Show UP arrows on left margin
                    arrow_list[1] = '▲'
                    arrow_list[2] = '▲'
                    arrow_list[3] = '▲'
                elif profit_direction == 'down':
                    # Show DOWN arrows on left margin
                    arrow_list[1] = '▼'
                    arrow_list[2] = '▼'
                    arrow_list[3] = '▼'
                spread_content = ''.join(arrow_list)

            # Add vertical lines to spread area
            spread_content = add_vertical_lines(spread_content)

            # Zone indicators
            zone_label = ""
            if row == 0 and self.rocket_y > 0.7:
                if self.aliens and self.aliens.color == AlienColor.GREEN:
                    zone_label = " ▲▲▲ SELL HERE FOR PROFIT ▲▲▲"
                elif self.aliens is None:
                    zone_label = " PICKUP RED!"
            elif row == spread_rows - 1 and self.rocket_y < 0.3:
                if self.aliens and self.aliens.color == AlienColor.RED:
                    zone_label = " ▼▼▼ BUY HERE FOR PROFIT ▼▼▼"
                elif self.aliens is None:
                    zone_label = " PICKUP GREEN!"

            lines.append(f"   {spread_content} {zone_label}")

        # BOTTOM WALL (bids = green aliens) - bars grow UP from floor
        lines.append(f"  ┌{'─' * inner_width}┐")

        bids_to_show = self.bids[:num_columns]

        # Render bid histogram rows (from spread down, bars grow up)
        for row in range(max_bar_height):
            row_chars = []
            for i in range(num_columns):
                if i < len(bids_to_show):
                    level = bids_to_show[i]
                    bar_height = int((level.quantity / self.max_quantity) * max_bar_height) if self.max_quantity > 0 else 0
                    bar_height = max(1, min(max_bar_height, bar_height))
                    # Draw bar from bottom up (so check if this row should have bar)
                    rows_from_bottom = max_bar_height - 1 - row
                    if rows_from_bottom < bar_height:
                        row_chars.append("░" * (col_width - 1) + " ")
                    else:
                        row_chars.append(" " * col_width)
                else:
                    row_chars.append(" " * col_width)
            row_content = ''.join(row_chars)[:inner_width]
            row_content = add_vertical_lines(row_content, is_histogram=True)
            lines.append(f"  │{row_content}│")

        lines.append(f"  └{'─' * inner_width}┘")

        if self.kid_mode:
            lines.append("  GREEN ALIENS (pick up here, deliver to top)")
        else:
            lines.append("  BIDS (buyers)")

        # Price line legend
        if self.aliens:
            in_profit = False
            if self.aliens.color == AlienColor.GREEN:
                in_profit = mid_price > self.aliens.pickup_price
            else:
                in_profit = mid_price < self.aliens.pickup_price

            profit_indicator = "✓ PROFIT" if in_profit else "✗ LOSS"
            if self.kid_mode:
                lines.append(f"  │ = current   ┃ = entry  [{profit_indicator}]")
            else:
                lines.append(f"  │ = ${mid_price:.2f}   ┃ = entry ${self.aliens.pickup_price:.2f}  [{profit_indicator}]")
        else:
            lines.append(f"  │ = current price (${mid_price:.2f})")

        # Stop level legend (if we have any history)
        if self.dropstop_history or self.popstop_history:
            stop_legend = "  "
            if self.dropstop_history:
                stop_legend += "╹╵· = uptrend stops (newest→oldest)  "
            if self.popstop_history:
                stop_legend += "╻╷· = downtrend stops"
            lines.append(stop_legend)

        lines.append("")

        # Cargo status with profit direction indicator
        if self.aliens:
            color_name = "GREEN" if self.aliens.color == AlienColor.GREEN else "RED"
            dest = "TOP" if self.aliens.color == AlienColor.GREEN else "BOTTOM"
            direction_arrow = "▲▲▲" if self.aliens.color == AlienColor.GREEN else "▼▼▼"

            # Calculate current P&L
            if self.aliens.color == AlienColor.GREEN:
                current_pnl = (mid_price - self.aliens.pickup_price) * self.aliens.count
            else:
                current_pnl = (self.aliens.pickup_price - mid_price) * self.aliens.count
            pnl_str = f"+${current_pnl:.2f}" if current_pnl >= 0 else f"-${abs(current_pnl):.2f}"

            if self.kid_mode:
                lines.append(f"{direction_arrow} Carrying {self.aliens.count} {color_name} aliens! Fly {dest}! {direction_arrow}")
            else:
                lines.append(f"{direction_arrow} {color_name} {self.aliens.count}x @ ${self.aliens.pickup_price:.2f}  Current: {pnl_str}  {direction_arrow}")
        elif self.pending_limit_order:
            # Show pending limit order status
            order = self.pending_limit_order
            if self.kid_mode:
                lines.append(f"⏳ Waiting for order @ ${order['price']:.2f}... [C] to cancel")
            else:
                lines.append(f"⏳ LIMIT {order['side'].upper()} {order['count']}x @ ${order['price']:.2f}  [C] to cancel")
        else:
            if self.kid_mode:
                lines.append("Fly to a wall and press SPACE to pick up aliens!")
            else:
                lines.append("No cargo - fly to either wall to pickup")

        # Limit order mode indicator
        if self.limit_order_mode and self.rocket_x > 0:
            near_top = self.rocket_y > 0.7
            near_bottom = self.rocket_y < 0.3
            if near_bottom and self.bids and self.rocket_x < len(self.bids):
                level = self.bids[self.rocket_x]
                lines.append(f">>> LIMIT ORDER: ${level.price:.2f}  [←/→] adjust  [SPACE] place <<<")
            elif near_top and self.asks and self.rocket_x < len(self.asks):
                level = self.asks[self.rocket_x]
                lines.append(f">>> LIMIT ORDER: ${level.price:.2f}  [←/→] adjust  [SPACE] place <<<")

        # Message
        if self.message and (self.tick - self.message_tick) < 15:
            lines.append(f"*** {self.message} ***")

        lines.append("")
        if self.kid_mode:
            lines.append("[↑/↓] Fly  [←/→] Limit  [SPACE] Action  [C] Cancel  [R] Restart  [Q] Quit")
        else:
            lines.append("CONTROLS: [↑/↓] Move  [←/→] Limit price  [SPACE] Order  [C] Cancel  [R] Reset  [Q] Quit")

        if self._stream_started:
            lines.append(f"LIVE - {self.symbol}")
        elif not self.kid_mode:
            lines.append("SIMULATED - run with --live for real data")

        return "\n".join(lines)

    def reset(self):
        """Reset game state."""
        self.state = GameState.PLAYING
        self.tick = 0
        self.rocket_y = 0.5
        self.rocket_x = 0
        self.limit_order_mode = False
        self.pending_limit_order = None
        self.aliens = None
        self.completed_trips = []
        self.total_pnl = 0.0
        self.score = 0
        self.message = None


def demo_live():
    """Demo the alien taxi game."""
    print("=" * 70)
    print("ALIEN TAXI DEMO")
    print("=" * 70)
    print()

    game = LiveOrderBookRocket(symbol="BTCUSDT", num_levels=6, kid_mode=True)

    # Use simulated data for demo
    game.set_simulated_book(mid_price=50000.0, spread=10.0)

    print(game.render())
    print()

    # Demo 1: Pick up GREEN aliens from bottom, deliver to top
    print("--- Flying DOWN to pick up GREEN aliens ---")
    for _ in range(6):
        game.input_down()
        game.tick_update()
    print(game.render())
    print()

    print("--- Picking up GREEN aliens ---")
    game.input_action()
    print(game.render())
    print()

    print("--- Flying UP to deliver GREEN aliens ---")
    for _ in range(12):
        game.input_up()
        game.tick_update()
    print(game.render())
    print()

    print("--- Delivering GREEN aliens ---")
    game.input_action()
    print(game.render())
    print()

    # Demo 2: Pick up RED aliens from top, deliver to bottom
    print("=" * 70)
    print("Now let's try RED aliens!")
    print("=" * 70)
    print()

    print("--- Picking up RED aliens from top ---")
    game.input_action()  # Already at top
    print(game.render())
    print()

    print("--- Flying DOWN to deliver RED aliens ---")
    for _ in range(12):
        game.input_down()
        game.tick_update()
    print(game.render())
    print()

    print("--- Delivering RED aliens ---")
    game.input_action()
    print(game.render())


def demo():
    """Demo the order book rocket game."""
    print("=" * 60)
    print("ORDER BOOK ROCKET DEMO")
    print("=" * 60)
    print()

    game = OrderBookRocket()
    print(game.render())
    print()

    # Move up to pickup zone (but not into crash zone)
    print("--- Moving toward ASK wall (top) ---")
    # Move up 4 steps (0.05 each = 0.20 total, ending at y=0.70)
    for i in range(4):
        game.input_up()
        game.tick_update()
    print(f"Position: y={game.rocket.y:.2f}")
    print(game.render())
    print()

    # Pickup cargo
    print("--- Picking up cargo (SPACE) ---")
    success = game.input_action()
    print(f"Pickup result: {success}")
    if success:
        print(game.render())
        print()

        # Fly down to drop zone
        print("--- Flying to BID wall (bottom) ---")
        for _ in range(8):
            game.input_down()
            game.tick_update()
            if game.state != GameState.PLAYING:
                break
        print(f"Position: y={game.rocket.y:.2f}")
        print(game.render())
        print()

        # Drop off cargo
        if game.state == GameState.PLAYING:
            print("--- Dropping off cargo (SPACE) ---")
            success = game.input_action()
            print(f"Dropoff result: {success}")
            print(game.render())
    else:
        print("Could not pick up cargo. Game state:", game.state)
        print(game.render())


def play_interactive():
    """Play the game interactively in the terminal using curses."""
    import curses
    import time

    game = OrderBookRocket()
    final_pnl = 0.0
    final_trips = 0

    def main(stdscr):
        nonlocal final_pnl, final_trips

        # Setup curses
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(True)  # Non-blocking input
        stdscr.timeout(100)  # 100ms refresh

        last_tick = time.time()
        tick_interval = 0.5  # Update game state every 0.5 seconds

        while True:
            # Handle input
            try:
                key = stdscr.getch()
            except:
                key = -1

            if key == ord('q') or key == ord('Q'):
                break
            elif key == curses.KEY_UP or key == ord('w') or key == ord('k'):
                game.input_up()
            elif key == curses.KEY_DOWN or key == ord('s') or key == ord('j'):
                game.input_down()
            elif key == ord(' '):  # Space
                game.input_action()
            elif key == ord('r') or key == ord('R'):
                game.reset()

            # Update game state periodically
            now = time.time()
            if now - last_tick >= tick_interval:
                game.tick_update()
                last_tick = now

            # Render
            stdscr.clear()
            lines = game.render(height=20, width=60).split('\n')
            for i, line in enumerate(lines):
                try:
                    stdscr.addstr(i, 0, line[:curses.COLS-1])
                except curses.error:
                    pass  # Ignore if line is too long
            stdscr.refresh()

            # Check for game over
            if game.state == GameState.GAME_OVER:
                # Wait for R to restart or Q to quit
                stdscr.nodelay(False)
                while True:
                    key = stdscr.getch()
                    if key == ord('r') or key == ord('R'):
                        game.reset()
                        stdscr.nodelay(True)
                        break
                    elif key == ord('q') or key == ord('Q'):
                        final_pnl = game.total_pnl
                        final_trips = len(game.completed_trips)
                        return

        final_pnl = game.total_pnl
        final_trips = len(game.completed_trips)

    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass

    print("\nThanks for playing Order Book Rocket!")
    print(f"Final P&L: ${final_pnl:.2f}")
    print(f"Completed trips: {final_trips}")


def play_live_interactive(symbol: str = "BTCUSDT", live: bool = False, kid_mode: bool = True):
    """Play the histogram version interactively."""
    import curses
    import time

    game = LiveOrderBookRocket(symbol=symbol, num_levels=6, kid_mode=kid_mode)

    if live:
        print(f"Connecting to Binance {symbol}...")
        if not game.connect():
            print("Using simulated data instead.")
            game.set_simulated_book(mid_price=50000.0, spread=10.0)
            input("Press Enter to continue...")
    else:
        game.set_simulated_book(mid_price=50000.0, spread=10.0)

    final_pnl = 0.0
    final_trips = 0

    def main(stdscr):
        nonlocal final_pnl, final_trips

        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(100)

        last_tick = time.time()
        tick_interval = 0.3

        while True:
            try:
                key = stdscr.getch()
            except:
                key = -1

            if key == ord('q') or key == ord('Q'):
                break
            elif key == curses.KEY_UP or key == ord('w') or key == ord('k'):
                game.input_up()
            elif key == curses.KEY_DOWN or key == ord('s') or key == ord('j'):
                game.input_down()
            elif key == curses.KEY_LEFT or key == ord('a') or key == ord('h'):
                game.input_left()
            elif key == curses.KEY_RIGHT or key == ord('d') or key == ord('l'):
                game.input_right()
            elif key == ord(' '):
                game.input_action()
            elif key == ord('c') or key == ord('C'):
                game.cancel_limit_order()
            elif key == ord('r') or key == ord('R'):
                game.reset()
                if not live:
                    game.set_simulated_book(mid_price=50000.0, spread=10.0)

            now = time.time()
            if now - last_tick >= tick_interval:
                game.tick_update()
                last_tick = now

            stdscr.clear()
            lines = game.render(height=28, width=70).split('\n')
            for i, line in enumerate(lines):
                try:
                    stdscr.addstr(i, 0, line[:curses.COLS-1])
                except curses.error:
                    pass
            stdscr.refresh()

        final_pnl = game.total_pnl
        final_trips = len(game.completed_trips)

    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
    finally:
        game.disconnect()

    print("\nThanks for playing Order Book Rocket!")
    print(f"Final P&L: ${final_pnl:.2f}")
    print(f"Completed trips: {final_trips}")


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]

    if "--demo" in args:
        demo()
    elif "--demo-live" in args or "--histogram" in args:
        demo_live()
    elif "--live" in args:
        # Extract symbol if provided
        symbol = "BTCUSDT"
        kid_mode = "--pro" not in args
        for arg in args:
            if arg.startswith("--symbol="):
                symbol = arg.split("=")[1].upper()
            elif not arg.startswith("--"):
                symbol = arg.upper()

        print(f"Alien Taxi - LIVE MODE ({symbol})")
        print()
        print("Connecting to Binance WebSocket...")
        print("This requires: pip install python-binance")
        print()
        input("Press Enter to start...")
        play_live_interactive(symbol=symbol, live=True, kid_mode=kid_mode)
    else:
        print()
        print("  ╔═══════════════════════════════════════╗")
        print("  ║         🚀 ALIEN TAXI 🚀              ║")
        print("  ╠═══════════════════════════════════════╣")
        print("  ║  Pick up aliens and deliver them!     ║")
        print("  ║                                       ║")
        print("  ║  🟢 GREEN aliens at BOTTOM            ║")
        print("  ║     → Deliver to TOP                  ║")
        print("  ║                                       ║")
        print("  ║  🔴 RED aliens at TOP                 ║")
        print("  ║     → Deliver to BOTTOM               ║")
        print("  ║                                       ║")
        print("  ║  The walls show how many aliens       ║")
        print("  ║  are waiting at each level!           ║")
        print("  ╚═══════════════════════════════════════╝")
        print()
        print("Controls:")
        print("  [UP/DOWN] or [W/S]  - Fly your rocket")
        print("  [SPACE]             - Pickup/Dropoff aliens")
        print("  [R]                 - Restart")
        print("  [Q]                 - Quit")
        print()
        print("Options:")
        print("  --live              Connect to real Binance data")
        print("  --live ETHUSDT      Use a different trading pair")
        print("  --pro               Show trading details instead of kid mode")
        print("  --histogram         Run demo mode")
        print()
        input("Press Enter to play...")
        play_live_interactive(symbol="BTCUSDT", live=False, kid_mode=True)
