"""
Egg Trading Floor - Full Order Book Simulation

A complete trading simulation with:
- 10 price levels ($100-$109)
- Green eggs (LONG) and Red eggs (SHORT) at each level
- Pick up, carry, set down (limit), or market sell
- Competing traders that grab liquidity
- Real P&L tracking

THE TRADING FLOOR:

  PRICE │ BIDS (buy longs) │ ASKS (buy shorts) │ YOUR ORDERS
  ──────┼──────────────────┼───────────────────┼─────────────
  $109  │ ●●               │ ○○○               │
  $108  │ ●●●●             │ ○○                │ [2● limit]
  $107  │ ●●●              │ ○○○○              │
  $106  │ ●●●●●            │ ○○○               │
  $105  │══ PRICE ════════════════════════════│══════════════
  $104  │ ●●●●             │ ○○○○○             │
  $103  │ ●●               │ ○○○               │ ◆ YOU
  $102  │ ●                │ ○○○○              │
  $101  │ ●●●              │ ○                 │
  $100  │ ●●               │ ○○                │

ACTIONS:
  ↑/↓   Move to price level
  ←/→   Move between BIDS/ASKS columns
  SPACE Pick up eggs (open position)
  D     Drop/Set limit order (leave eggs at price)
  M     Market order (sell NOW at current price)

"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import random


class Side(Enum):
    """Which side of the book."""
    BID = "bid"   # Green/Long side
    ASK = "ask"   # Red/Short side


@dataclass
class PriceLevel:
    """A single price level on the trading floor."""
    price: float           # e.g., $103
    bid_eggs: int = 0      # Green eggs available (people want to buy longs)
    ask_eggs: int = 0      # Red eggs available (people want to buy shorts)
    your_bid_limit: int = 0  # Your limit orders on bid side
    your_ask_limit: int = 0  # Your limit orders on ask side


@dataclass
class Position:
    """Your current position (eggs you're carrying)."""
    side: Side            # BID (long) or ASK (short)
    quantity: int         # How many eggs
    entry_price: float    # Price where you picked them up


@dataclass
class LimitOrder:
    """A limit order you've placed."""
    side: Side
    price: float
    quantity: int
    tick_placed: int


@dataclass
class Trade:
    """A completed trade."""
    side: Side
    quantity: int
    entry_price: float
    exit_price: float
    pnl: float
    order_type: str  # "market" or "limit"


@dataclass
class Portfolio:
    """Your trading account."""
    cash: float = 10000.0
    total_pnl: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    limit_orders: List[LimitOrder] = field(default_factory=list)


class EggTradingFloor:
    """
    The complete egg trading floor simulation.
    """

    def __init__(
        self,
        base_price: float = 100.0,
        num_levels: int = 10,
        price_step: float = 1.0,
    ):
        self.base_price = base_price
        self.num_levels = num_levels
        self.price_step = price_step

        # Price levels (bottom to top)
        self.levels: List[PriceLevel] = []
        for i in range(num_levels):
            price = base_price + i * price_step
            self.levels.append(PriceLevel(price=price))

        # Current market price (index into levels)
        self.price_index: int = num_levels // 2  # Start in middle

        # Player state
        self.player_level: int = num_levels // 2  # Which price row
        self.player_side: Side = Side.BID  # BID or ASK column
        self.carrying: Optional[Position] = None

        # Portfolio
        self.portfolio = Portfolio()

        # Game state
        self.tick_count = 0
        self.game_over = False

        # Initialize order book
        self._init_order_book()

    def _init_order_book(self):
        """Initialize eggs at each price level."""
        for i, level in enumerate(self.levels):
            # More liquidity near current price
            distance = abs(i - self.price_index)
            base_qty = max(1, 5 - distance)

            level.bid_eggs = base_qty + random.randint(0, 2)
            level.ask_eggs = base_qty + random.randint(0, 2)

    @property
    def current_price(self) -> float:
        """Get current market price."""
        return self.levels[self.price_index].price

    @property
    def player_price(self) -> float:
        """Get price at player's current level."""
        return self.levels[self.player_level].price

    def input_up(self):
        """Move up one price level."""
        if not self.game_over and self.player_level < self.num_levels - 1:
            self.player_level += 1

    def input_down(self):
        """Move down one price level."""
        if not self.game_over and self.player_level > 0:
            self.player_level -= 1

    def input_left(self):
        """Move to BID (green/long) side."""
        if not self.game_over:
            self.player_side = Side.BID

    def input_right(self):
        """Move to ASK (red/short) side."""
        if not self.game_over:
            self.player_side = Side.ASK

    def input_pickup(self, quantity: int = 1):
        """Pick up eggs from current level (open position)."""
        if self.game_over or self.carrying is not None:
            return False

        level = self.levels[self.player_level]

        if self.player_side == Side.BID:
            available = level.bid_eggs
        else:
            available = level.ask_eggs

        qty = min(quantity, available)
        if qty <= 0:
            return False

        # Take eggs
        if self.player_side == Side.BID:
            level.bid_eggs -= qty
        else:
            level.ask_eggs -= qty

        self.carrying = Position(
            side=self.player_side,
            quantity=qty,
            entry_price=self.player_price,
        )
        return True

    def input_drop(self):
        """Drop eggs as limit order at current level."""
        if self.game_over or self.carrying is None:
            return False

        level = self.levels[self.player_level]

        # Place limit order
        if self.carrying.side == Side.BID:
            level.your_bid_limit += self.carrying.quantity
        else:
            level.your_ask_limit += self.carrying.quantity

        # Record limit order
        self.portfolio.limit_orders.append(LimitOrder(
            side=self.carrying.side,
            price=self.player_price,
            quantity=self.carrying.quantity,
            tick_placed=self.tick_count,
        ))

        self.carrying = None
        return True

    def input_market_sell(self):
        """Market sell - dump eggs at current market price."""
        if self.game_over or self.carrying is None:
            return False

        # Execute at current market price
        exit_price = self.current_price
        entry_price = self.carrying.entry_price
        qty = self.carrying.quantity

        # Calculate P&L
        if self.carrying.side == Side.BID:
            # Long position: profit if exit > entry
            pnl = (exit_price - entry_price) * qty
        else:
            # Short position: profit if exit < entry
            pnl = (entry_price - exit_price) * qty

        # Record trade
        trade = Trade(
            side=self.carrying.side,
            quantity=qty,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=pnl,
            order_type="market",
        )
        self.portfolio.trades.append(trade)
        self.portfolio.total_pnl += pnl
        self.portfolio.cash += pnl

        self.carrying = None
        return True

    def tick(self) -> bool:
        """Advance simulation by one tick."""
        if self.game_over:
            return False

        self.tick_count += 1

        # Move price randomly
        self._update_price()

        # Competitors grab some eggs
        self._simulate_competition()

        # Check if any limit orders filled
        self._check_limit_fills()

        # Replenish some liquidity
        self._replenish_liquidity()

        return True

    def _update_price(self):
        """Random walk price movement."""
        move = random.choice([-1, 0, 0, 1])  # Slight upward bias possible
        self.price_index = max(0, min(self.num_levels - 1, self.price_index + move))

    def _simulate_competition(self):
        """Other traders grab eggs."""
        # Randomly remove some eggs near the price
        for i in range(self.num_levels):
            distance = abs(i - self.price_index)
            if distance <= 2 and random.random() < 0.3:
                level = self.levels[i]
                if level.bid_eggs > 0 and random.random() < 0.5:
                    level.bid_eggs -= 1
                if level.ask_eggs > 0 and random.random() < 0.5:
                    level.ask_eggs -= 1

    def _check_limit_fills(self):
        """Check if price crossed any of your limit orders."""
        for i, level in enumerate(self.levels):
            # Check bid limits (you're selling longs)
            if level.your_bid_limit > 0:
                # Filled if price >= limit price (someone bought your longs)
                if i <= self.price_index:
                    qty = level.your_bid_limit
                    # Find the original entry (simplified: use level price)
                    pnl = (level.price - self.base_price) * qty * 0.5  # Simplified

                    trade = Trade(
                        side=Side.BID,
                        quantity=qty,
                        entry_price=level.price - 1,  # Assume bought $1 lower
                        exit_price=level.price,
                        pnl=pnl,
                        order_type="limit",
                    )
                    self.portfolio.trades.append(trade)
                    self.portfolio.total_pnl += pnl
                    self.portfolio.cash += pnl
                    level.your_bid_limit = 0

            # Check ask limits (you're selling shorts)
            if level.your_ask_limit > 0:
                # Filled if price <= limit price
                if i >= self.price_index:
                    qty = level.your_ask_limit
                    pnl = (self.base_price - level.price) * qty * 0.5

                    trade = Trade(
                        side=Side.ASK,
                        quantity=qty,
                        entry_price=level.price + 1,
                        exit_price=level.price,
                        pnl=pnl,
                        order_type="limit",
                    )
                    self.portfolio.trades.append(trade)
                    self.portfolio.total_pnl += pnl
                    self.portfolio.cash += pnl
                    level.your_ask_limit = 0

    def _replenish_liquidity(self):
        """Add some eggs back to the book."""
        for i, level in enumerate(self.levels):
            distance = abs(i - self.price_index)
            if random.random() < 0.2:
                max_qty = max(1, 5 - distance)
                if level.bid_eggs < max_qty:
                    level.bid_eggs += 1
                if level.ask_eggs < max_qty:
                    level.ask_eggs += 1

    def render(self) -> str:
        """Render the trading floor."""
        lines = []

        # Header
        pnl_str = f"+${self.portfolio.total_pnl:.2f}" if self.portfolio.total_pnl >= 0 else f"-${abs(self.portfolio.total_pnl):.2f}"
        lines.append(f"═══ EGG TRADING FLOOR ═══  P&L: {pnl_str}  Cash: ${self.portfolio.cash:.2f}")
        lines.append(f"Trades: {len(self.portfolio.trades)}  |  Tick: {self.tick_count}")
        lines.append("")

        # Column headers
        lines.append("  PRICE │ LONGS (●)  │ SHORTS (○) │ YOUR ORDERS")
        lines.append("  ──────┼────────────┼────────────┼─────────────")

        # Render each level (top to bottom = high to low price)
        for i in range(self.num_levels - 1, -1, -1):
            level = self.levels[i]

            # Price column
            price_str = f"${level.price:5.0f}"

            # Is this the current market price?
            is_market_price = (i == self.price_index)

            # Bid eggs column
            bid_eggs = "●" * level.bid_eggs + " " * (5 - level.bid_eggs)

            # Ask eggs column
            ask_eggs = "○" * level.ask_eggs + " " * (5 - level.ask_eggs)

            # Your orders column
            your_orders = ""
            if level.your_bid_limit > 0:
                your_orders += f"{level.your_bid_limit}● "
            if level.your_ask_limit > 0:
                your_orders += f"{level.your_ask_limit}○ "

            # Player position
            player_here = (i == self.player_level)
            if player_here:
                if self.player_side == Side.BID:
                    bid_eggs = "◆" + bid_eggs[1:]  # Replace first char with player
                else:
                    ask_eggs = "◆" + ask_eggs[1:]

            # Format the row
            if is_market_price:
                row = f"►{price_str}│═{bid_eggs}═│═{ask_eggs}═│ {your_orders}"
            else:
                row = f" {price_str} │ {bid_eggs} │ {ask_eggs} │ {your_orders}"

            lines.append(row)

        lines.append("  ──────┴────────────┴────────────┴─────────────")
        lines.append("")

        # Player status
        side_name = "LONG (●)" if self.player_side == Side.BID else "SHORT (○)"
        lines.append(f"You are at ${self.player_price:.0f} on {side_name} side")

        if self.carrying:
            carry_side = "LONG" if self.carrying.side == Side.BID else "SHORT"
            carry_symbol = "●" if self.carrying.side == Side.BID else "○"
            lines.append(f"CARRYING: {self.carrying.quantity}{carry_symbol} {carry_side} from ${self.carrying.entry_price:.0f}")

            # P&L preview
            if self.carrying.side == Side.BID:
                preview_pnl = (self.current_price - self.carrying.entry_price) * self.carrying.quantity
            else:
                preview_pnl = (self.carrying.entry_price - self.current_price) * self.carrying.quantity

            pnl_str = f"+${preview_pnl:.2f}" if preview_pnl >= 0 else f"-${abs(preview_pnl):.2f}"
            lines.append(f"If MARKET SELL now: {pnl_str}")
        else:
            # Show what's available at current level
            level = self.levels[self.player_level]
            if self.player_side == Side.BID:
                avail = level.bid_eggs
                lines.append(f"Available: {avail}● LONG eggs")
            else:
                avail = level.ask_eggs
                lines.append(f"Available: {avail}○ SHORT eggs")

        lines.append("")
        lines.append("CONTROLS:")
        lines.append("  [↑/↓] Change price level   [←/→] Switch LONG/SHORT side")
        lines.append("  [1-5] Pick up eggs         [D] Drop as limit order")
        lines.append("  [M]   Market sell NOW      [Q] Quit  [R] Reset")

        return "\n".join(lines)

    def reset(self):
        """Reset the game."""
        self.price_index = self.num_levels // 2
        self.player_level = self.num_levels // 2
        self.player_side = Side.BID
        self.carrying = None
        self.tick_count = 0
        self.game_over = False

        for level in self.levels:
            level.your_bid_limit = 0
            level.your_ask_limit = 0

        self._init_order_book()


def demo():
    """Demo the trading floor."""
    print("=" * 60)
    print("EGG TRADING FLOOR DEMO")
    print("=" * 60)
    print()

    floor = EggTradingFloor(base_price=100, num_levels=10)

    print("Initial state:")
    print(floor.render())
    print()

    # Pick up some eggs
    print("Picking up 3 LONG eggs...")
    floor.input_pickup(3)
    print(floor.render())
    print()

    # Move up
    print("Moving up to higher price...")
    floor.input_up()
    floor.input_up()
    floor.tick()
    print(floor.render())
    print()

    # Drop as limit
    print("Dropping as limit order...")
    floor.input_drop()
    floor.tick()
    print(floor.render())


if __name__ == "__main__":
    demo()
