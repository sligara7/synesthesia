"""
Egg Trader v2 - Position Sizing and Order Book Liquidity

Enhanced trading game with:
- Egg piles of different sizes (1-5 eggs = position size)
- Order book showing where buyers/sellers exist
- Partial fills (may not be able to sell all eggs at same price)
- Current price line visible

THE GAME:
    ┌─────────────────────────────────────────────────────────┐
    │░░░░░░░░░░░░░░░░░░ RESISTANCE ░░░░░░░░░░░░░░░░░░░░░░░░░░│
    │                                                         │
    │ ○         (above price = premium)              ○        │
    │ ○○                                            ○○        │
    │ ○○○  ════════════ PRICE ════════════════════  ○○○      │
    │ ○○○○                                          ○○        │ ← only 2 buyers here!
    │ ○○○○○   (below price = discount)              (none)    │ ← no buyers!
    │                         ◆                               │
    │                                                         │
    │░░░░░░░░░░░░░░░░░░░ SUPPORT ░░░░░░░░░░░░░░░░░░░░░░░░░░░│
    └─────────────────────────────────────────────────────────┘

RULES:
- LEFT: Pick up egg piles (1-5 eggs)
  - More eggs = bigger position = more risk/reward
  - Upper eggs = LONG, Lower eggs = SHORT

- RIGHT: Deposit eggs to close position
  - Order book shows liquidity at each price level
  - If you have 5 eggs but only 2 buyers, you must split:
    - Sell 2 at good price
    - Fly to another level to sell remaining 3

- PRICE LINE: Shows current market price
  - LONG eggs above price line when depositing = PROFIT
  - SHORT eggs below price line when depositing = PROFIT

"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum
import random


class EggType(Enum):
    """Type of egg (position direction)."""
    NONE = "none"
    LONG = "long"    # Green - bet price goes UP
    SHORT = "short"  # Red - bet price goes DOWN


@dataclass
class EggPile:
    """A pile of eggs at a price level."""
    egg_type: EggType
    count: int              # 1-5 eggs
    price_level: float      # Y position (0-1 normalized)

    def symbol(self) -> str:
        """Get display symbol for this pile."""
        if self.egg_type == EggType.LONG:
            symbols = ["●", "●●", "●●●", "●●●●", "●●●●●"]
        else:
            symbols = ["○", "○○", "○○○", "○○○○", "○○○○○"]
        return symbols[min(self.count - 1, 4)]


@dataclass
class OrderBookLevel:
    """A level in the order book (where you can sell)."""
    price_level: float      # Y position (0-1)
    bid_size: int           # How many eggs buyers want (0-5)
    ask_size: int           # How many eggs sellers have (0-5)


@dataclass
class CarriedEggs:
    """Eggs being carried by the player."""
    egg_type: EggType
    count: int
    pickup_price: float
    pickup_time: int


@dataclass
class Trade:
    """A completed trade."""
    egg_type: EggType
    count: int
    entry_price: float
    exit_price: float
    pnl: float
    tick: int


@dataclass
class Portfolio:
    """Track trading performance."""
    starting_cash: float = 10000.0
    cash: float = 10000.0
    total_pnl: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    total_eggs_traded: int = 0


class EggTraderV2:
    """
    Enhanced Egg Trader with position sizing and order book.
    """

    def __init__(
        self,
        width: int = 60,
        height: int = 16,
        step_size: float = 0.04,
        price_per_egg: float = 100.0,  # $ value per egg
    ):
        self.width = width
        self.height = height
        self.step_size = step_size
        self.price_per_egg = price_per_egg

        # Player position
        self.player_x: float = 0.5
        self.player_y: float = 0.5

        # What player is carrying
        self.carrying: Optional[CarriedEggs] = None

        # Market state
        self.current_price: float = 0.5      # 0-1, where in the cave
        self.support: float = 0.15
        self.resistance: float = 0.85

        # Egg piles on left (pickup zone)
        self.long_piles: List[EggPile] = []   # Upper half
        self.short_piles: List[EggPile] = []  # Lower half

        # Order book on right (deposit zone)
        self.order_book: List[OrderBookLevel] = []

        # Portfolio tracking
        self.portfolio = Portfolio()

        # Game state
        self.tick_count = 0
        self.game_over = False

        # Initialize market
        self._init_piles()
        self._init_order_book()

    def _init_piles(self):
        """Initialize egg piles on the left side."""
        # Long piles (upper half, above price)
        self.long_piles = [
            EggPile(EggType.LONG, 1, 0.85),   # 1 egg at top
            EggPile(EggType.LONG, 2, 0.75),   # 2 eggs
            EggPile(EggType.LONG, 3, 0.65),   # 3 eggs
            EggPile(EggType.LONG, 4, 0.55),   # 4 eggs
            EggPile(EggType.LONG, 5, 0.52),   # 5 eggs (just above price)
        ]

        # Short piles (lower half, below price)
        self.short_piles = [
            EggPile(EggType.SHORT, 5, 0.48),  # 5 eggs (just below price)
            EggPile(EggType.SHORT, 4, 0.45),  # 4 eggs
            EggPile(EggType.SHORT, 3, 0.35),  # 3 eggs
            EggPile(EggType.SHORT, 2, 0.25),  # 2 eggs
            EggPile(EggType.SHORT, 1, 0.15),  # 1 egg at bottom
        ]

    def _init_order_book(self):
        """Initialize order book (liquidity on right side)."""
        self.order_book = []

        # Create levels from 0.15 to 0.85
        for i in range(8):
            price = 0.15 + i * 0.1

            # Liquidity is highest near current price, lower at extremes
            distance_from_price = abs(price - self.current_price)
            base_liquidity = max(1, 5 - int(distance_from_price * 10))

            # Add some randomness
            bid_size = max(0, base_liquidity + random.randint(-1, 1))

            self.order_book.append(OrderBookLevel(
                price_level=price,
                bid_size=min(5, bid_size),
                ask_size=0,  # We only care about bids for selling
            ))

    def _update_order_book(self):
        """Update order book liquidity (shifts over time)."""
        for level in self.order_book:
            # Random walk
            change = random.choice([-1, 0, 0, 1])
            level.bid_size = max(0, min(5, level.bid_size + change))

            # More liquidity near price
            distance = abs(level.price_level - self.current_price)
            if distance < 0.1 and level.bid_size < 3:
                level.bid_size += 1

    def input_up(self):
        """Move up."""
        if not self.game_over:
            self.player_y = min(0.95, self.player_y + self.step_size)

    def input_down(self):
        """Move down."""
        if not self.game_over:
            self.player_y = max(0.05, self.player_y - self.step_size)

    def input_left(self):
        """Move left (toward pickup)."""
        if not self.game_over:
            self.player_x = max(0.0, self.player_x - self.step_size)

    def input_right(self):
        """Move right (toward deposit)."""
        if not self.game_over:
            self.player_x = min(1.0, self.player_x + self.step_size)

    def input_action(self):
        """Pick up or deposit eggs."""
        if self.game_over:
            return

        # In pickup zone (left side)?
        if self.player_x < 0.2 and self.carrying is None:
            self._try_pickup()

        # In deposit zone (right side)?
        elif self.player_x > 0.8 and self.carrying is not None:
            self._try_deposit()

    def _try_pickup(self):
        """Try to pick up eggs from a pile."""
        # Find nearest pile
        all_piles = self.long_piles + self.short_piles

        for pile in all_piles:
            if abs(pile.price_level - self.player_y) < 0.08:
                self.carrying = CarriedEggs(
                    egg_type=pile.egg_type,
                    count=pile.count,
                    pickup_price=pile.price_level,
                    pickup_time=self.tick_count,
                )
                return

    def _try_deposit(self):
        """Try to deposit eggs at current price level."""
        if self.carrying is None:
            return

        # Find order book level near player
        for level in self.order_book:
            if abs(level.price_level - self.player_y) < 0.08:
                eggs_to_sell = min(self.carrying.count, level.bid_size)

                if eggs_to_sell > 0:
                    # Calculate P&L
                    if self.carrying.egg_type == EggType.LONG:
                        # Long: profit if sell price > buy price
                        pnl = (self.player_y - self.carrying.pickup_price) * eggs_to_sell * self.price_per_egg * 10
                    else:
                        # Short: profit if sell price < buy price
                        pnl = (self.carrying.pickup_price - self.player_y) * eggs_to_sell * self.price_per_egg * 10

                    # Record trade
                    trade = Trade(
                        egg_type=self.carrying.egg_type,
                        count=eggs_to_sell,
                        entry_price=self.carrying.pickup_price,
                        exit_price=self.player_y,
                        pnl=pnl,
                        tick=self.tick_count,
                    )
                    self.portfolio.trades.append(trade)
                    self.portfolio.total_pnl += pnl
                    self.portfolio.cash += pnl
                    self.portfolio.total_eggs_traded += eggs_to_sell

                    # Reduce order book liquidity
                    level.bid_size -= eggs_to_sell

                    # Update carrying
                    self.carrying.count -= eggs_to_sell
                    if self.carrying.count <= 0:
                        self.carrying = None

                    return

        # No liquidity at this level!
        pass

    def tick(self) -> bool:
        """Advance game by one tick."""
        if self.game_over:
            return False

        self.tick_count += 1

        # Update market
        self._update_market()
        self._update_order_book()

        # Check collision with support/resistance
        if self.player_y <= self.support or self.player_y >= self.resistance:
            self.game_over = True
            return False

        return True

    def _update_market(self):
        """Update support, resistance, and price."""
        # Random walk for support/resistance
        self.support += random.uniform(-0.008, 0.012)
        self.resistance += random.uniform(-0.012, 0.008)

        # Price moves within support/resistance
        self.current_price += random.uniform(-0.02, 0.02)
        self.current_price = max(self.support + 0.1, min(self.resistance - 0.1, self.current_price))

        # Keep minimum gap
        min_gap = 0.4
        if self.resistance - self.support < min_gap:
            mid = (self.resistance + self.support) / 2
            self.support = mid - min_gap / 2
            self.resistance = mid + min_gap / 2

        # Clamp
        self.support = max(0.05, min(0.35, self.support))
        self.resistance = max(0.65, min(0.95, self.resistance))

    def render(self) -> str:
        """Render the game."""
        lines = []

        # Header
        pnl_sign = "+" if self.portfolio.total_pnl >= 0 else ""
        lines.append(f"═══ EGG TRADER v2 ═══  P&L: ${pnl_sign}{self.portfolio.total_pnl:.2f}  Cash: ${self.portfolio.cash:.2f}")
        lines.append(f"Eggs traded: {self.portfolio.total_eggs_traded}  |  Trades: {len(self.portfolio.trades)}")
        lines.append("")

        # Game area
        inner_h = self.height - 4
        inner_w = self.width - 4

        # Create grid
        grid = [[' ' for _ in range(inner_w)] for _ in range(inner_h)]

        # Draw support/resistance
        support_row = int((1 - self.support) * (inner_h - 1))
        resistance_row = int((1 - self.resistance) * (inner_h - 1))

        for col in range(inner_w):
            if 0 <= resistance_row < inner_h:
                grid[resistance_row][col] = '░'
            if 0 <= support_row < inner_h:
                grid[support_row][col] = '░'

        # Draw price line
        price_row = int((1 - self.current_price) * (inner_h - 1))
        if 0 <= price_row < inner_h:
            for col in range(5, inner_w - 5):
                if grid[price_row][col] == ' ':
                    grid[price_row][col] = '═'

        # Draw egg piles on left (pickup zone)
        pile_col = 2
        for pile in self.long_piles + self.short_piles:
            pile_row = int((1 - pile.price_level) * (inner_h - 1))
            if 0 <= pile_row < inner_h:
                symbol = pile.symbol()
                for i, ch in enumerate(symbol):
                    if pile_col + i < inner_w:
                        grid[pile_row][pile_col + i] = ch

        # Draw order book on right (deposit zone)
        book_col = inner_w - 8
        for level in self.order_book:
            level_row = int((1 - level.price_level) * (inner_h - 1))
            if 0 <= level_row < inner_h and level.bid_size > 0:
                # Show bid size as squares
                symbol = "□" * level.bid_size
                for i, ch in enumerate(symbol):
                    if book_col + i < inner_w:
                        grid[level_row][book_col + i] = ch

        # Draw player
        player_col = int(self.player_x * (inner_w - 1))
        player_row = int((1 - self.player_y) * (inner_h - 1))
        player_col = max(0, min(inner_w - 1, player_col))
        player_row = max(0, min(inner_h - 1, player_row))

        if self.carrying:
            # Show carrying indicator
            if self.carrying.egg_type == EggType.LONG:
                player_char = str(self.carrying.count) + "●"
            else:
                player_char = str(self.carrying.count) + "○"
            grid[player_row][player_col] = player_char[0]
            if player_col + 1 < inner_w:
                grid[player_row][player_col + 1] = player_char[1]
        else:
            grid[player_row][player_col] = '◆'

        # Build output
        lines.append("┌" + "─" * inner_w + "┐")

        # Labels row
        pickup_label = "PICKUP"
        deposit_label = "DEPOSIT"
        mid_space = inner_w - len(pickup_label) - len(deposit_label)
        lines.append("│" + pickup_label + " " * mid_space + deposit_label + "│")

        for row in range(inner_h):
            lines.append("│" + "".join(grid[row]) + "│")

        lines.append("└" + "─" * inner_w + "┘")

        # Legend
        lines.append("")
        lines.append("LEFT: ● Long eggs (bet UP)   ○ Short eggs (bet DOWN)")
        lines.append("RIGHT: □ = buyer waiting (order book liquidity)")
        lines.append("═══ = Current price line")
        lines.append("")

        # Carrying status
        if self.carrying:
            egg_name = "LONG" if self.carrying.egg_type == EggType.LONG else "SHORT"
            lines.append(f"CARRYING: {self.carrying.count}x {egg_name} eggs from price {self.carrying.pickup_price:.2f}")

            # Preview P&L
            if self.carrying.egg_type == EggType.LONG:
                preview_pnl = (self.player_y - self.carrying.pickup_price) * self.carrying.count * self.price_per_egg * 10
            else:
                preview_pnl = (self.carrying.pickup_price - self.player_y) * self.carrying.count * self.price_per_egg * 10

            pnl_sign = "+" if preview_pnl >= 0 else ""
            lines.append(f"If deposited here: ${pnl_sign}{preview_pnl:.2f}")

            # Find liquidity at current level
            liquidity = 0
            for level in self.order_book:
                if abs(level.price_level - self.player_y) < 0.08:
                    liquidity = level.bid_size
                    break

            if liquidity == 0:
                lines.append("⚠ NO BUYERS HERE! Move to find liquidity.")
            elif liquidity < self.carrying.count:
                lines.append(f"⚠ Only {liquidity} buyers here. Need {self.carrying.count}.")
            else:
                lines.append(f"✓ {liquidity} buyers available. Press SPACE to deposit!")
        else:
            lines.append("Go LEFT to pick up eggs. More eggs = more risk/reward!")

        lines.append("")
        lines.append("[Arrows] Move  [SPACE] Pickup/Deposit  [M] Menu  [Q] Quit")

        if self.game_over:
            lines.append("")
            lines.append("*** CRASHED INTO SUPPORT/RESISTANCE! ***")
            lines.append(f"Final P&L: ${self.portfolio.total_pnl:+.2f}")
            lines.append("[R] Restart")

        return "\n".join(lines)

    def reset(self):
        """Reset the game."""
        self.player_x = 0.5
        self.player_y = 0.5
        self.carrying = None
        self.current_price = 0.5
        self.support = 0.15
        self.resistance = 0.85
        self.tick_count = 0
        self.game_over = False
        self._init_piles()
        self._init_order_book()
        # Keep portfolio for cumulative tracking


def demo():
    """Demo the enhanced egg trader."""
    print("=" * 60)
    print("EGG TRADER v2 - Position Sizing & Order Book")
    print("=" * 60)
    print()

    game = EggTraderV2()

    print("Initial state:")
    print(game.render())
    print()

    # Move to pickup and grab 3 eggs
    print("Moving to pickup zone for 3 LONG eggs...")
    for _ in range(12):
        game.input_left()
    game.player_y = 0.65  # Move to 3-egg pile
    game.input_action()

    print(game.render())
    print()

    # Move toward deposit
    print("Flying to deposit zone...")
    for _ in range(5):
        game.input_right()
        game.tick()

    print(game.render())


if __name__ == "__main__":
    demo()
