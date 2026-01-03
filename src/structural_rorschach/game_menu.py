"""
Game Menu - Simple Crypto Selection for Kids

A super simple menu system for 8-year-olds to pick which
cryptocurrency cave to fly through.

MAIN MENU:
    ╔══════════════════════════════════════╗
    ║      CAVE TRADER - PICK A GAME       ║
    ╠══════════════════════════════════════╣
    ║                                      ║
    ║    [1]  CAVE FLIGHT (dodge walls)    ║
    ║                                      ║
    ║    [2]  EGG TRADER (make money!)     ║
    ║                                      ║
    ╠══════════════════════════════════════╣
    ║   Press 1 or 2 to pick game mode!    ║
    ╚══════════════════════════════════════╝

CRYPTO MENU (for Cave Flight):
    ╔══════════════════════════════════════╗
    ║      CAVE TRADER - PICK YOUR CAVE    ║
    ╠══════════════════════════════════════╣
    ║                                      ║
    ║    [1]  BITCOIN CAVE      (BTC)      ║
    ║                                      ║
    ║    [2]  ETHEREUM CAVE     (ETH)      ║
    ║                                      ║
    ║    [3]  RIPPLE CAVE       (XRP)      ║
    ║                                      ║
    ╠══════════════════════════════════════╣
    ║   Press 1, 2, or 3 to start!         ║
    ╚══════════════════════════════════════╝

"""

from dataclasses import dataclass
from typing import Optional, Callable
from enum import Enum


class CryptoChoice(Enum):
    """Available cryptocurrencies to trade."""
    BITCOIN = "BTCUSDT"
    ETHEREUM = "ETHUSDT"
    RIPPLE = "XRPUSDT"


@dataclass
class CaveOption:
    """A menu option for selecting a cave/crypto."""
    key: str           # Key to press (1, 2, 3)
    name: str          # Display name
    symbol: str        # Binance symbol
    description: str   # Fun description for kids


# The three cave options
CAVE_OPTIONS = [
    CaveOption(
        key="1",
        name="BITCOIN CAVE",
        symbol="BTCUSDT",
        description="The big orange cave! Slow but steady.",
    ),
    CaveOption(
        key="2",
        name="ETHEREUM CAVE",
        symbol="ETHUSDT",
        description="The purple crystal cave! Fast and exciting!",
    ),
    CaveOption(
        key="3",
        name="RIPPLE CAVE",
        symbol="XRPUSDT",
        description="The blue water cave! Quick and splashy!",
    ),
]


class GameMode(Enum):
    """Available game modes."""
    CAVE_FLIGHT = "cave"
    EGG_TRADER = "egg"


def render_main_menu(selected: Optional[str] = None) -> str:
    """
    Render the main game mode selection menu.

    Args:
        selected: Currently highlighted option (1, 2, or 3)

    Returns:
        ASCII art menu string
    """
    width = 44
    lines = []

    lines.append("╔" + "═" * width + "╗")

    title = "CAVE TRADER - PICK A GAME"
    pad = (width - len(title)) // 2
    lines.append("║" + " " * pad + title + " " * (width - pad - len(title)) + "║")

    lines.append("╠" + "═" * width + "╣")
    lines.append("║" + " " * width + "║")

    # Option 1: Cave Flight
    if selected == "1":
        opt1 = ">>> [1]  CAVE FLIGHT (dodge walls)    <<<"
    else:
        opt1 = "    [1]  CAVE FLIGHT (dodge walls)       "
    pad = (width - len(opt1)) // 2
    lines.append("║" + " " * pad + opt1 + " " * (width - pad - len(opt1)) + "║")

    lines.append("║" + " " * width + "║")

    # Option 2: Egg Trader (simple)
    if selected == "2":
        opt2 = ">>> [2]  EGG TRADER (simple)          <<<"
    else:
        opt2 = "    [2]  EGG TRADER (simple)             "
    pad = (width - len(opt2)) // 2
    lines.append("║" + " " * pad + opt2 + " " * (width - pad - len(opt2)) + "║")

    lines.append("║" + " " * width + "║")

    # Option 3: Trading Floor (full order book)
    if selected == "3":
        opt3 = ">>> [3]  TRADING FLOOR (order book)   <<<"
    else:
        opt3 = "    [3]  TRADING FLOOR (order book)      "
    pad = (width - len(opt3)) // 2
    lines.append("║" + " " * pad + opt3 + " " * (width - pad - len(opt3)) + "║")

    lines.append("║" + " " * width + "║")
    lines.append("╠" + "═" * width + "╣")

    instructions = "Press 1, 2, or 3 to pick game mode!"
    pad = (width - len(instructions)) // 2
    lines.append("║" + " " * pad + instructions + " " * (width - pad - len(instructions)) + "║")

    lines.append("╚" + "═" * width + "╝")

    return "\n".join(lines)


def render_menu(selected: Optional[str] = None) -> str:
    """
    Render the main menu.

    Args:
        selected: Currently highlighted option (1, 2, or 3)

    Returns:
        ASCII art menu string
    """
    width = 42

    lines = []

    # Top border
    lines.append("╔" + "═" * width + "╗")

    # Title
    title = "CAVE TRADER - PICK YOUR CAVE"
    pad = (width - len(title)) // 2
    lines.append("║" + " " * pad + title + " " * (width - pad - len(title)) + "║")

    # Separator
    lines.append("╠" + "═" * width + "╣")

    # Empty line
    lines.append("║" + " " * width + "║")

    # Options
    for opt in CAVE_OPTIONS:
        # Check if this option is selected/highlighted
        if selected == opt.key:
            marker = ">>>"
            end_marker = "<<<"
        else:
            marker = "   "
            end_marker = "   "

        # Option line
        option_text = f"{marker} [{opt.key}]  {opt.name:16} ({opt.symbol[:3]}) {end_marker}"
        pad = (width - len(option_text)) // 2
        lines.append("║" + " " * pad + option_text + " " * (width - pad - len(option_text)) + "║")

        # Empty line between options
        lines.append("║" + " " * width + "║")

    # Separator
    lines.append("╠" + "═" * width + "╣")

    # Instructions
    instructions = "Press 1, 2, or 3 to start!"
    pad = (width - len(instructions)) // 2
    lines.append("║" + " " * pad + instructions + " " * (width - pad - len(instructions)) + "║")

    # Bottom border
    lines.append("╚" + "═" * width + "╝")

    return "\n".join(lines)


def render_loading(symbol: str) -> str:
    """Render a loading screen while connecting."""
    width = 42

    # Find the option
    opt = None
    for o in CAVE_OPTIONS:
        if o.symbol == symbol:
            opt = o
            break

    cave_name = opt.name if opt else symbol

    lines = []
    lines.append("╔" + "═" * width + "╗")
    lines.append("║" + " " * width + "║")

    loading = f"Entering {cave_name}..."
    pad = (width - len(loading)) // 2
    lines.append("║" + " " * pad + loading + " " * (width - pad - len(loading)) + "║")

    lines.append("║" + " " * width + "║")

    # Simple rocket animation
    rocket = "    ◆ ~~~~>"
    pad = (width - len(rocket)) // 2
    lines.append("║" + " " * pad + rocket + " " * (width - pad - len(rocket)) + "║")

    lines.append("║" + " " * width + "║")
    lines.append("╚" + "═" * width + "╝")

    return "\n".join(lines)


def render_controls() -> str:
    """Render the controls help screen."""
    lines = []

    lines.append("╔════════════════════════════════════════╗")
    lines.append("║           HOW TO PLAY                  ║")
    lines.append("╠════════════════════════════════════════╣")
    lines.append("║                                        ║")
    lines.append("║              [UP]                      ║")
    lines.append("║               ↑                        ║")
    lines.append("║               │                        ║")
    lines.append("║    [LEFT] ←───◆───→ [RIGHT]           ║")
    lines.append("║               │                        ║")
    lines.append("║               ↓                        ║")
    lines.append("║             [DOWN]                     ║")
    lines.append("║                                        ║")
    lines.append("╠════════════════════════════════════════╣")
    lines.append("║  Use arrow keys to fly your rocket!    ║")
    lines.append("║  Don't crash into the walls!           ║")
    lines.append("║                                        ║")
    lines.append("║  Press [R] to restart after crash      ║")
    lines.append("║  Press [Q] to quit                     ║")
    lines.append("║  Press [M] for menu                    ║")
    lines.append("╚════════════════════════════════════════╝")

    return "\n".join(lines)


def render_game_over(score: int, high_score: int) -> str:
    """Render game over screen."""
    width = 42
    lines = []

    lines.append("╔" + "═" * width + "╗")
    lines.append("║" + " " * width + "║")

    title = "CRASH!"
    pad = (width - len(title)) // 2
    lines.append("║" + " " * pad + title + " " * (width - pad - len(title)) + "║")

    lines.append("║" + " " * width + "║")

    # Score
    score_line = f"Score: {score}"
    pad = (width - len(score_line)) // 2
    lines.append("║" + " " * pad + score_line + " " * (width - pad - len(score_line)) + "║")

    # High score
    if score >= high_score:
        high_line = "NEW HIGH SCORE!"
    else:
        high_line = f"High Score: {high_score}"
    pad = (width - len(high_line)) // 2
    lines.append("║" + " " * pad + high_line + " " * (width - pad - len(high_line)) + "║")

    lines.append("║" + " " * width + "║")

    restart = "[R] Restart   [M] Menu   [Q] Quit"
    pad = (width - len(restart)) // 2
    lines.append("║" + " " * pad + restart + " " * (width - pad - len(restart)) + "║")

    lines.append("║" + " " * width + "║")
    lines.append("╚" + "═" * width + "╝")

    return "\n".join(lines)


def get_symbol_from_choice(choice: str) -> Optional[str]:
    """
    Convert a menu choice (1, 2, 3) to a Binance symbol.

    Args:
        choice: The key pressed (1, 2, or 3)

    Returns:
        Binance symbol string or None if invalid choice
    """
    for opt in CAVE_OPTIONS:
        if opt.key == choice:
            return opt.symbol
    return None


def get_choice_from_symbol(symbol: str) -> Optional[CaveOption]:
    """Get the CaveOption for a symbol."""
    for opt in CAVE_OPTIONS:
        if opt.symbol == symbol:
            return opt
    return None


class GameMenu:
    """
    Interactive menu manager for the cave game.

    Handles menu state and rendering.
    """

    def __init__(self):
        self.state = "menu"  # menu, loading, playing, gameover, controls
        self.selected_symbol: Optional[str] = None
        self.highlighted: Optional[str] = None

    def select(self, choice: str) -> Optional[str]:
        """
        Handle a menu selection.

        Args:
            choice: Key pressed (1, 2, 3)

        Returns:
            Selected symbol if valid, None otherwise
        """
        symbol = get_symbol_from_choice(choice)
        if symbol:
            self.selected_symbol = symbol
            self.state = "loading"
            return symbol
        return None

    def start_game(self):
        """Transition to playing state."""
        self.state = "playing"

    def game_over(self):
        """Transition to game over state."""
        self.state = "gameover"

    def return_to_menu(self):
        """Return to main menu."""
        self.state = "menu"
        self.selected_symbol = None

    def show_controls(self):
        """Show controls screen."""
        self.state = "controls"

    def highlight(self, choice: str):
        """Highlight a menu option."""
        if choice in ("1", "2", "3"):
            self.highlighted = choice

    def render(self, score: int = 0, high_score: int = 0) -> str:
        """
        Render the current menu state.

        Args:
            score: Current game score (for game over screen)
            high_score: High score (for game over screen)

        Returns:
            ASCII art string
        """
        if self.state == "menu":
            return render_menu(self.highlighted)
        elif self.state == "loading":
            return render_loading(self.selected_symbol or "BTCUSDT")
        elif self.state == "controls":
            return render_controls()
        elif self.state == "gameover":
            return render_game_over(score, high_score)
        else:
            return ""


def demo():
    """Demo the game menu."""
    print("=" * 50)
    print("CAVE TRADER MENU DEMO")
    print("=" * 50)
    print()

    print("MAIN MENU:")
    print(render_menu())
    print()

    print("MENU WITH OPTION 2 HIGHLIGHTED:")
    print(render_menu(selected="2"))
    print()

    print("LOADING SCREEN (ETHEREUM):")
    print(render_loading("ETHUSDT"))
    print()

    print("CONTROLS SCREEN:")
    print(render_controls())
    print()

    print("GAME OVER SCREEN:")
    print(render_game_over(score=142, high_score=500))
    print()

    print("GAME OVER (NEW HIGH SCORE):")
    print(render_game_over(score=650, high_score=500))


if __name__ == "__main__":
    demo()
