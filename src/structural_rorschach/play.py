#!/usr/bin/env python3
"""
Cave Trader - Play the Game!

A simple cave flying game for kids. Use arrow keys to navigate.
The cave walls come from real cryptocurrency price data!

Run with:
    cd /home/ajs7/project/synesthesia
    PYTHONPATH=src python -m structural_rorschach.play
"""

import sys
import os
import time
import select

# Terminal handling for raw keyboard input
try:
    import termios
    import tty
    HAS_TERMIOS = True
except ImportError:
    HAS_TERMIOS = False


def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')


def get_key_nonblocking():
    """
    Get a keypress without blocking (Unix only).
    Returns None if no key pressed.
    """
    if not HAS_TERMIOS:
        return None

    # Check if input is available
    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.read(1)
    return None


def get_key_blocking():
    """Get a single keypress (blocking)."""
    if not HAS_TERMIOS:
        return input("Press key: ")[:1] if input("Press key: ") else None

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        # Handle arrow keys (escape sequences)
        if ch == '\x1b':
            ch2 = sys.stdin.read(1)
            ch3 = sys.stdin.read(1)
            if ch2 == '[':
                if ch3 == 'A':
                    return 'UP'
                elif ch3 == 'B':
                    return 'DOWN'
                elif ch3 == 'C':
                    return 'RIGHT'
                elif ch3 == 'D':
                    return 'LEFT'
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def get_arrow_key():
    """Get arrow key or regular key."""
    if not HAS_TERMIOS:
        key = input("Enter (u/d/l/r/1/2/3/q): ").lower()
        mapping = {'u': 'UP', 'd': 'DOWN', 'l': 'LEFT', 'r': 'RIGHT'}
        return mapping.get(key, key)

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)

        # Handle escape sequences (arrow keys)
        if ch == '\x1b':
            sys.stdin.read(1)  # Skip '['
            arrow = sys.stdin.read(1)
            arrows = {'A': 'UP', 'B': 'DOWN', 'C': 'RIGHT', 'D': 'LEFT'}
            return arrows.get(arrow, ch)

        # Ctrl+C
        if ch == '\x03':
            return 'QUIT'

        return ch.upper() if ch.isalpha() else ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def run_main_menu():
    """Run the main menu to select game mode. Returns 'cave', 'egg', 'floor', or None."""
    try:
        from .game_menu import render_main_menu, render_controls
    except ImportError:
        from game_menu import render_main_menu, render_controls

    while True:
        clear_screen()
        print(render_main_menu())
        print("\nPress 1 (Cave), 2 (Egg Trader), 3 (Trading Floor), H (Help), Q (Quit)")

        key = get_arrow_key()

        if key == '1':
            return 'cave'
        elif key == '2':
            return 'egg'
        elif key == '3':
            return 'floor'
        elif key in ('Q', 'QUIT'):
            return None
        elif key == 'H':
            clear_screen()
            print(render_controls())
            print("\nPress any key to go back...")
            get_arrow_key()


def run_crypto_menu():
    """Run the crypto selection menu. Returns symbol or None."""
    try:
        from .game_menu import GameMenu, render_controls
    except ImportError:
        from game_menu import GameMenu, render_controls

    menu = GameMenu()

    while True:
        clear_screen()
        print(menu.render())
        print("\nPress 1, 2, 3, H for help, M for main menu, or Q to quit")

        key = get_arrow_key()

        if key in ('1', '2', '3'):
            symbol = menu.select(key)
            clear_screen()
            print(menu.render())  # Show loading screen
            time.sleep(1)
            return symbol
        elif key in ('Q', 'QUIT'):
            return None
        elif key == 'M':
            return 'MENU'
        elif key == 'H':
            clear_screen()
            print(render_controls())
            print("\nPress any key to go back...")
            get_arrow_key()


def show_countdown(cave_name: str):
    """Show a GET READY countdown."""
    for count in [3, 2, 1]:
        clear_screen()
        print()
        print("╔══════════════════════════════════════════╗")
        print(f"║  {cave_name:^38}  ║")
        print("╠══════════════════════════════════════════╣")
        print("║                                          ║")
        print("║              GET READY!                  ║")
        print("║                                          ║")
        print(f"║                  {count}                      ║")
        print("║                                          ║")
        print("║        Use ARROW KEYS to move            ║")
        print("║                                          ║")
        print("╚══════════════════════════════════════════╝")
        time.sleep(1)

    clear_screen()
    print()
    print("╔══════════════════════════════════════════╗")
    print(f"║  {cave_name:^38}  ║")
    print("╠══════════════════════════════════════════╣")
    print("║                                          ║")
    print("║                 GO!!!                    ║")
    print("║                                          ║")
    print("║                  ◆                       ║")
    print("║                                          ║")
    print("╚══════════════════════════════════════════╝")
    time.sleep(0.5)


def run_game_offline(symbol: str):
    """
    Run the game with synthetic/offline data.
    No internet connection required!
    """
    try:
        from .simple_cave import SimpleCaveGame, HistoryTracker, GameState
        from .game_menu import render_game_over, get_choice_from_symbol
        from .snake_trail import SnakeTrail
    except ImportError:
        from simple_cave import SimpleCaveGame, HistoryTracker, GameState
        from game_menu import render_game_over, get_choice_from_symbol
        from snake_trail import SnakeTrail

    opt = get_choice_from_symbol(symbol)
    cave_name = opt.name if opt else symbol

    # Show countdown before starting
    show_countdown(cave_name)

    game = SimpleCaveGame(cave_width=40)
    tracker = HistoryTracker(max_history=20)
    snake = SnakeTrail(max_length=30)  # Snake trail for movement history

    # Game loop timing
    last_update = time.time()
    update_interval = 0.5  # Slower: 2 updates per second (like 1-min candles feel)

    # For non-blocking input
    if HAS_TERMIOS:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

    try:
        while True:
            current_time = time.time()

            # Check for input (non-blocking)
            if HAS_TERMIOS and select.select([sys.stdin], [], [], 0.0)[0]:
                ch = sys.stdin.read(1)

                # Handle escape sequences
                if ch == '\x1b':
                    sys.stdin.read(1)  # Skip '['
                    arrow = sys.stdin.read(1)
                    if arrow == 'A':
                        game.input_up()
                        snake.move_up()
                    elif arrow == 'B':
                        game.input_down()
                        snake.move_down()
                    elif arrow == 'C':
                        game.input_right()
                        snake.move_right()
                    elif arrow == 'D':
                        game.input_left()
                        snake.move_left()
                elif ch.upper() == 'Q' or ch == '\x03':
                    break
                elif ch.upper() == 'R' and game.game.state == GameState.CRASHED:
                    # Restore terminal, show countdown, then re-enter raw mode
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    show_countdown(cave_name)
                    tty.setcbreak(fd)
                    game.reset()
                    tracker = HistoryTracker(max_history=20)
                    snake = SnakeTrail(max_length=30)
                elif ch.upper() == 'M':
                    return 'MENU'

            # Update game at fixed interval
            if current_time - last_update >= update_interval:
                last_update = current_time

                if game.game.state == GameState.PLAYING:
                    # Generate random cave (offline mode)
                    game.generate_random_cave()
                    game.tick()

                    # Record history (use center slice)
                    center_idx = len(game.game.cave) // 2
                    if center_idx < len(game.game.cave):
                        tracker.record(
                            game.game.cave[center_idx],
                            game.game.rocket.x,
                            game.game.rocket.y
                        )

                # Render all views
                clear_screen()
                print(f"═══ {cave_name} ═══  (Offline Practice Mode)")
                print()

                # 1. MAIN CAVE VIEW (forward flying view)
                print("┌─ CAVE VIEW ─────────────────────────────┐")
                print(game.render())
                print()

                # 2. HISTORY PLOTS (UP/DOWN and LEFT/RIGHT) side by side
                v_lines = tracker.render_vertical(height=8, width=20).split('\n')
                h_lines = tracker.render_horizontal(height=8, width=20).split('\n')

                print("┌─ UP/DOWN HISTORY ─┐  ┌─ LEFT/RIGHT HISTORY ─┐")
                max_lines = max(len(v_lines), len(h_lines))
                for i in range(max_lines):
                    v_line = v_lines[i] if i < len(v_lines) else " " * 20
                    h_line = h_lines[i] if i < len(h_lines) else " " * 20
                    # Pad to consistent width
                    v_line = f"{v_line:<20}"
                    h_line = f"{h_line:<22}"
                    print(f"{v_line}  {h_line}")
                print()

                # 3. SNAKE TRAIL (movement pattern)
                print("┌─ SNAKE TRAIL (your moves) ─┐")
                print(snake.render(width=20, height=9))

                print()
                print("[Arrow Keys] Move  [R] Restart  [M] Menu  [Q] Quit")

                if game.game.state == GameState.CRASHED:
                    print()
                    print(render_game_over(game.game.score, game.game.high_score))

            time.sleep(0.01)  # Small sleep to prevent CPU spinning

    finally:
        if HAS_TERMIOS:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return None


def run_game_live(symbol: str):
    """
    Run the game with live Binance data.
    Requires internet connection and python-binance package.
    """
    try:
        from .binance_live import LiveCaveGame
        from .simple_cave import GameState
        from .game_menu import render_game_over, get_choice_from_symbol
    except ImportError:
        from binance_live import LiveCaveGame
        from simple_cave import GameState
        from game_menu import render_game_over, get_choice_from_symbol

    opt = get_choice_from_symbol(symbol)
    cave_name = opt.name if opt else symbol

    print(f"Connecting to {cave_name}...")

    try:
        live_game = LiveCaveGame(symbol=symbol, testnet=False)
        live_game.start()
    except Exception as e:
        print(f"Could not connect to live data: {e}")
        print("Falling back to offline mode...")
        time.sleep(2)
        return run_game_offline(symbol)

    # Show countdown
    show_countdown(cave_name)

    # Game loop
    last_update = time.time()
    update_interval = 0.15  # Slower for kids

    if HAS_TERMIOS:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

    try:
        while True:
            current_time = time.time()

            # Check for input
            if HAS_TERMIOS and select.select([sys.stdin], [], [], 0.0)[0]:
                ch = sys.stdin.read(1)

                if ch == '\x1b':
                    sys.stdin.read(1)
                    arrow = sys.stdin.read(1)
                    if arrow == 'A':
                        live_game.game.input_up()
                    elif arrow == 'B':
                        live_game.game.input_down()
                    elif arrow == 'C':
                        live_game.game.input_right()
                    elif arrow == 'D':
                        live_game.game.input_left()
                elif ch.upper() == 'Q' or ch == '\x03':
                    break
                elif ch.upper() == 'R' and live_game.game.game.state == GameState.CRASHED:
                    live_game.game.reset()
                elif ch.upper() == 'M':
                    live_game.stop()
                    return 'MENU'

            # Update
            if current_time - last_update >= update_interval:
                last_update = current_time

                if live_game.game.game.state == GameState.PLAYING:
                    live_game.tick()

                # Render
                clear_screen()
                print(f"═══ {cave_name} ═══")
                print(f"(LIVE: {symbol})")
                print()
                print(live_game.render())
                print()
                print("[Arrow Keys] Move  [R] Restart  [M] Menu  [Q] Quit")

                if live_game.game.game.state == GameState.CRASHED:
                    print()
                    print(render_game_over(
                        live_game.game.game.score,
                        live_game.game.game.high_score
                    ))

            time.sleep(0.01)

    finally:
        if HAS_TERMIOS:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        live_game.stop()

    return None


def run_trading_floor():
    """Run the full Trading Floor game with order book."""
    try:
        from .egg_trading_floor import EggTradingFloor
    except ImportError:
        from egg_trading_floor import EggTradingFloor

    # Show countdown
    show_countdown("TRADING FLOOR")

    game = EggTradingFloor(base_price=100, num_levels=10)

    # Game loop timing
    last_update = time.time()
    update_interval = 0.8  # Slower for order book game

    # For non-blocking input
    if HAS_TERMIOS:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

    try:
        while True:
            current_time = time.time()

            # Check for input (non-blocking)
            if HAS_TERMIOS and select.select([sys.stdin], [], [], 0.0)[0]:
                ch = sys.stdin.read(1)

                # Handle escape sequences (arrow keys)
                if ch == '\x1b':
                    sys.stdin.read(1)  # Skip '['
                    arrow = sys.stdin.read(1)
                    if arrow == 'A':
                        game.input_up()
                    elif arrow == 'B':
                        game.input_down()
                    elif arrow == 'C':
                        game.input_right()
                    elif arrow == 'D':
                        game.input_left()
                elif ch in '12345':  # Pick up eggs
                    game.input_pickup(int(ch))
                elif ch.upper() == 'D':  # Drop/limit order
                    game.input_drop()
                elif ch.upper() == 'M':  # Market sell
                    if game.carrying:
                        game.input_market_sell()
                    else:
                        return 'MENU'
                elif ch.upper() == 'Q' or ch == '\x03':
                    break
                elif ch.upper() == 'R' and game.game_over:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    show_countdown("TRADING FLOOR")
                    tty.setcbreak(fd)
                    game.reset()

            # Update game at fixed interval
            if current_time - last_update >= update_interval:
                last_update = current_time
                game.tick()

                # Render
                clear_screen()
                print(game.render())

            time.sleep(0.01)

    finally:
        if HAS_TERMIOS:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return None


def render_level_menu(selected: str = None) -> str:
    """Render the level selection menu for Egg Trader."""
    width = 44
    lines = []

    lines.append("╔" + "═" * width + "╗")

    title = "EGG TRADER - SELECT LEVEL"
    pad = (width - len(title)) // 2
    lines.append("║" + " " * pad + title + " " * (width - pad - len(title)) + "║")

    lines.append("╠" + "═" * width + "╣")
    lines.append("║" + " " * width + "║")

    # Level options
    levels = [
        ("1", "LEVEL 1: Green Eggs Only", "Want price UP! ↑"),
        ("2", "LEVEL 1.5: Red Eggs Only", "Want price DOWN! ↓"),
        ("3", "LEVEL 2: Both Colors", "Green=↑ Red=↓"),
        ("4", "LEVEL 3: Position Reversal", "Close + flip in one move"),
    ]

    for key, name, desc in levels:
        if selected == key:
            opt = f">>> [{key}] {name:<26} <<<"
        else:
            opt = f"    [{key}] {name:<26}    "
        pad = (width - len(opt)) // 2
        lines.append("║" + " " * pad + opt + " " * (width - pad - len(opt)) + "║")

        # Description line
        desc_line = f"        {desc}"
        pad = (width - len(desc_line)) // 2
        lines.append("║" + " " * pad + desc_line + " " * (width - pad - len(desc_line)) + "║")

        lines.append("║" + " " * width + "║")

    lines.append("╠" + "═" * width + "╣")

    instructions = "Press 1, 2, 3, or 4 to start!"
    pad = (width - len(instructions)) // 2
    lines.append("║" + " " * pad + instructions + " " * (width - pad - len(instructions)) + "║")

    lines.append("╚" + "═" * width + "╝")

    return "\n".join(lines)


def run_level_menu():
    """Run the level selection menu. Returns level key or None."""
    while True:
        clear_screen()
        print(render_level_menu())
        print("\nPress 1, 2, 3, 4 to select level, M for main menu, Q to quit")

        key = get_arrow_key()

        if key in ('1', '2', '3', '4'):
            return key  # Return as string to handle mapping
        elif key in ('Q', 'QUIT'):
            return None
        elif key == 'M':
            return 'MENU'


def run_egg_trader():
    """Run the Egg Trader game with level selection."""
    try:
        from .egg_trader_levels import EggTraderLevels, DifficultyLevel
    except ImportError:
        from egg_trader_levels import EggTraderLevels, DifficultyLevel

    # Select level
    level_key = run_level_menu()
    if level_key is None:
        return None
    if level_key == 'MENU':
        return 'MENU'

    # Map menu key to difficulty level
    level_map = {
        '1': (DifficultyLevel.LEVEL_1, "LEVEL 1"),
        '2': (DifficultyLevel.LEVEL_1B, "LEVEL 1.5"),
        '3': (DifficultyLevel.LEVEL_2, "LEVEL 2"),
        '4': (DifficultyLevel.LEVEL_3, "LEVEL 3"),
    }
    level, level_name = level_map.get(level_key, (DifficultyLevel.LEVEL_1, "LEVEL 1"))
    level_name = f"EGG TRADER {level_name}"

    # Show countdown
    show_countdown(level_name)

    game = EggTraderLevels(level=level, num_levels=10, base_price=100.0)

    # Game loop timing
    last_update = time.time()
    update_interval = 0.8  # Slower for order book style

    # For non-blocking input
    if HAS_TERMIOS:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        tty.setcbreak(fd)

    try:
        while True:
            current_time = time.time()

            # Check for input (non-blocking)
            if HAS_TERMIOS and select.select([sys.stdin], [], [], 0.0)[0]:
                ch = sys.stdin.read(1)

                # Handle escape sequences (arrow keys)
                if ch == '\x1b':
                    sys.stdin.read(1)  # Skip '['
                    arrow = sys.stdin.read(1)
                    if arrow == 'A':
                        game.input_up()
                    elif arrow == 'B':
                        game.input_down()
                    elif arrow == 'C':
                        game.input_right()
                    elif arrow == 'D':
                        game.input_left()
                elif ch == '\t':  # TAB = cycle zones
                    game.input_cycle_zone()
                elif ch in '12345':  # Pick up eggs
                    game.input_pickup(int(ch))
                elif ch.upper() == 'D':  # Deposit
                    game.input_deposit()
                elif ch.upper() == 'R':  # Reverse (Level 3+)
                    if game.carrying:
                        # Try to reverse with same quantity
                        game.input_reverse(game.carrying.quantity)
                    else:
                        # Reset game
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                        show_countdown(level_name)
                        tty.setcbreak(fd)
                        game.reset()
                elif ch == ' ':  # Space = advance time
                    game.tick_update()
                elif ch.upper() == 'Q' or ch == '\x03':
                    break
                elif ch.upper() == 'M':
                    return 'MENU'

            # Update game at fixed interval
            if current_time - last_update >= update_interval:
                last_update = current_time
                game.tick_update()

                # Render
                clear_screen()
                print(game.render())

            time.sleep(0.01)  # Small sleep to prevent CPU spinning

    finally:
        if HAS_TERMIOS:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    return None


def main():
    """Main entry point."""
    print("╔════════════════════════════════════════╗")
    print("║         CAVE TRADER                    ║")
    print("║   Fly through the crypto caves!        ║")
    print("║   Or collect eggs for profit!          ║")
    print("╚════════════════════════════════════════╝")
    print()

    # Check if we can do live data
    try:
        from binance.client import Client
        has_binance = True
        print("[✓] Live data available (python-binance installed)")
    except ImportError:
        has_binance = False
        print("[!] Live data not available (pip install python-binance)")
        print("    Running in offline practice mode")

    print()
    time.sleep(1)

    while True:
        # First, select game mode
        game_mode = run_main_menu()

        if game_mode is None:
            print("\nThanks for playing!")
            break

        if game_mode == 'egg':
            # Run Egg Trader (simple version)
            result = run_egg_trader()
            if result != 'MENU':
                break
        elif game_mode == 'floor':
            # Run Trading Floor (full order book)
            result = run_trading_floor()
            if result != 'MENU':
                break
        else:
            # Run Cave Flight - select crypto first
            symbol = run_crypto_menu()

            if symbol is None:
                print("\nThanks for playing!")
                break
            elif symbol == 'MENU':
                continue  # Go back to main menu

            # Run cave game
            if has_binance:
                result = run_game_live(symbol)
            else:
                result = run_game_offline(symbol)

            if result != 'MENU':
                break

    clear_screen()
    print("Thanks for playing Cave Trader!")
    print()


if __name__ == "__main__":
    main()
