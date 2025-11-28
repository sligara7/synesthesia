"""
Market Instruments - Structural Sonification of Market Data

Key insights:
1. Price is cyclical - like a vibrating string with harmonics
2. Past becomes less relevant over time - like harmonic decay
3. Multiple market features = multiple instruments = chords
4. Trading actions map to game actions (joystick up = long position)

The SPECTRAL structure of market data maps naturally to musical harmonics:
- Recent prices = fundamental frequency (loudest)
- Older prices = overtones (quieter, geometric decay)
- Volume = amplitude/dynamics
- Volatility = timbre/texture
- Trend = mode (major/minor)
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


# ═══════════════════════════════════════════════════════════════════════════════
# TRADING MECHANICS - Player Actions as Trading Positions
# ═══════════════════════════════════════════════════════════════════════════════

class PositionType(Enum):
    FLAT = "flat"       # No position
    LONG = "long"       # Bullish - profit when price rises
    SHORT = "short"     # Bearish - profit when price falls


@dataclass
class StopOrder:
    """A stop-loss or take-profit order."""
    price: float
    order_type: str  # "stop_loss" or "take_profit"


@dataclass
class Position:
    """A trading position."""
    position_type: PositionType
    entry_price: float
    size: float = 1.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    entry_time: int = 0


@dataclass
class TradingAccount:
    """Player's trading account state."""
    balance: float = 10000.0
    initial_balance: float = 10000.0
    position: Optional[Position] = None

    # Tracking
    total_trades: int = 0
    winning_trades: int = 0
    total_commission_paid: float = 0.0
    total_pnl: float = 0.0

    # Commission structure (the overtrading penalty)
    commission_per_trade: float = 10.0  # Fixed commission
    commission_rate: float = 0.001      # 0.1% of trade value

    @property
    def equity(self) -> float:
        """Current equity (balance + unrealized PnL)."""
        return self.balance + (self.unrealized_pnl if self.position else 0)

    @property
    def unrealized_pnl(self) -> float:
        """Calculated when we have current price."""
        return 0.0  # Set by game based on current price

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def avg_pnl_per_trade(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.total_pnl / self.total_trades

    @property
    def commission_drag(self) -> float:
        """How much commissions have hurt returns."""
        if self.total_trades == 0:
            return 0.0
        return self.total_commission_paid / self.initial_balance


class TradingMechanics:
    """
    Handles trading logic - position entry, exit, stops, commissions.

    Maps game actions to trading:
    - Joystick UP (sustained) = LONG position
    - Joystick DOWN (sustained) = SHORT position
    - Joystick CENTER = FLAT (exit position)
    - Set floor = Stop Loss
    - Set ceiling = Take Profit
    """

    def __init__(self, account: TradingAccount = None):
        self.account = account or TradingAccount()
        self.current_price = 100.0
        self.price_history: List[float] = []

    def calculate_commission(self, trade_value: float) -> float:
        """Calculate commission for a trade."""
        return self.account.commission_per_trade + (trade_value * self.account.commission_rate)

    def enter_position(
        self,
        position_type: PositionType,
        price: float,
        size: float = 1.0,
        stop_loss: float = None,
        take_profit: float = None,
        timestamp: int = 0
    ) -> Tuple[bool, str]:
        """
        Enter a new position.

        Returns (success, message).
        """
        if self.account.position is not None:
            return False, "Already in position - exit first"

        if position_type == PositionType.FLAT:
            return False, "Cannot enter FLAT position"

        # Calculate trade value and commission
        trade_value = price * size
        commission = self.calculate_commission(trade_value)

        # Check if we can afford it
        if commission > self.account.balance:
            return False, f"Insufficient balance for commission (${commission:.2f})"

        # Deduct commission
        self.account.balance -= commission
        self.account.total_commission_paid += commission
        self.account.total_trades += 1

        # Create position
        self.account.position = Position(
            position_type=position_type,
            entry_price=price,
            size=size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=timestamp
        )

        direction = "LONG" if position_type == PositionType.LONG else "SHORT"
        return True, f"Entered {direction} @ ${price:.2f} (comm: ${commission:.2f})"

    def exit_position(self, price: float) -> Tuple[bool, str, float]:
        """
        Exit current position.

        Returns (success, message, pnl).
        """
        if self.account.position is None:
            return False, "No position to exit", 0.0

        pos = self.account.position

        # Calculate PnL
        if pos.position_type == PositionType.LONG:
            pnl = (price - pos.entry_price) * pos.size
        else:  # SHORT
            pnl = (pos.entry_price - price) * pos.size

        # Calculate exit commission
        trade_value = price * pos.size
        commission = self.calculate_commission(trade_value)

        # Net PnL after commission
        net_pnl = pnl - commission

        # Update account
        self.account.balance += net_pnl
        self.account.total_commission_paid += commission
        self.account.total_pnl += net_pnl

        if net_pnl > 0:
            self.account.winning_trades += 1

        # Clear position
        direction = "LONG" if pos.position_type == PositionType.LONG else "SHORT"
        self.account.position = None

        return True, f"Exited {direction} @ ${price:.2f}, PnL: ${net_pnl:.2f}", net_pnl

    def check_stops(self, high: float, low: float) -> Optional[Tuple[str, float]]:
        """
        Check if stop loss or take profit was hit.

        Returns (trigger_type, trigger_price) or None.
        """
        if self.account.position is None:
            return None

        pos = self.account.position

        if pos.position_type == PositionType.LONG:
            # Long: stop loss triggers on low, take profit on high
            if pos.stop_loss and low <= pos.stop_loss:
                return ("stop_loss", pos.stop_loss)
            if pos.take_profit and high >= pos.take_profit:
                return ("take_profit", pos.take_profit)
        else:
            # Short: stop loss triggers on high, take profit on low
            if pos.stop_loss and high >= pos.stop_loss:
                return ("stop_loss", pos.stop_loss)
            if pos.take_profit and low <= pos.take_profit:
                return ("take_profit", pos.take_profit)

        return None

    def set_stop_loss(self, price: float) -> Tuple[bool, str]:
        """Set stop loss for current position (player sets floor)."""
        if self.account.position is None:
            return False, "No position - cannot set stop"

        self.account.position.stop_loss = price
        return True, f"Stop loss set @ ${price:.2f}"

    def set_take_profit(self, price: float) -> Tuple[bool, str]:
        """Set take profit for current position (player sets ceiling)."""
        if self.account.position is None:
            return False, "No position - cannot set take profit"

        self.account.position.take_profit = price
        return True, f"Take profit set @ ${price:.2f}"

    def get_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL at current price."""
        if self.account.position is None:
            return 0.0

        pos = self.account.position
        if pos.position_type == PositionType.LONG:
            return (current_price - pos.entry_price) * pos.size
        else:
            return (pos.entry_price - current_price) * pos.size

    def get_status(self, current_price: float) -> str:
        """Get current trading status as string."""
        acc = self.account

        lines = [
            f"Balance: ${acc.balance:.2f}",
            f"Trades: {acc.total_trades} (Win rate: {acc.win_rate*100:.1f}%)",
            f"Commissions paid: ${acc.total_commission_paid:.2f}",
        ]

        if acc.position:
            pos = acc.position
            pnl = self.get_unrealized_pnl(current_price)
            direction = "LONG ▲" if pos.position_type == PositionType.LONG else "SHORT ▼"
            lines.append(f"Position: {direction} @ ${pos.entry_price:.2f}")
            lines.append(f"Unrealized PnL: ${pnl:+.2f}")
            if pos.stop_loss:
                lines.append(f"Stop Loss: ${pos.stop_loss:.2f}")
            if pos.take_profit:
                lines.append(f"Take Profit: ${pos.take_profit:.2f}")
        else:
            lines.append("Position: FLAT")

        return " | ".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# MARKET INSTRUMENTS - Harmonic Sonification
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Harmonic:
    """A single harmonic component."""
    frequency: float      # Hz
    amplitude: float      # 0-1
    phase: float = 0.0    # Radians


@dataclass
class InstrumentState:
    """Current state of a market instrument."""
    fundamental_freq: float = 220.0    # A3
    harmonics: List[Harmonic] = field(default_factory=list)
    amplitude: float = 0.5

    # Envelope
    attack: float = 0.1
    decay: float = 0.2
    sustain: float = 0.7
    release: float = 0.3

    # Character
    timbre: str = "sine"  # sine, saw, square, complex


class MarketInstrument:
    """
    Converts a market data stream into continuous audio parameters.

    Key insight: Price history is like a plucked string:
    - Recent prices = fundamental (loudest, most important)
    - Older prices = overtones (geometric decay)

    We use the SPECTRAL structure of price changes to generate harmonics.
    """

    # Base frequency mapping (price percentile to Hz)
    MIN_FREQ = 55.0    # A1
    MAX_FREQ = 880.0   # A5

    def __init__(
        self,
        name: str = "price",
        harmonic_decay: float = 0.7,    # Each older sample is 0.7x as loud
        num_harmonics: int = 8,         # How many past bars to consider
        base_amplitude: float = 0.5,
    ):
        self.name = name
        self.harmonic_decay = harmonic_decay
        self.num_harmonics = num_harmonics
        self.base_amplitude = base_amplitude

        self.price_history: List[float] = []
        self.state = InstrumentState()

    def update(self, price: float, price_min: float, price_max: float) -> InstrumentState:
        """
        Update instrument state with new price.

        Returns the new InstrumentState for audio synthesis.
        """
        self.price_history.append(price)

        # Keep limited history
        if len(self.price_history) > self.num_harmonics * 2:
            self.price_history = self.price_history[-self.num_harmonics * 2:]

        # Normalize current price to frequency
        price_range = price_max - price_min if price_max > price_min else 1
        price_pct = (price - price_min) / price_range

        fundamental = self.MIN_FREQ + (price_pct * (self.MAX_FREQ - self.MIN_FREQ))
        self.state.fundamental_freq = fundamental

        # Build harmonics from price history
        # Each past price becomes an overtone with decaying amplitude
        self.state.harmonics = []

        history = list(reversed(self.price_history[-self.num_harmonics:]))

        for i, past_price in enumerate(history):
            if i == 0:
                # Fundamental = current price
                amplitude = self.base_amplitude
            else:
                # Overtones decay geometrically
                amplitude = self.base_amplitude * (self.harmonic_decay ** i)

            # Map past price to frequency offset
            past_pct = (past_price - price_min) / price_range
            harmonic_freq = self.MIN_FREQ + (past_pct * (self.MAX_FREQ - self.MIN_FREQ))

            # Phase based on position in history
            phase = (i * math.pi / 4) % (2 * math.pi)

            self.state.harmonics.append(Harmonic(
                frequency=harmonic_freq,
                amplitude=amplitude,
                phase=phase
            ))

        return self.state

    def get_continuous_value(self, bar_progress: float) -> Tuple[float, float]:
        """
        Get interpolated frequency and amplitude at a point within a bar.

        bar_progress: 0.0 (bar start) to 1.0 (bar end)

        Returns (frequency, amplitude).
        """
        if len(self.price_history) < 2:
            return (self.state.fundamental_freq, self.base_amplitude)

        # Interpolate between previous close and current
        prev_freq = self.state.harmonics[-1].frequency if self.state.harmonics else self.MIN_FREQ
        curr_freq = self.state.fundamental_freq

        # Smooth interpolation with slight anticipation
        t = self._ease_in_out(bar_progress)
        freq = prev_freq + (curr_freq - prev_freq) * t

        # Amplitude varies during bar (builds toward close)
        amp = self.base_amplitude * (0.7 + 0.3 * t)

        return (freq, amp)

    def _ease_in_out(self, t: float) -> float:
        """Smooth easing function."""
        return t * t * (3 - 2 * t)

    def describe(self) -> str:
        """Human-readable description of current state."""
        if not self.state.harmonics:
            return f"{self.name}: No data"

        fundamental = self.state.fundamental_freq
        note = self._freq_to_note(fundamental)

        # Describe harmonic content
        n_harmonics = len(self.state.harmonics)
        total_energy = sum(h.amplitude for h in self.state.harmonics)

        return f"{self.name}: {note} ({fundamental:.1f}Hz) | {n_harmonics} harmonics | energy: {total_energy:.2f}"

    def _freq_to_note(self, freq: float) -> str:
        """Convert frequency to note name."""
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        if freq <= 0:
            return "?"

        # A4 = 440Hz
        half_steps = 12 * math.log2(freq / 440.0)
        note_idx = int(round(half_steps)) % 12
        octave = int(4 + (half_steps + 9) // 12)

        return f"{notes[note_idx]}{octave}"


class VolumeInstrument(MarketInstrument):
    """
    Instrument that sonifies volume.

    Volume maps to:
    - Amplitude/dynamics (louder = higher volume)
    - Density (more notes = higher volume)
    - Percussion intensity
    """

    def __init__(self, **kwargs):
        super().__init__(name="volume", **kwargs)
        self.state.timbre = "percussion"

    def update(self, volume: float, vol_min: float, vol_max: float) -> InstrumentState:
        """Update with volume data."""
        self.price_history.append(volume)

        # Volume affects amplitude more than frequency
        vol_range = vol_max - vol_min if vol_max > vol_min else 1
        vol_pct = (volume - vol_min) / vol_range

        self.state.amplitude = 0.3 + (vol_pct * 0.7)  # 0.3 to 1.0

        # Higher volume = faster attack (more urgent)
        self.state.attack = 0.2 - (vol_pct * 0.15)  # 0.05 to 0.2

        # Frequency based on volume (higher vol = higher pitch for urgency)
        self.state.fundamental_freq = 100 + (vol_pct * 200)  # 100-300 Hz

        return self.state


class VolatilityInstrument(MarketInstrument):
    """
    Instrument that sonifies volatility.

    Volatility maps to:
    - Timbre complexity (more volatile = more harmonics)
    - Vibrato/tremolo (instability in pitch/amplitude)
    - Tension/dissonance
    """

    def __init__(self, **kwargs):
        super().__init__(name="volatility", **kwargs)

    def update(self, volatility: float, vol_min: float, vol_max: float) -> InstrumentState:
        """Update with volatility data."""
        self.price_history.append(volatility)

        vol_range = vol_max - vol_min if vol_max > vol_min else 0.01
        vol_pct = (volatility - vol_min) / vol_range

        # High volatility = more complex timbre
        if vol_pct > 0.7:
            self.state.timbre = "complex"
        elif vol_pct > 0.3:
            self.state.timbre = "saw"
        else:
            self.state.timbre = "sine"

        # Volatility adds slight detuning to harmonics (dissonance)
        for i, harmonic in enumerate(self.state.harmonics):
            # Detune by up to 5% based on volatility
            detune = 1.0 + (vol_pct * 0.05 * math.sin(i * 1.5))
            harmonic.frequency *= detune

        return self.state


@dataclass
class ChordVoicing:
    """A chord created by multiple instruments."""
    root_note: str
    chord_type: str  # "major", "minor", "diminished", "augmented", "sus4"
    instruments: Dict[str, InstrumentState] = field(default_factory=dict)

    def describe(self) -> str:
        return f"{self.root_note}{self.chord_type}"


class MarketOrchestra:
    """
    Combines multiple market instruments into a coherent soundscape.

    Each market feature becomes an instrument:
    - Price → Bass/drone (fundamental structure)
    - Volume → Percussion/dynamics
    - Volatility → Lead/texture
    - Trend → Pad/mode

    Together they form chords and key signatures.
    """

    def __init__(self):
        self.instruments = {
            'price': MarketInstrument(name='price', harmonic_decay=0.75),
            'volume': VolumeInstrument(harmonic_decay=0.5),
            'volatility': VolatilityInstrument(harmonic_decay=0.6),
        }

        self.current_chord: Optional[ChordVoicing] = None
        self.key_signature: str = "C major"
        self.tempo: float = 120.0

    def update(
        self,
        price: float,
        volume: float,
        volatility: float,
        price_range: Tuple[float, float],
        volume_range: Tuple[float, float],
        volatility_range: Tuple[float, float],
        trend: str = "neutral"
    ) -> ChordVoicing:
        """
        Update all instruments with new market data.

        Returns the current chord voicing.
        """
        # Update each instrument
        price_state = self.instruments['price'].update(price, *price_range)
        volume_state = self.instruments['volume'].update(volume, *volume_range)
        vol_state = self.instruments['volatility'].update(volatility, *volatility_range)

        # Determine chord type from trend
        if trend == "bullish":
            chord_type = "major"
            self.key_signature = "C major"
        elif trend == "bearish":
            chord_type = "minor"
            self.key_signature = "A minor"
        elif volatility > volatility_range[1] * 0.7:
            chord_type = "diminished"  # Tension
            self.key_signature = "chromatic"
        else:
            chord_type = "sus4"  # Ambiguous
            self.key_signature = "modal"

        # Tempo from volatility
        vol_pct = (volatility - volatility_range[0]) / (volatility_range[1] - volatility_range[0])
        self.tempo = 60 + (vol_pct * 120)  # 60-180 BPM

        # Create chord voicing
        root = self.instruments['price']._freq_to_note(price_state.fundamental_freq)

        self.current_chord = ChordVoicing(
            root_note=root,
            chord_type=chord_type,
            instruments={
                'price': price_state,
                'volume': volume_state,
                'volatility': vol_state,
            }
        )

        return self.current_chord

    def get_continuous_state(self, bar_progress: float) -> Dict[str, Tuple[float, float]]:
        """
        Get interpolated state for all instruments at a point within a bar.

        Returns dict of instrument_name -> (frequency, amplitude).
        """
        return {
            name: inst.get_continuous_value(bar_progress)
            for name, inst in self.instruments.items()
        }

    def describe(self) -> str:
        """Describe current orchestral state."""
        lines = [
            f"Key: {self.key_signature} | Tempo: {self.tempo:.0f} BPM",
        ]

        if self.current_chord:
            lines.append(f"Chord: {self.current_chord.describe()}")

        for name, inst in self.instruments.items():
            lines.append(f"  {inst.describe()}")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATED DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def demo_trading_mechanics():
    """Demonstrate trading mechanics."""
    print("=" * 70)
    print("TRADING MECHANICS DEMO")
    print("=" * 70)
    print("""
    Player actions map to trading:
    - Joystick UP (hold) → Enter LONG position
    - Joystick DOWN (hold) → Enter SHORT position
    - Joystick CENTER → Exit to FLAT
    - Set floor boundary → Stop Loss
    - Set ceiling boundary → Take Profit
    """)

    trader = TradingMechanics()

    # Simulate a trading session
    prices = [100, 102, 104, 103, 101, 99, 97, 98, 100, 103, 106, 105]

    print("\nSimulating trading session:")
    print("-" * 70)

    for i, price in enumerate(prices):
        print(f"\nBar {i}: Price = ${price:.2f}")

        # Simulate player actions
        if i == 1:  # Enter long
            success, msg = trader.enter_position(
                PositionType.LONG, price,
                stop_loss=price - 3,
                take_profit=price + 5
            )
            print(f"  Action: {msg}")

        if i == 5:  # Exit on downturn
            success, msg, pnl = trader.exit_position(price)
            print(f"  Action: {msg}")

        if i == 6:  # Enter short
            success, msg = trader.enter_position(
                PositionType.SHORT, price,
                stop_loss=price + 2,
                take_profit=price - 4
            )
            print(f"  Action: {msg}")

        if i == 8:  # Exit short
            success, msg, pnl = trader.exit_position(price)
            print(f"  Action: {msg}")

        print(f"  Status: {trader.get_status(price)}")

    # Summary
    print("\n" + "=" * 70)
    print("TRADING SUMMARY")
    print("=" * 70)
    acc = trader.account
    print(f"""
    Initial Balance: ${acc.initial_balance:.2f}
    Final Balance:   ${acc.balance:.2f}
    Total PnL:       ${acc.total_pnl:+.2f}

    Total Trades:    {acc.total_trades}
    Win Rate:        {acc.win_rate*100:.1f}%

    Commission Paid: ${acc.total_commission_paid:.2f}
    Commission Drag: {acc.commission_drag*100:.2f}% of initial balance

    {"⚠️  OVERTRADING WARNING!" if acc.commission_drag > 0.01 else "✓ Commission impact acceptable"}
    """)


def demo_market_instruments():
    """Demonstrate market instruments."""
    print("\n" + "=" * 70)
    print("MARKET INSTRUMENTS DEMO")
    print("=" * 70)
    print("""
    Each market feature becomes an instrument:
    - Price → Bass drone (harmonics decay like a plucked string)
    - Volume → Percussion (dynamics, urgency)
    - Volatility → Lead synth (timbre, tension)

    Together they form the market's "chord".
    """)

    orchestra = MarketOrchestra()

    # Simulate market data
    data = [
        (100, 500, 0.02, "neutral"),
        (102, 600, 0.025, "bullish"),
        (105, 800, 0.03, "bullish"),
        (104, 1200, 0.045, "bearish"),
        (101, 1500, 0.06, "bearish"),
        (99, 2000, 0.08, "bearish"),  # Panic
        (98, 1000, 0.05, "neutral"),
        (100, 700, 0.03, "bullish"),
    ]

    price_range = (95, 110)
    vol_range = (300, 2500)
    volat_range = (0.01, 0.10)

    print("\nMarket progression:")
    print("-" * 70)

    for i, (price, volume, volatility, trend) in enumerate(data):
        chord = orchestra.update(
            price, volume, volatility,
            price_range, vol_range, volat_range,
            trend
        )

        print(f"\nBar {i}: Price=${price} Vol={volume} Volat={volatility*100:.1f}% ({trend})")
        print(orchestra.describe())

        # Show continuous interpolation
        print("  Continuous (0→0.5→1.0):")
        for t in [0.0, 0.5, 1.0]:
            continuous = orchestra.get_continuous_state(t)
            freqs = [f"{name}:{freq:.0f}Hz" for name, (freq, amp) in continuous.items()]
            print(f"    t={t}: {' | '.join(freqs)}")

    print("\n" + "=" * 70)
    print("HARMONIC DECAY VISUALIZATION")
    print("=" * 70)
    print("""
    Price history as harmonics (newer = louder):

    Current price (t=0):  ████████████████████  1.00 (fundamental)
    t-1:                  ██████████████        0.75 (1st harmonic)
    t-2:                  ██████████            0.56 (2nd harmonic)
    t-3:                  ███████               0.42 (3rd harmonic)
    t-4:                  █████                 0.32 (4th harmonic)
    t-5:                  ████                  0.24 (5th harmonic)
    t-6:                  ███                   0.18 (6th harmonic)
    t-7:                  ██                    0.13 (7th harmonic)

    Just like a guitar string: fundamental dominates, overtones fade.
    The "shape" of this decay IS the market's tonal character.
    """)


def demo():
    """Run full demo."""
    demo_trading_mechanics()
    demo_market_instruments()

    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print("""
    1. TRADING AS GAMEPLAY
       - Joystick controls map directly to position management
       - Setting boundaries = setting stops
       - Over-trading penalty = commission drain

    2. PRICE AS HARMONICS
       - Recent prices = fundamental (loudest)
       - Older prices = overtones (geometric decay)
       - Just like a plucked string!

    3. MULTIPLE INSTRUMENTS
       - Price → Bass (foundation)
       - Volume → Percussion (intensity)
       - Volatility → Lead (texture, tension)
       - Together = market's "chord"

    4. CONTINUOUS FROM DISCRETE
       - Bars are discrete, but sound is continuous
       - Interpolate within bars for smooth audio
       - Preserve uncertainty at bar boundaries

    The market SOUNDS like something - and that sound has structure.
    Learning to hear it is learning to read it.
    """)


if __name__ == "__main__":
    demo()
