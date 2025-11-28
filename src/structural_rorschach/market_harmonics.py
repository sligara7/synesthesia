"""
Market Harmonics - True Cyclical Pattern Detection and Sonification

The key insight: Markets oscillate at natural frequencies.
- Every ~7 bars might bounce off support
- Every ~20 bars might complete a swing
- Every ~60 bars might have a larger cycle

These ARE the market's harmonics - not "older = quieter" but
"the market vibrates at these frequencies".

We use spectral analysis (FFT, autocorrelation) to DETECT these
cycles, then map them directly to musical harmonics.

The market's spectrum IS the instrument's spectrum.
"""

import math
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import deque


@dataclass
class DetectedCycle:
    """A detected cyclical pattern in market data."""
    period: float           # Period in bars (e.g., 7 bars)
    frequency: float        # 1/period (cycles per bar)
    amplitude: float        # Strength of this cycle (0-1)
    phase: float           # Current phase in the cycle (0 to 2π)
    cycle_type: str        # "price", "support_bounce", "resistance_bounce", "volume", etc.
    confidence: float      # How confident we are in this detection (0-1)

    @property
    def wavelength_bars(self) -> int:
        """Period rounded to nearest bar."""
        return max(1, round(self.period))


@dataclass
class MarketSpectrum:
    """
    The spectral content of market data - its natural frequencies.

    This IS the market's harmonic series:
    - Dominant cycle = fundamental
    - Secondary cycles = overtones
    - Amplitude of each = loudness
    """
    cycles: List[DetectedCycle] = field(default_factory=list)

    # Derived properties
    fundamental_period: float = 0.0
    spectral_centroid: float = 0.0  # "Center of mass" of spectrum
    spectral_spread: float = 0.0    # How spread out the frequencies are

    def get_dominant_cycles(self, n: int = 5) -> List[DetectedCycle]:
        """Get the N strongest cycles."""
        return sorted(self.cycles, key=lambda c: c.amplitude, reverse=True)[:n]

    def get_harmonic_ratios(self) -> List[float]:
        """
        Get frequency ratios relative to fundamental.

        In music, harmonics are integer multiples (1, 2, 3, 4...).
        In markets, we find the ACTUAL ratios.
        """
        if not self.cycles or self.fundamental_period == 0:
            return []

        fundamental_freq = 1.0 / self.fundamental_period
        return [c.frequency / fundamental_freq for c in self.cycles]


class CycleDetector:
    """
    Detects cyclical patterns in market data.

    Methods:
    1. Autocorrelation - finds repeating patterns
    2. FFT - finds dominant frequencies
    3. Support/Resistance bounces - finds price level cycles
    4. Peak/trough analysis - finds swing cycles
    """

    def __init__(
        self,
        min_period: int = 3,      # Minimum cycle length in bars
        max_period: int = 100,    # Maximum cycle length
        min_confidence: float = 0.3,  # Minimum confidence to report
    ):
        self.min_period = min_period
        self.max_period = max_period
        self.min_confidence = min_confidence

    def analyze(self, prices: List[float], volumes: List[float] = None) -> MarketSpectrum:
        """
        Full spectral analysis of market data.

        Returns MarketSpectrum containing all detected cycles.
        """
        if len(prices) < self.min_period * 2:
            return MarketSpectrum()

        cycles = []

        # 1. Autocorrelation analysis (finds repeating patterns)
        autocorr_cycles = self._autocorrelation_analysis(prices)
        cycles.extend(autocorr_cycles)

        # 2. FFT analysis (finds dominant frequencies)
        fft_cycles = self._fft_analysis(prices)
        cycles.extend(fft_cycles)

        # 3. Peak/trough swing analysis
        swing_cycles = self._swing_analysis(prices)
        cycles.extend(swing_cycles)

        # 4. Support/resistance bounce detection
        sr_cycles = self._support_resistance_analysis(prices)
        cycles.extend(sr_cycles)

        # 5. Volume cycles (if provided)
        if volumes and len(volumes) == len(prices):
            vol_cycles = self._autocorrelation_analysis(volumes)
            for c in vol_cycles:
                c.cycle_type = "volume"
            cycles.extend(vol_cycles)

        # Filter by confidence and merge similar cycles
        cycles = [c for c in cycles if c.confidence >= self.min_confidence]
        cycles = self._merge_similar_cycles(cycles)

        # Build spectrum
        spectrum = MarketSpectrum(cycles=cycles)

        if cycles:
            # Fundamental = strongest cycle
            dominant = max(cycles, key=lambda c: c.amplitude * c.confidence)
            spectrum.fundamental_period = dominant.period

            # Spectral centroid (weighted average frequency)
            total_weight = sum(c.amplitude for c in cycles)
            if total_weight > 0:
                spectrum.spectral_centroid = sum(
                    c.frequency * c.amplitude for c in cycles
                ) / total_weight

        return spectrum

    def _autocorrelation_analysis(self, data: List[float]) -> List[DetectedCycle]:
        """
        Find cycles using autocorrelation.

        Autocorrelation measures how similar a signal is to a
        time-shifted version of itself. Peaks indicate periodicity.
        """
        n = len(data)
        if n < self.min_period * 2:
            return []

        # Normalize data
        mean = sum(data) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in data) / n)
        if std == 0:
            return []
        normalized = [(x - mean) / std for x in data]

        # Calculate autocorrelation for different lags
        cycles = []
        prev_corr = 1.0

        for lag in range(self.min_period, min(self.max_period, n // 2)):
            # Autocorrelation at this lag
            corr = sum(
                normalized[i] * normalized[i + lag]
                for i in range(n - lag)
            ) / (n - lag)

            # Look for peaks in autocorrelation (local maxima)
            if lag > self.min_period:
                # Check if previous was a peak
                if prev_corr > corr and prev_corr > 0.2:
                    # Found a peak at lag-1
                    period = lag - 1
                    amplitude = prev_corr

                    # Calculate current phase
                    phase = (n % period) / period * 2 * math.pi

                    cycles.append(DetectedCycle(
                        period=period,
                        frequency=1.0 / period,
                        amplitude=amplitude,
                        phase=phase,
                        cycle_type="autocorr",
                        confidence=min(1.0, amplitude * 1.5)
                    ))

            prev_corr = corr

        return cycles

    def _fft_analysis(self, data: List[float]) -> List[DetectedCycle]:
        """
        Find cycles using FFT (Fast Fourier Transform).

        FFT decomposes the signal into its frequency components.
        """
        n = len(data)
        if n < self.min_period * 4:
            return []

        # Simple DFT (would use numpy.fft in production)
        # For demo, we'll use a simplified approach
        cycles = []

        # Test specific frequencies
        for period in range(self.min_period, min(self.max_period, n // 2)):
            freq = 1.0 / period

            # Calculate power at this frequency (simplified DFT)
            real_sum = 0.0
            imag_sum = 0.0

            for i, val in enumerate(data):
                angle = 2 * math.pi * freq * i
                real_sum += val * math.cos(angle)
                imag_sum += val * math.sin(angle)

            power = math.sqrt(real_sum**2 + imag_sum**2) / n
            phase = math.atan2(imag_sum, real_sum)

            # Normalize power relative to data range
            data_range = max(data) - min(data)
            if data_range > 0:
                normalized_power = power / data_range

                if normalized_power > 0.1:  # Threshold
                    cycles.append(DetectedCycle(
                        period=period,
                        frequency=freq,
                        amplitude=normalized_power,
                        phase=phase,
                        cycle_type="fft",
                        confidence=min(1.0, normalized_power * 2)
                    ))

        # Keep only the strongest at each approximate period
        return self._filter_harmonics(cycles)

    def _swing_analysis(self, prices: List[float]) -> List[DetectedCycle]:
        """
        Find cycles by analyzing peak-to-peak and trough-to-trough distances.

        This captures the actual swing trading rhythm.
        """
        if len(prices) < self.min_period * 2:
            return []

        # Find peaks and troughs
        peaks = []
        troughs = []

        for i in range(2, len(prices) - 2):
            # Local maximum
            if (prices[i] > prices[i-1] and prices[i] > prices[i-2] and
                prices[i] > prices[i+1] and prices[i] > prices[i+2]):
                peaks.append(i)
            # Local minimum
            if (prices[i] < prices[i-1] and prices[i] < prices[i-2] and
                prices[i] < prices[i+1] and prices[i] < prices[i+2]):
                troughs.append(i)

        cycles = []

        # Analyze peak-to-peak distances
        if len(peaks) >= 2:
            distances = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
            if distances:
                avg_period = sum(distances) / len(distances)
                std_period = math.sqrt(sum((d - avg_period)**2 for d in distances) / len(distances))

                # Confidence based on consistency
                if avg_period > 0:
                    consistency = 1.0 - min(1.0, std_period / avg_period)

                    if consistency > 0.3:
                        cycles.append(DetectedCycle(
                            period=avg_period,
                            frequency=1.0 / avg_period,
                            amplitude=0.8,  # High importance for swing cycles
                            phase=(len(prices) - peaks[-1]) / avg_period * 2 * math.pi if peaks else 0,
                            cycle_type="peak_swing",
                            confidence=consistency
                        ))

        # Analyze trough-to-trough distances
        if len(troughs) >= 2:
            distances = [troughs[i+1] - troughs[i] for i in range(len(troughs)-1)]
            if distances:
                avg_period = sum(distances) / len(distances)
                std_period = math.sqrt(sum((d - avg_period)**2 for d in distances) / len(distances))

                if avg_period > 0:
                    consistency = 1.0 - min(1.0, std_period / avg_period)

                    if consistency > 0.3:
                        cycles.append(DetectedCycle(
                            period=avg_period,
                            frequency=1.0 / avg_period,
                            amplitude=0.8,
                            phase=(len(prices) - troughs[-1]) / avg_period * 2 * math.pi if troughs else 0,
                            cycle_type="trough_swing",
                            confidence=consistency
                        ))

        return cycles

    def _support_resistance_analysis(self, prices: List[float]) -> List[DetectedCycle]:
        """
        Find cycles in support/resistance bounces.

        "Every 7th bar bounces off support" - this is a cycle!
        """
        if len(prices) < self.min_period * 3:
            return []

        # Find support levels (recent lows)
        # Find resistance levels (recent highs)
        window = min(20, len(prices) // 4)

        # Simple support/resistance: rolling min/max
        support_tests = []  # Bars where price touched support
        resistance_tests = []  # Bars where price touched resistance

        for i in range(window, len(prices)):
            local_min = min(prices[i-window:i])
            local_max = max(prices[i-window:i])
            local_range = local_max - local_min

            if local_range > 0:
                # Test if current bar touches support (within 5% of local min)
                if (prices[i] - local_min) / local_range < 0.05:
                    support_tests.append(i)

                # Test if current bar touches resistance
                if (local_max - prices[i]) / local_range < 0.05:
                    resistance_tests.append(i)

        cycles = []

        # Analyze support bounce periodicity
        if len(support_tests) >= 3:
            distances = [support_tests[i+1] - support_tests[i]
                        for i in range(len(support_tests)-1)]
            if distances:
                avg_period = sum(distances) / len(distances)
                if avg_period >= self.min_period:
                    std_period = math.sqrt(sum((d - avg_period)**2 for d in distances) / len(distances))
                    consistency = 1.0 - min(1.0, std_period / avg_period) if avg_period > 0 else 0

                    if consistency > 0.2:
                        cycles.append(DetectedCycle(
                            period=avg_period,
                            frequency=1.0 / avg_period,
                            amplitude=0.9,  # Support bounces are significant
                            phase=(len(prices) - support_tests[-1]) / avg_period * 2 * math.pi,
                            cycle_type="support_bounce",
                            confidence=consistency
                        ))

        # Analyze resistance bounce periodicity
        if len(resistance_tests) >= 3:
            distances = [resistance_tests[i+1] - resistance_tests[i]
                        for i in range(len(resistance_tests)-1)]
            if distances:
                avg_period = sum(distances) / len(distances)
                if avg_period >= self.min_period:
                    std_period = math.sqrt(sum((d - avg_period)**2 for d in distances) / len(distances))
                    consistency = 1.0 - min(1.0, std_period / avg_period) if avg_period > 0 else 0

                    if consistency > 0.2:
                        cycles.append(DetectedCycle(
                            period=avg_period,
                            frequency=1.0 / avg_period,
                            amplitude=0.9,
                            phase=(len(prices) - resistance_tests[-1]) / avg_period * 2 * math.pi,
                            cycle_type="resistance_bounce",
                            confidence=consistency
                        ))

        return cycles

    def _merge_similar_cycles(self, cycles: List[DetectedCycle], tolerance: float = 0.15) -> List[DetectedCycle]:
        """Merge cycles with similar periods."""
        if not cycles:
            return []

        # Sort by period
        sorted_cycles = sorted(cycles, key=lambda c: c.period)
        merged = []

        current_group = [sorted_cycles[0]]

        for cycle in sorted_cycles[1:]:
            # Check if this cycle is similar to current group
            group_avg_period = sum(c.period for c in current_group) / len(current_group)

            if abs(cycle.period - group_avg_period) / group_avg_period < tolerance:
                current_group.append(cycle)
            else:
                # Merge current group and start new one
                merged.append(self._merge_cycle_group(current_group))
                current_group = [cycle]

        # Don't forget last group
        merged.append(self._merge_cycle_group(current_group))

        return merged

    def _merge_cycle_group(self, group: List[DetectedCycle]) -> DetectedCycle:
        """Merge a group of similar cycles into one."""
        if len(group) == 1:
            return group[0]

        # Weighted average by amplitude
        total_weight = sum(c.amplitude for c in group)

        avg_period = sum(c.period * c.amplitude for c in group) / total_weight
        avg_phase = sum(c.phase * c.amplitude for c in group) / total_weight
        max_amplitude = max(c.amplitude for c in group)
        max_confidence = max(c.confidence for c in group)

        # Combine cycle types
        types = set(c.cycle_type for c in group)
        combined_type = "+".join(sorted(types))

        return DetectedCycle(
            period=avg_period,
            frequency=1.0 / avg_period,
            amplitude=max_amplitude,
            phase=avg_phase,
            cycle_type=combined_type,
            confidence=max_confidence
        )

    def _filter_harmonics(self, cycles: List[DetectedCycle]) -> List[DetectedCycle]:
        """Keep only the strongest cycle at each approximate period."""
        if not cycles:
            return []

        # Group by approximate period (within 10%)
        result = []
        sorted_cycles = sorted(cycles, key=lambda c: c.amplitude, reverse=True)

        for cycle in sorted_cycles:
            # Check if we already have a cycle at this period
            dominated = False
            for existing in result:
                if abs(cycle.period - existing.period) / existing.period < 0.1:
                    dominated = True
                    break

            if not dominated:
                result.append(cycle)

        return result


class SpectralSynthesizer:
    """
    Synthesizes audio parameters from market spectrum.

    The market's detected cycles become actual musical harmonics.
    """

    # Map cycle periods to frequency multipliers
    # Longer cycles = lower frequencies (bass)
    # Shorter cycles = higher frequencies (treble)
    BASE_FREQ = 110.0  # A2

    def __init__(self, base_freq: float = 110.0):
        self.base_freq = base_freq

    def synthesize(self, spectrum: MarketSpectrum) -> Dict[str, any]:
        """
        Convert market spectrum to audio synthesis parameters.

        Returns parameters suitable for an audio synthesizer:
        - harmonics: list of (frequency, amplitude, phase)
        - fundamental: the root frequency
        - timbre: description of the overall sound
        """
        if not spectrum.cycles:
            return {
                'harmonics': [(self.base_freq, 0.5, 0.0)],
                'fundamental': self.base_freq,
                'timbre': 'silent'
            }

        # Get dominant cycles
        dominant = spectrum.get_dominant_cycles(8)

        if not dominant:
            return {
                'harmonics': [(self.base_freq, 0.5, 0.0)],
                'fundamental': self.base_freq,
                'timbre': 'silent'
            }

        # The longest significant cycle = fundamental (bass)
        longest = max(dominant, key=lambda c: c.period)

        # Map period to musical frequency
        # Longer period = lower frequency
        # We'll map so that a 50-bar cycle = base_freq
        # and scale inversely with period

        harmonics = []

        for cycle in dominant:
            # Frequency: shorter cycles = higher freq
            # If longest cycle is fundamental, shorter cycles are overtones
            period_ratio = longest.period / cycle.period
            freq = self.base_freq * period_ratio

            # Amplitude from cycle strength
            amp = cycle.amplitude * cycle.confidence

            # Phase from cycle phase
            phase = cycle.phase

            harmonics.append((freq, amp, phase))

        # Sort by frequency
        harmonics.sort(key=lambda h: h[0])

        # Determine timbre from spectral shape
        if len(harmonics) == 1:
            timbre = "pure_sine"
        elif len(harmonics) <= 3:
            timbre = "simple"
        elif spectrum.spectral_spread > 0.5:
            timbre = "complex_wide"
        else:
            timbre = "complex_focused"

        # Check for special patterns
        has_support_cycle = any('support' in c.cycle_type for c in dominant)
        has_resistance_cycle = any('resistance' in c.cycle_type for c in dominant)

        if has_support_cycle and has_resistance_cycle:
            timbre += "_bounded"  # Price oscillating in range
        elif has_support_cycle:
            timbre += "_supported"  # Strong floor
        elif has_resistance_cycle:
            timbre += "_capped"  # Strong ceiling

        return {
            'harmonics': harmonics,
            'fundamental': self.base_freq,
            'timbre': timbre,
            'num_cycles': len(dominant),
            'dominant_period': longest.period,
            'cycle_types': [c.cycle_type for c in dominant]
        }

    def describe_sound(self, spectrum: MarketSpectrum) -> str:
        """Human-readable description of what this market sounds like."""
        params = self.synthesize(spectrum)

        lines = []
        lines.append(f"Market Sound Profile:")
        lines.append(f"  Timbre: {params['timbre']}")
        lines.append(f"  Fundamental: {params['fundamental']:.1f} Hz (from {params['dominant_period']:.1f}-bar cycle)")
        lines.append(f"  Harmonics: {params['num_cycles']}")

        lines.append(f"\n  Frequency Stack:")
        for i, (freq, amp, phase) in enumerate(params['harmonics']):
            bar = '█' * int(amp * 20)
            lines.append(f"    {freq:7.1f} Hz: {bar} ({amp:.2f})")

        lines.append(f"\n  Cycle Types: {', '.join(params['cycle_types'])}")

        return '\n'.join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

def generate_cyclic_market(
    base_price: float = 100.0,
    cycles: List[Tuple[int, float]] = None,  # (period, amplitude) pairs
    noise: float = 0.3,
    length: int = 200
) -> List[float]:
    """Generate synthetic market data with known cycles."""
    if cycles is None:
        # Default: 7-bar short cycle, 23-bar medium cycle, 61-bar long cycle
        cycles = [(7, 2.0), (23, 5.0), (61, 10.0)]

    prices = []

    for i in range(length):
        price = base_price

        # Add each cycle
        for period, amplitude in cycles:
            phase = 2 * math.pi * i / period
            price += amplitude * math.sin(phase)

        # Add noise
        price += (2 * (hash(str(i)) % 1000) / 1000 - 1) * noise * base_price * 0.01

        prices.append(price)

    return prices


def demo():
    """Demonstrate cycle detection and spectral synthesis."""
    print("=" * 70)
    print("MARKET HARMONICS - True Cyclical Pattern Detection")
    print("=" * 70)
    print("""
    The insight: Markets oscillate at natural frequencies.

    - Every ~7 bars might bounce off support
    - Every ~20 bars might complete a swing
    - Every ~60 bars might have a larger cycle

    These ARE the market's harmonics. We DETECT them, then HEAR them.
    """)

    # Generate market with known cycles
    print("\n" + "─" * 70)
    print("SYNTHETIC MARKET (known cycles: 7, 23, 61 bars)")
    print("─" * 70)

    known_cycles = [(7, 2.0), (23, 5.0), (61, 10.0)]
    prices = generate_cyclic_market(cycles=known_cycles, length=250)

    print(f"\nGenerated {len(prices)} bars with embedded cycles:")
    for period, amp in known_cycles:
        print(f"  - {period}-bar cycle (amplitude: {amp})")

    # Detect cycles
    detector = CycleDetector(min_period=3, max_period=80)
    spectrum = detector.analyze(prices)

    print(f"\nDetected {len(spectrum.cycles)} cycles:")
    print("─" * 40)

    for cycle in spectrum.get_dominant_cycles(8):
        bar = '█' * int(cycle.amplitude * 20)
        print(f"  Period: {cycle.period:5.1f} bars | "
              f"Strength: {bar} ({cycle.amplitude:.2f}) | "
              f"Type: {cycle.cycle_type} | "
              f"Confidence: {cycle.confidence:.2f}")

    # Show harmonic ratios
    ratios = spectrum.get_harmonic_ratios()
    if ratios:
        print(f"\n  Harmonic ratios (relative to fundamental):")
        for i, ratio in enumerate(ratios[:6]):
            print(f"    {i+1}: {ratio:.2f}x")

    # Synthesize sound
    synth = SpectralSynthesizer(base_freq=110.0)
    print("\n" + synth.describe_sound(spectrum))

    # Now test with a more realistic market pattern
    print("\n" + "═" * 70)
    print("REALISTIC MARKET (with support/resistance)")
    print("═" * 70)

    # Generate price action with support/resistance bounces
    realistic_prices = []
    price = 100.0
    support = 95.0
    resistance = 110.0

    for i in range(200):
        # Random walk with boundaries
        price += (2 * (hash(str(i * 7)) % 1000) / 1000 - 1) * 1.5

        # Add swing cycle
        price += 3 * math.sin(2 * math.pi * i / 17)

        # Bounce off support/resistance
        if price < support:
            price = support + 1.0
        elif price > resistance:
            price = resistance - 1.0

        realistic_prices.append(price)

    spectrum2 = detector.analyze(realistic_prices)

    print(f"\nDetected {len(spectrum2.cycles)} cycles:")
    for cycle in spectrum2.get_dominant_cycles(6):
        bar = '█' * int(cycle.amplitude * 20)
        print(f"  Period: {cycle.period:5.1f} bars | "
              f"Type: {cycle.cycle_type:20} | "
              f"Strength: {bar} | "
              f"Confidence: {cycle.confidence:.2f}")

    print("\n" + synth.describe_sound(spectrum2))

    # Summary
    print("\n" + "═" * 70)
    print("KEY INSIGHT")
    print("═" * 70)
    print("""
    The market's ACTUAL cyclical patterns become musical harmonics:

    DETECTED CYCLES              →    MUSICAL HARMONICS
    ───────────────────────────────────────────────────
    7-bar support bounce         →    High harmonic (fast oscillation)
    17-bar swing cycle           →    Mid harmonic
    61-bar trend cycle           →    Fundamental (bass)

    Support/resistance bounces   →    Characteristic timbre
    Cycle regularity             →    Harmonic clarity
    Multiple cycles              →    Rich, complex sound

    This is TRUE structural sonification:
    - The market's spectrum IS the instrument's spectrum
    - Cycle detection = harmonic analysis
    - Trading patterns become musical patterns

    You don't just HEAR the market - you hear its NATURAL FREQUENCIES.
    """)


if __name__ == "__main__":
    demo()
