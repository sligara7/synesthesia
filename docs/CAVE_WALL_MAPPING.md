# Cave Wall Mapping - 4-Surface Market Structure

## FINALIZED DESIGN

### The 4 Market Fundamentals

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│   1. PRICE          - Where we are (Floor height)                │
│   2. VOLUME         - Activity/pressure (Ceiling pressure)       │
│   3. SUPPORT/RESIST - Historical memory (Ledges & barriers)      │
│   4. ORDER BOOK     - Liquidity (Left & right walls)             │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### The 3D Cave Model

```
                    CEILING: Volume + Resistance Barriers
                    ┌─────────────────────────────────┐
                   ╱  ▼▼▼ (resistance = stalactites)   ╲
    LEFT WALL     ╱                                     ╲     RIGHT WALL
    Bid Depth    ╱                                       ╲    Ask Depth
    ┌───────────╱                                         ╲───────────┐
    │           │                                         │           │
    │  SELL     │              ROCKET                     │    BUY    │
    │  INTO     │                >                        │    INTO   │
    │  BIDS     │                                         │    ASKS   │
    │           │                                         │           │
    └───────────╲                                         ╱───────────┘
                 ╲                                       ╱
                  ╲    ═══╔══╗═══ (support = ledge)     ╱
                   ╲______║  ║________________________╱
                    FLOOR: Price + Support Ledges
```

### Wall Mappings

| Surface | Market Data | Visual Representation | Behavior |
|---------|-------------|----------------------|----------|
| **Floor** | Price level | Terrain height | Higher price = higher floor |
| **Floor+** | Support levels | Ledges/platforms | Safe zones to land on |
| **Ceiling** | Volume | Ceiling height | High volume = ceiling drops (pressure) |
| **Ceiling+** | Resistance levels | Stalactites/barriers | Dangerous to hit |
| **Left Wall** | Bid depth | Wall distance | Thin bids = wall far (no support) |
| **Right Wall** | Ask depth | Wall distance | Thin asks = wall far (easy to buy) |

### Player Actions

| Joystick | Trading Action | Market Interaction |
|----------|----------------|-------------------|
| **UP** | Go long / Stay long | Express bullish view |
| **DOWN** | Go short / Stay short | Express bearish view |
| **RIGHT** | Buy (add long OR cover short) | Consume asks (lift offers) |
| **LEFT** | Sell (reduce long OR add short) | Consume bids (hit bids) |

### Combined Actions

```
                        BUY (RIGHT)
                            │
                            │
            ┌───────────────┼───────────────┐
            │               │               │
            │   UP+LEFT     │   UP+RIGHT    │
            │   Sell to     │   Buy to      │
            │   reduce long │   add long    │
            │               │               │
 SHORT ─────┼───────────────┼───────────────┼───── LONG
 (DOWN)     │               │               │      (UP)
            │               │               │
            │  DOWN+LEFT    │  DOWN+RIGHT   │
            │  Sell to      │  Buy to       │
            │  add short    │  cover short  │
            │               │               │
            └───────────────┼───────────────┘
                            │
                            │
                        SELL (LEFT)
```

### Wall Physics (How Walls Affect Gameplay)

| Wall State | Market Meaning | Game Feel |
|------------|----------------|-----------|
| Right wall **close** | Heavy asks (resistance) | Hard to buy - pushing into wall |
| Right wall **far** | Thin asks | Easy to buy - open space |
| Left wall **close** | Heavy bids (support) | Sells absorbed - wall catches you |
| Left wall **far** | Thin bids | Sells drop price - no safety net |

### The Force Field Metaphor

```
    ┌────────────────────────────────────────────┐
    │                                            │
    │   SELLING PRESSURE (volume, resistance)    │
    │              pushes DOWN                   │
    │                   ▼                        │
    │                                            │
    │  BIDS    ◄──     ●     ──►    ASKS        │
    │ (left)       (rocket)        (right)       │
    │ absorbs                      resists       │
    │ sells                        buys          │
    │                   ▲                        │
    │              pushes UP                     │
    │   BUYING PRESSURE (bid depth, support)     │
    │                                            │
    └────────────────────────────────────────────┘
```

The cave is a **force field visualization** of market pressures.
The rocket navigates through competing forces.

---

## DETAILED SPECIFICATIONS

## Support/Resistance Integration

**Support → Floor Features**
```
    Normal floor:      ════════════════

    With support:      ════╔═══╗════════
                           ║   ║  ← Platform/ledge
                           ╚═══╝    (safe to land on)
```

**Resistance → Ceiling Features**
```
    Normal ceiling:    ════════════════

    With resistance:   ════╔═══╗════════
                           ║▼▼▼║  ← Stalactites/barrier
                           ╚═══╝    (dangerous to hit)
```

**Why this works:**
- Support = price floor where buyers appear → literal floor ledge
- Resistance = price ceiling where sellers appear → literal ceiling barrier
- Breakout = passing through the barrier

### Option B: Build into Side Walls

Less intuitive because S/R are horizontal price levels, not separate dimensions.

## Proposed: Left/Right Walls

What fundamental market data is DIRECTIONAL or represents OPPOSING FORCES?

### Option 1: Order Book Depth

```
LEFT WALL = Bid Depth (Buying Pressure)
RIGHT WALL = Ask Depth (Selling Pressure)

                    │
    BIDS            │            ASKS
    (buy orders)    │      (sell orders)
    ████████        │  ████████████████
    ████████████    │  ████████████
    ████████        │  ████████
    ████            │  ████
                    │
                 PRICE
```

**Mapping:**
- Deep bids (many buy orders) = LEFT wall far away (safety net below)
- Deep asks (many sell orders) = RIGHT wall close (resistance overhead)
- Thin bids = LEFT wall close (no support, danger of gap down)
- Thin asks = RIGHT wall far away (easy to push higher)

**Game feel:**
- Heavy selling pressure → right wall squeezes in → hard to go up
- Heavy buying support → left wall recedes → soft landing if you fall

### Option 2: Momentum vs Mean Reversion

```
LEFT WALL = Trend/Momentum
RIGHT WALL = Mean Reversion Pressure (RSI, distance from MA)
```

**Mapping:**
- Strong uptrend = LEFT wall pushes you forward (tailwind)
- Strong downtrend = LEFT wall pulls back (headwind)
- Overbought (RSI > 70) = RIGHT wall closes in (pressure to fall)
- Oversold (RSI < 30) = RIGHT wall recedes (room to bounce)

**Game feel:**
- Uptrend + overbought = squeezed between momentum and mean reversion
- Creates natural tension zones

### Option 3: Relative Strength

```
LEFT WALL = Asset vs Market (SPY, sector)
RIGHT WALL = Asset vs Historical Self (deviation from mean)
```

**Mapping:**
- Outperforming market = LEFT wall pushes you right (you're winning)
- Underperforming = LEFT wall pulls back (you're lagging)
- Above historical average = RIGHT wall close (stretched)
- Below historical average = RIGHT wall far (compressed)

## Recommended Mapping

Based on fundamental market data priority:

| Surface | Primary Data | Secondary Data | Game Feel |
|---------|--------------|----------------|-----------|
| **FLOOR** | Price level | Support levels (ledges) | Navigate terrain |
| **CEILING** | Volume | Resistance levels (barriers) | Avoid squeeze |
| **LEFT** | Bid depth / Buy pressure | Trend direction | Safety net / momentum |
| **RIGHT** | Ask depth / Sell pressure | Overbought/oversold | Resistance / ceiling |

## Full 4-Wall Visualization

```
    CEILING: Volume + Resistance
    ▼▼▼▼▼▼▼▼▼ (resistance = stalactites)
    ┌─────────────────────────────────────┐
    │                                     │
    │  BID DEPTH          ASK DEPTH       │
    │  (buy pressure)     (sell pressure) │
    │  ████│                    │████████ │
    │  ████│                    │████████ │
    │  ████│        >           │████     │
    │  ████│      rocket        │         │
    │      │                    │         │
    └─────────────────────────────────────┘
    ══════╔═══╗════════════════════════════
          ║   ║ (support = ledge)
    FLOOR: Price + Support
```

## Alternative: Simpler 2.5D Model

If 4 walls is too complex, use 2.5D:

```
    ┌─────────────────────────────────────┐
    │ CEILING: Volume + Resistance        │
    │                                     │
    │                                     │
    │         > rocket                    │
    │                                     │
    │ FLOOR: Price + Support              │
    └─────────────────────────────────────┘

    WIDTH = Volatility × (1 / Bid-Ask Spread)

    Narrow passage = high volatility OR illiquid market
```

## Data Sources

| Data | Source | Update Frequency |
|------|--------|------------------|
| Price | OHLC bars | Per bar (1 min) |
| Volume | Bar volume | Per bar |
| Volatility | ATR or bar range | Per bar |
| Support/Resistance | Detected from history | Rolling window |
| Bid/Ask depth | Level 2 data | Real-time (if available) |
| Trend | SMA crossover or ADX | Per bar |
| RSI/Overbought | Calculated | Per bar |

## Implementation Priority

1. **Phase 1:** Floor (price) + Ceiling (volume) + Width (volatility)
   - Already implemented ✓

2. **Phase 2:** Add Support/Resistance
   - Detect S/R levels from price history
   - Render as floor ledges and ceiling barriers

3. **Phase 3:** Add Order Book (if data available)
   - Left wall = bid depth
   - Right wall = ask depth

4. **Phase 4:** Add Derived Indicators
   - Trend direction affects momentum/wind
   - RSI affects color/lighting (overbought = red glow)

## The Key Insight

Each wall represents a FORCE acting on price:

```
    ┌────────────────────────────────────────────┐
    │                                            │
    │   SELLING PRESSURE (volume, resistance)    │
    │              pushes DOWN                   │
    │                   ▼                        │
    │                                            │
    │  BIDS    ◄──     ●     ──►    ASKS        │
    │ (support)     (price)    (resistance)      │
    │                                            │
    │                   ▲                        │
    │              pushes UP                     │
    │   BUYING PRESSURE (bid depth, support)     │
    │                                            │
    └────────────────────────────────────────────┘
```

The cave is a FORCE FIELD visualization of market pressures.
The rocket navigates through the competing forces.
