# Order Book Tug of War - Web UI

Real-time order book visualization showing the battle between buyers and sellers.

## Features

- **Live Binance WebSocket** - Real-time order book and trade data
- **Center of Mass Analysis** - Volume-weighted average price for bid/ask sides
- **Tension Field** - Quantile lines connecting bid/ask distributions (10%-90%)
- **Trade Density** - Where trades actually execute (shown in center column)
- **VWAP** - Volume-weighted average price of recent trades
- **Pattern Matching** - Detects when similar tension patterns repeat

## Quick Start

```bash
cd web/orderbook-tug

# Install dependencies
npm install

# Start development server
npm run dev
```

Then open http://localhost:5173

## Visualization Guide

```
     BIDS          TRADES         ASKS
     ████           ▓▓             ████
     ██████         ▓▓▓▓           ██████
●────██████████────▓▓▓▓▓▓────██████████────●
BID  ████████        ▓▓         ████████    ASK
COM  ██████          ▓            ██████    COM
     ████                          ████
```

- **Green bars (left)** - Bid volume at each price level
- **Red bars (right)** - Ask volume at each price level
- **Orange bars (center)** - Trade density (where trades happen)
- **White line** - Tension line connecting bid COM to ask COM
- **Purple dashed** - VWAP line
- **Gray lines** - Quantile mapping (10%, 20%, ..., 90%)

## Shape Analysis

| Shape | Convergence | Meaning |
|-------|-------------|---------|
| HOURGLASS | > 1.1 | Lines converge at center - pressure concentrated near mid-prices |
| DIAMOND | < 0.9 | Lines diverge at center - extreme prices have more volume |
| PARALLEL | ≈ 1.0 | Uniform tension across all price levels |

## Tech Stack

- SvelteKit
- TypeScript
- D3.js for visualization
- Binance WebSocket API
