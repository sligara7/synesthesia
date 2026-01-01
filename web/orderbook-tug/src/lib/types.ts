/**
 * Order Book Tug-of-War Types
 */

export interface OrderLevel {
	price: number;
	quantity: number;
}

export interface OrderBook {
	bids: OrderLevel[];
	asks: OrderLevel[];
	lastUpdateId: number;
	timestamp: Date;
}

export interface Trade {
	price: number;
	quantity: number;
	timestamp: Date;
	isBuyerMaker: boolean; // true = sell aggressor, false = buy aggressor
}

export interface CenterOfMass {
	price: number;
	totalQuantity: number;
	activeQuantity: number;  // Volume within ±1 sigma of COM
	sigma: number;           // Standard deviation of price distribution
	side: 'bid' | 'ask';
}

export interface QuantileLine {
	percentile: number;
	bidPrice: number;
	askPrice: number;
}

export interface TensionField {
	lines: QuantileLine[];
	bidTotal: number;
	askTotal: number;
	convergenceRatio: number;
	shape: 'hourglass' | 'diamond' | 'parallel';
}

export interface TensionSnapshot {
	tick: number;
	timestamp: Date;
	bidCom: CenterOfMass;
	askCom: CenterOfMass;
	spread: number;
	slopeAngle: number;
	bidDominance: number;
}

export interface PatternMatch {
	historicalTick: number;
	similarityScore: number;
	angleDifference: number;
}

export interface TradeDensity {
	price: number;
	density: number; // 0-1 normalized
	count?: number;  // Raw count of trades in this bin
}

export interface Candle {
	startTime: Date;
	open: number;
	high: number;
	low: number;
	close: number;
	volume: number;
	tradeCount: number;
}

export interface TrendState {
	// Candle midpoint calculations
	bodyMid: number;           // (open + close) / 2 - center of candle body
	rangeMid: number;          // (high + low) / 2 - center of candle range
	center: number;            // (bodyMid + rangeMid) / 2 - overall price center
	momentum: number;          // center - prevCenter - directional change

	// Midpoint extremes
	upperMid: number;          // max(bodyMid, rangeMid)
	lowerMid: number;          // min(bodyMid, rangeMid)

	// Dynamic support/resistance (from midpoint movements)
	dynamicSupport: number;    // updates when lowerMid rises
	dynamicResistance: number; // updates when upperMid falls

	// Pressure calculations
	bullPressure: number;      // center - dynamicSupport + momentum
	bearPressure: number;      // center - dynamicResistance + momentum

	// Decomposed pressure components
	bullPositive: number;      // max(0, bullPressure)
	bullNegative: number;      // min(0, bullPressure)
	bearPositive: number;      // max(0, bearPressure)
	bearNegative: number;      // min(0, bearPressure)

	// Overall trend strength
	bullStrength: number;      // bullPositive + bearPositive (total upward force)
	bearStrength: number;      // bullNegative + bearNegative (total downward force)

	// Confirmed support/resistance (from actual highs/lows during pure trends)
	trendSupport: number;      // candle low during pure uptrend
	trendResistance: number;   // candle high during pure downtrend
}

export interface MarketState {
	symbol: string;
	orderBook: OrderBook | null;
	trades: Trade[];
	bidCom: CenterOfMass | null;
	askCom: CenterOfMass | null;
	tensionField: TensionField | null;
	tradeDensity: TradeDensity[];
	vwap: number | null;
	tensionHistory: TensionSnapshot[];
	patternMatches: PatternMatch[];
	connected: boolean;
	tick: number;
	// 30-second candles
	appStartTime: Date | null;
	currentCandle: Candle | null;
	candles: Candle[];
	// Trend analysis
	trendState: TrendState | null;
	prevCenter: number | null;    // Previous center for calculating momentum
	prevUpperMid: number | null;  // Previous upperMid for dynamic resistance
	prevLowerMid: number | null;  // Previous lowerMid for dynamic support
	// Support/resistance history for histogram
	supportHistory: number[];     // Last N trendSupport values
	resistanceHistory: number[];  // Last N trendResistance values
}
