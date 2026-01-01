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
}
