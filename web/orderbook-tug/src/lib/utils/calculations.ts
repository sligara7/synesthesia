/**
 * Order Book Analysis Calculations
 */

import type {
	OrderLevel,
	CenterOfMass,
	QuantileLine,
	TensionField,
	TensionSnapshot,
	Trade,
	TradeDensity,
	PatternMatch
} from '../types';

/**
 * Calculate center of mass for one side of the order book
 * COM = Σ(price × quantity) / Σ(quantity)
 * Also calculates volume within ±0.5 sigma of COM (the "active" volume)
 */
export function calculateCOM(levels: OrderLevel[], side: 'bid' | 'ask'): CenterOfMass {
	if (levels.length === 0) {
		return { price: 0, totalQuantity: 0, activeQuantity: 0, sigma: 0, side };
	}

	let weightedSum = 0;
	let totalQty = 0;

	for (const level of levels) {
		weightedSum += level.price * level.quantity;
		totalQty += level.quantity;
	}

	const comPrice = totalQty > 0 ? weightedSum / totalQty : levels[0].price;

	// Calculate weighted standard deviation
	let varianceSum = 0;
	for (const level of levels) {
		const diff = level.price - comPrice;
		varianceSum += level.quantity * diff * diff;
	}
	const sigma = totalQty > 0 ? Math.sqrt(varianceSum / totalQty) : 0;

	// Calculate volume within COM ± 0.5 sigma (the "active" volume)
	const halfSigma = sigma * 0.5;
	let activeQty = 0;
	for (const level of levels) {
		if (Math.abs(level.price - comPrice) <= halfSigma) {
			activeQty += level.quantity;
		}
	}

	return {
		price: comPrice,
		totalQuantity: totalQty,
		activeQuantity: activeQty,
		sigma: sigma,
		side
	};
}

/**
 * Calculate quantile prices for a distribution
 * Treats volume as probability mass and finds CDF percentiles
 */
export function calculateQuantiles(
	levels: OrderLevel[],
	percentiles: number[] = [10, 20, 30, 40, 50, 60, 70, 80, 90]
): Map<number, number> {
	if (levels.length === 0) return new Map();

	// Sort by price (bids: high to low, asks: low to high)
	const sorted = [...levels];

	// Calculate total volume
	const total = sorted.reduce((sum, l) => sum + l.quantity, 0);
	if (total <= 0) return new Map();

	// Build CDF
	const cdf: Array<{ price: number; cumPct: number }> = [];
	let cumulative = 0;

	for (const level of sorted) {
		cumulative += level.quantity;
		cdf.push({ price: level.price, cumPct: (cumulative / total) * 100 });
	}

	// Interpolate quantiles
	const result = new Map<number, number>();

	for (const targetPct of percentiles) {
		let prevPrice = cdf[0].price;
		let prevPct = 0;

		for (const { price, cumPct } of cdf) {
			if (cumPct >= targetPct) {
				// Interpolate
				if (cumPct === prevPct) {
					result.set(targetPct, price);
				} else {
					const ratio = (targetPct - prevPct) / (cumPct - prevPct);
					result.set(targetPct, prevPrice + ratio * (price - prevPrice));
				}
				break;
			}
			prevPrice = price;
			prevPct = cumPct;
		}

		// If we didn't find it, use last price
		if (!result.has(targetPct)) {
			result.set(targetPct, cdf[cdf.length - 1].price);
		}
	}

	return result;
}

/**
 * Calculate the full tension field between bid and ask distributions
 */
export function calculateTensionField(
	bids: OrderLevel[],
	asks: OrderLevel[]
): TensionField {
	const percentiles = [10, 20, 30, 40, 50, 60, 70, 80, 90];

	// For bids: sort highest first (accumulate from top of book)
	const sortedBids = [...bids].sort((a, b) => b.price - a.price);
	// For asks: sort lowest first (accumulate from bottom of book)
	const sortedAsks = [...asks].sort((a, b) => a.price - b.price);

	const bidQuantiles = calculateQuantiles(sortedBids, percentiles);
	const askQuantiles = calculateQuantiles(sortedAsks, percentiles);

	const lines: QuantileLine[] = [];

	for (const pct of percentiles) {
		const bidPrice = bidQuantiles.get(pct);
		// Mirror the percentile: 10% bid → 90% ask, 20% bid → 80% ask, etc.
		// This creates a fan pattern that doesn't cross
		const askPrice = askQuantiles.get(100 - pct);

		if (bidPrice !== undefined && askPrice !== undefined) {
			lines.push({
				percentile: pct,
				bidPrice,
				askPrice
			});
		}
	}

	const bidTotal = bids.reduce((sum, l) => sum + l.quantity, 0);
	const askTotal = asks.reduce((sum, l) => sum + l.quantity, 0);

	// Calculate convergence ratio
	let convergenceRatio = 1.0;
	let shape: 'hourglass' | 'diamond' | 'parallel' = 'parallel';

	if (lines.length >= 5) {
		const outerSpread =
			(lines[lines.length - 1].askPrice -
				lines[lines.length - 1].bidPrice +
				(lines[0].askPrice - lines[0].bidPrice)) /
			2;
		const innerIdx = Math.floor(lines.length / 2);
		const innerSpread = lines[innerIdx].askPrice - lines[innerIdx].bidPrice;

		if (innerSpread > 0) {
			convergenceRatio = outerSpread / innerSpread;
		}

		if (convergenceRatio > 1.1) {
			shape = 'hourglass';
		} else if (convergenceRatio < 0.9) {
			shape = 'diamond';
		}
	}

	return {
		lines,
		bidTotal,
		askTotal,
		convergenceRatio,
		shape
	};
}

/**
 * Create a tension snapshot from current market state
 */
export function createTensionSnapshot(
	tick: number,
	bids: OrderLevel[],
	asks: OrderLevel[]
): TensionSnapshot {
	const bidCom = calculateCOM(bids, 'bid');
	const askCom = calculateCOM(asks, 'ask');

	const bestBid = bids.length > 0 ? bids[0].price : 0;
	const bestAsk = asks.length > 0 ? asks[0].price : 0;
	const spread = Math.max(0.0001, bestAsk - bestBid);

	// Calculate slope angle
	const priceDiff = askCom.price - bidCom.price;
	const normalized = priceDiff / spread;
	const slopeAngle = (Math.atan2(normalized, 1.0) * 180) / Math.PI;

	// Calculate bid dominance
	const total = bidCom.totalQuantity + askCom.totalQuantity;
	const bidDominance = total > 0 ? bidCom.totalQuantity / total : 0.5;

	return {
		tick,
		timestamp: new Date(),
		bidCom,
		askCom,
		spread,
		slopeAngle,
		bidDominance
	};
}

/**
 * Find pattern matches between current and historical tension
 */
export function findPatternMatches(
	current: TensionSnapshot,
	history: TensionSnapshot[],
	angleTolerance: number = 5.0,
	minScore: number = 0.6,
	minAge: number = 5
): PatternMatch[] {
	const matches: PatternMatch[] = [];

	for (const h of history) {
		const age = current.tick - h.tick;
		if (age < minAge) continue;

		// Calculate similarity score
		const angleDiff = Math.abs(current.slopeAngle - h.slopeAngle);
		const angleSim = Math.max(0, 1.0 - angleDiff / 45.0);

		const magDiff = Math.abs(
			(current.askCom.price - current.bidCom.price) / current.spread -
				(h.askCom.price - h.bidCom.price) / h.spread
		);
		const magSim = Math.max(0, 1.0 - magDiff);

		const balanceDiff = Math.abs(current.bidDominance - h.bidDominance);
		const balanceSim = Math.max(0, 1.0 - balanceDiff * 2);

		const score = 0.6 * angleSim + 0.2 * magSim + 0.2 * balanceSim;

		if (score >= minScore) {
			matches.push({
				historicalTick: h.tick,
				similarityScore: score,
				angleDifference: angleDiff
			});
		}
	}

	// Sort by score descending
	matches.sort((a, b) => b.similarityScore - a.similarityScore);

	return matches;
}

/**
 * Calculate trade density using unique prices
 * Mimics numpy: unique_prices, counts = np.unique(prices, return_counts=True)
 * Then normalize counts to sum to 1
 */
export function calculateTradeDensity(
	trades: Trade[],
	_minPrice: number,  // unused, kept for API compatibility
	_maxPrice: number   // unused, kept for API compatibility
): TradeDensity[] {
	if (trades.length === 0) {
		return [];
	}

	// Extract prices from trades
	const prices = trades.map(t => t.price);

	// Count occurrences of each unique price (like np.unique with return_counts=True)
	const countMap = new Map<number, number>();
	for (const price of prices) {
		countMap.set(price, (countMap.get(price) || 0) + 1);
	}

	// Convert to array and sort by price
	const uniquePrices = Array.from(countMap.entries())
		.map(([price, count]) => ({ price, count }))
		.sort((a, b) => a.price - b.price);

	// Normalize counts to sum to 1
	const totalCount = trades.length;

	// Return normalized densities
	return uniquePrices.map(({ price, count }) => ({
		price,
		density: count / totalCount,  // Normalized so all densities sum to 1
		count
	}));
}

/**
 * Calculate VWAP from trades
 */
export function calculateVWAP(trades: Trade[]): number | null {
	if (trades.length === 0) return null;

	let totalValue = 0;
	let totalVolume = 0;

	for (const trade of trades) {
		totalValue += trade.price * trade.quantity;
		totalVolume += trade.quantity;
	}

	return totalVolume > 0 ? totalValue / totalVolume : null;
}
