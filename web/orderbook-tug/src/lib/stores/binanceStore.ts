/**
 * Binance WebSocket Store
 *
 * Connects to Binance WebSocket streams for real-time order book and trade data.
 */

import { writable, derived, get } from 'svelte/store';
import type {
	OrderBook,
	OrderLevel,
	Trade,
	MarketState,
	TensionSnapshot,
	Candle,
	TrendState
} from '../types';
import {
	calculateCOM,
	calculateTensionField,
	createTensionSnapshot,
	findPatternMatches,
	calculateTradeDensity,
	calculateVWAP,
	calculateTrendState
} from '../utils/calculations';

// Configuration
const MAX_TRADES = 100;
const MAX_HISTORY = 200;
const DEPTH_LEVELS = 20;
const CANDLE_INTERVAL_MS = 30000; // 30 seconds
const MAX_SR_HISTORY = 30; // Support/resistance history length

// Initial state
const initialState: MarketState = {
	symbol: 'BTCUSDT',
	orderBook: null,
	trades: [],
	bidCom: null,
	askCom: null,
	tensionField: null,
	tradeDensity: [],
	vwap: null,
	tensionHistory: [],
	patternMatches: [],
	connected: false,
	tick: 0,
	appStartTime: null,
	currentCandle: null,
	candles: [],
	trendState: null,
	prevCenter: null,
	prevUpperMid: null,
	prevLowerMid: null,
	supportHistory: [],
	resistanceHistory: [],
	supIntersectHistory: [],
	resIntersectHistory: [],
	lastIntersectHistory: []
};

// Create the main store
function createBinanceStore() {
	const { subscribe, set, update } = writable<MarketState>(initialState);

	let depthSocket: WebSocket | null = null;
	let tradeSocket: WebSocket | null = null;

	function connect(symbol: string = 'BTCUSDT') {
		disconnect();

		const symbolLower = symbol.toLowerCase();
		const now = new Date();

		update((state) => ({
			...state,
			symbol,
			connected: false,
			appStartTime: now,
			currentCandle: null,
			candles: []
		}));

		// Connect to depth stream (order book updates)
		const depthUrl = `wss://stream.binance.us:9443/ws/${symbolLower}@depth${DEPTH_LEVELS}@100ms`;
		depthSocket = new WebSocket(depthUrl);

		depthSocket.onopen = () => {
			console.log('Depth stream connected');
			update((state) => ({ ...state, connected: true }));
		};

		depthSocket.onmessage = (event) => {
			try {
				const data = JSON.parse(event.data);
				handleDepthUpdate(data);
			} catch (e) {
				console.error('Error parsing depth message:', e);
			}
		};

		depthSocket.onerror = (error) => {
			console.error('Depth stream error:', error);
		};

		depthSocket.onclose = () => {
			console.log('Depth stream closed');
			update((state) => ({ ...state, connected: false }));
		};

		// Connect to trade stream (using aggTrade for aggregated trades)
		const tradeUrl = `wss://stream.binance.us:9443/ws/${symbolLower}@aggTrade`;
		console.log('Connecting to trade stream:', tradeUrl);
		tradeSocket = new WebSocket(tradeUrl);

		tradeSocket.onopen = () => {
			console.log('Trade stream connected');
		};

		tradeSocket.onmessage = (event) => {
			try {
				const data = JSON.parse(event.data);
				handleTradeUpdate(data);
			} catch (e) {
				console.error('Error parsing trade message:', e);
			}
		};

		tradeSocket.onerror = (error) => {
			console.error('Trade stream error:', error);
		};

		tradeSocket.onclose = () => {
			console.log('Trade stream closed');
		};
	}

	function disconnect() {
		if (depthSocket) {
			depthSocket.close();
			depthSocket = null;
		}
		if (tradeSocket) {
			tradeSocket.close();
			tradeSocket = null;
		}
		update((state) => ({ ...state, connected: false }));
	}

	function handleDepthUpdate(data: any) {
		update((state) => {
			// Parse bids and asks
			const bids: OrderLevel[] = (data.bids || []).map((b: string[]) => ({
				price: parseFloat(b[0]),
				quantity: parseFloat(b[1])
			}));

			const asks: OrderLevel[] = (data.asks || []).map((a: string[]) => ({
				price: parseFloat(a[0]),
				quantity: parseFloat(a[1])
			}));

			// Sort bids high to low, asks low to high
			bids.sort((a, b) => b.price - a.price);
			asks.sort((a, b) => a.price - b.price);

			const orderBook: OrderBook = {
				bids,
				asks,
				lastUpdateId: data.lastUpdateId || 0,
				timestamp: new Date()
			};

			// Calculate COMs
			const bidCom = calculateCOM(bids, 'bid');
			const askCom = calculateCOM(asks, 'ask');

			// Calculate tension field
			const tensionField = calculateTensionField(bids, asks);

			// Create tension snapshot and find patterns
			const newTick = state.tick + 1;
			const snapshot = createTensionSnapshot(newTick, bids, asks);

			// Add to history
			const tensionHistory = [...state.tensionHistory, snapshot];
			if (tensionHistory.length > MAX_HISTORY) {
				tensionHistory.shift();
			}

			// Find pattern matches
			const patternMatches = findPatternMatches(snapshot, tensionHistory);

			// Calculate trade density based on current price range
			const allPrices = [...bids.map((b) => b.price), ...asks.map((a) => a.price)];
			const minPrice = Math.min(...allPrices);
			const maxPrice = Math.max(...allPrices);
			const tradeDensity = calculateTradeDensity(state.trades, minPrice, maxPrice);

			// Calculate VWAP
			const vwap = calculateVWAP(state.trades);

			// Calculate intersection positions (0-1 normalized along COM line)
			// t = 0 means at bid COM, t = 1 means at ask COM
			let supIntersectHistory = state.supIntersectHistory;
			let resIntersectHistory = state.resIntersectHistory;
			let lastIntersectHistory = state.lastIntersectHistory;

			if (bidCom && askCom && askCom.price !== bidCom.price) {
				const priceRange = askCom.price - bidCom.price;

				// Support intersection position
				if (state.trendState?.trendSupport) {
					const supT = (state.trendState.trendSupport - bidCom.price) / priceRange;
					supIntersectHistory = [...supIntersectHistory, supT];
					if (supIntersectHistory.length > 200) supIntersectHistory = supIntersectHistory.slice(-200);
				}

				// Resistance intersection position
				if (state.trendState?.trendResistance) {
					const resT = (state.trendState.trendResistance - bidCom.price) / priceRange;
					resIntersectHistory = [...resIntersectHistory, resT];
					if (resIntersectHistory.length > 200) resIntersectHistory = resIntersectHistory.slice(-200);
				}

				// Last price intersection position
				if (state.trades.length > 0) {
					const lastPrice = state.trades[state.trades.length - 1].price;
					const lastT = (lastPrice - bidCom.price) / priceRange;
					lastIntersectHistory = [...lastIntersectHistory, lastT];
					if (lastIntersectHistory.length > 200) lastIntersectHistory = lastIntersectHistory.slice(-200);
				}
			}

			return {
				...state,
				orderBook,
				bidCom,
				askCom,
				tensionField,
				tensionHistory,
				patternMatches,
				tradeDensity,
				vwap,
				tick: newTick,
				supIntersectHistory,
				resIntersectHistory,
				lastIntersectHistory
			};
		});
	}

	function handleTradeUpdate(data: any) {
		update((state) => {
			const trade: Trade = {
				price: parseFloat(data.p),
				quantity: parseFloat(data.q),
				timestamp: new Date(data.T),
				isBuyerMaker: data.m // true = buyer is maker = sell aggressor
			};

			let trades = [...state.trades, trade];

			// Keep only the last MAX_TRADES
			while (trades.length > MAX_TRADES) {
				trades.shift();
			}

			// Recalculate trade density with each new trade
			let tradeDensity = state.tradeDensity;
			if (state.orderBook) {
				const allPrices = [
					...state.orderBook.bids.map((b) => b.price),
					...state.orderBook.asks.map((a) => a.price)
				];
				if (allPrices.length > 0) {
					const minPrice = Math.min(...allPrices);
					const maxPrice = Math.max(...allPrices);
					tradeDensity = calculateTradeDensity(trades, minPrice, maxPrice);
				}
			}

			// Calculate VWAP
			const vwap = calculateVWAP(trades);

			// Update 30-second candles
			let currentCandle = state.currentCandle;
			let candles = state.candles;
			let trendState = state.trendState;
			let prevCenter = state.prevCenter;
			let prevUpperMid = state.prevUpperMid;
			let prevLowerMid = state.prevLowerMid;
			let supportHistory = state.supportHistory;
			let resistanceHistory = state.resistanceHistory;

			if (state.appStartTime) {
				const now = trade.timestamp;
				const elapsed = now.getTime() - state.appStartTime.getTime();
				const currentInterval = Math.floor(elapsed / CANDLE_INTERVAL_MS);

				// Determine which interval the current candle belongs to
				let candleInterval = -1;
				if (currentCandle) {
					const candleElapsed = currentCandle.startTime.getTime() - state.appStartTime.getTime();
					candleInterval = Math.floor(candleElapsed / CANDLE_INTERVAL_MS);
				}

				// Check if we need to start a new candle
				if (!currentCandle || currentInterval > candleInterval) {
					// Save previous candle's close to use as new candle's open
					const prevClose = currentCandle?.close ?? trade.price;

					// Close current candle if it exists - save previous values for trend calc
					if (currentCandle && trendState) {
						candles = [...candles, currentCandle];
						// Save previous values for next candle's calculations
						prevCenter = trendState.center;
						prevUpperMid = trendState.upperMid;
						prevLowerMid = trendState.lowerMid;

						// Add to support/resistance history
						supportHistory = [...supportHistory, trendState.trendSupport];
						resistanceHistory = [...resistanceHistory, trendState.trendResistance];

						// Keep only last N values
						if (supportHistory.length > MAX_SR_HISTORY) {
							supportHistory = supportHistory.slice(-MAX_SR_HISTORY);
						}
						if (resistanceHistory.length > MAX_SR_HISTORY) {
							resistanceHistory = resistanceHistory.slice(-MAX_SR_HISTORY);
						}
					}

					// Start new candle - open from previous close for continuity
					const candleStartTime = new Date(
						state.appStartTime.getTime() + currentInterval * CANDLE_INTERVAL_MS
					);
					currentCandle = {
						startTime: candleStartTime,
						open: prevClose,
						high: Math.max(prevClose, trade.price),
						low: Math.min(prevClose, trade.price),
						close: trade.price,
						volume: trade.quantity,
						tradeCount: 1
					};
				} else {
					// Update current candle
					currentCandle = {
						...currentCandle,
						high: Math.max(currentCandle.high, trade.price),
						low: Math.min(currentCandle.low, trade.price),
						close: trade.price,
						volume: currentCandle.volume + trade.quantity,
						tradeCount: currentCandle.tradeCount + 1
					};
				}

				// Calculate trend state for current candle
				const prevDynamicSupport = trendState?.dynamicSupport ?? null;
				const prevDynamicResistance = trendState?.dynamicResistance ?? null;
				const prevTrendSupport = trendState?.trendSupport ?? null;
				const prevTrendResistance = trendState?.trendResistance ?? null;
				trendState = calculateTrendState(
					currentCandle,
					prevCenter,
					prevUpperMid,
					prevLowerMid,
					prevDynamicSupport,
					prevDynamicResistance,
					prevTrendSupport,
					prevTrendResistance
				);
			}

			return { ...state, trades, tradeDensity, vwap, currentCandle, candles, trendState, prevCenter, prevUpperMid, prevLowerMid, supportHistory, resistanceHistory };
		});
	}

	function reset() {
		update((state) => ({
			...state,
			trades: [],
			tensionHistory: [],
			patternMatches: [],
			tick: 0
		}));
	}

	return {
		subscribe,
		connect,
		disconnect,
		reset
	};
}

export const binanceStore = createBinanceStore();

// Derived stores for specific data
export const orderBook = derived(binanceStore, ($s) => $s.orderBook);
export const bidCom = derived(binanceStore, ($s) => $s.bidCom);
export const askCom = derived(binanceStore, ($s) => $s.askCom);
export const tensionField = derived(binanceStore, ($s) => $s.tensionField);
export const trades = derived(binanceStore, ($s) => $s.trades);
export const tradeDensity = derived(binanceStore, ($s) => $s.tradeDensity);
export const vwap = derived(binanceStore, ($s) => $s.vwap);
export const lastTradePrice = derived(binanceStore, ($s) =>
	$s.trades.length > 0 ? $s.trades[$s.trades.length - 1].price : null
);
export const tensionHistory = derived(binanceStore, ($s) => $s.tensionHistory);
export const patternMatches = derived(binanceStore, ($s) => $s.patternMatches);
export const connected = derived(binanceStore, ($s) => $s.connected);
export const currentCandle = derived(binanceStore, ($s) => $s.currentCandle);
export const candles = derived(binanceStore, ($s) => $s.candles);
export const trendState = derived(binanceStore, ($s) => $s.trendState);
export const supportHistory = derived(binanceStore, ($s) => $s.supportHistory);
export const resistanceHistory = derived(binanceStore, ($s) => $s.resistanceHistory);
export const supIntersectHistory = derived(binanceStore, ($s) => $s.supIntersectHistory);
export const resIntersectHistory = derived(binanceStore, ($s) => $s.resIntersectHistory);
export const lastIntersectHistory = derived(binanceStore, ($s) => $s.lastIntersectHistory);

// Computed values - uses activeQuantity (within ±1 sigma) for more meaningful dominance
export const bidDominance = derived(binanceStore, ($s) => {
	if (!$s.bidCom || !$s.askCom) return 0.5;
	const total = $s.bidCom.activeQuantity + $s.askCom.activeQuantity;
	return total > 0 ? $s.bidCom.activeQuantity / total : 0.5;
});

// Lean: which COM is the market trading toward?
// Positive = leaning toward asks (bullish), Negative = leaning toward bids (bearish)
export const lean = derived(binanceStore, ($s) => {
	if (!$s.bidCom || !$s.askCom || $s.trades.length === 0) return 0;
	const lastPrice = $s.trades[$s.trades.length - 1].price;
	const askDistance = Math.abs($s.askCom.price - lastPrice);
	const bidDistance = Math.abs($s.bidCom.price - lastPrice);
	const total = bidDistance + askDistance;
	return total > 0 ? (bidDistance - askDistance) / total : 0;
});

export const currentSpread = derived(binanceStore, ($s) => {
	if (!$s.orderBook) return 0;
	const bestBid = $s.orderBook.bids[0]?.price || 0;
	const bestAsk = $s.orderBook.asks[0]?.price || 0;
	return bestAsk - bestBid;
});

export const priceRange = derived(binanceStore, ($s) => {
	if (!$s.orderBook) return { min: 0, max: 0 };
	const allPrices = [
		...$s.orderBook.bids.map((b) => b.price),
		...$s.orderBook.asks.map((a) => a.price)
	];
	return {
		min: Math.min(...allPrices),
		max: Math.max(...allPrices)
	};
});
