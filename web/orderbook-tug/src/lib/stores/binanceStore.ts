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
	TensionSnapshot
} from '../types';
import {
	calculateCOM,
	calculateTensionField,
	createTensionSnapshot,
	findPatternMatches,
	calculateTradeDensity,
	calculateVWAP
} from '../utils/calculations';

// Configuration
const MAX_TRADES = 100;
const MAX_HISTORY = 200;
const DEPTH_LEVELS = 20;

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
	tick: 0
};

// Create the main store
function createBinanceStore() {
	const { subscribe, set, update } = writable<MarketState>(initialState);

	let depthSocket: WebSocket | null = null;
	let tradeSocket: WebSocket | null = null;

	function connect(symbol: string = 'BTCUSDT') {
		disconnect();

		const symbolLower = symbol.toLowerCase();

		update((state) => ({ ...state, symbol, connected: false }));

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
				tick: newTick
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

			return { ...state, trades, tradeDensity, vwap };
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
export const tensionHistory = derived(binanceStore, ($s) => $s.tensionHistory);
export const patternMatches = derived(binanceStore, ($s) => $s.patternMatches);
export const connected = derived(binanceStore, ($s) => $s.connected);

// Computed values - uses activeQuantity (within ±0.5 sigma) for more meaningful dominance
export const bidDominance = derived(binanceStore, ($s) => {
	if (!$s.bidCom || !$s.askCom) return 0.5;
	const total = $s.bidCom.activeQuantity + $s.askCom.activeQuantity;
	return total > 0 ? $s.bidCom.activeQuantity / total : 0.5;
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
