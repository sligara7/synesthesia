<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import * as d3 from 'd3';
	import type { OrderLevel, QuantileLine, TradeDensity, CenterOfMass } from '../types';

	export let bids: OrderLevel[] = [];
	export let asks: OrderLevel[] = [];
	export let quantileLines: QuantileLine[] = [];
	export let tradeDensity: TradeDensity[] = [];
	export let bidCom: CenterOfMass | null = null;
	export let askCom: CenterOfMass | null = null;
	export let vwap: number | null = null;
	export let lastTradePrice: number | null = null;
	export let trendSupport: number | null = null;
	export let trendResistance: number | null = null;
	export let supportHistory: number[] = [];
	export let resistanceHistory: number[] = [];
	export let supIntersectHistory: number[] = [];
	export let resIntersectHistory: number[] = [];
	export let lastIntersectHistory: number[] = [];
	export let candles: Array<{ open: number; high: number; low: number; close: number }> = [];
	export let currentCandle: { open: number; high: number; low: number; close: number } | null = null;
	export let bidDominance: number = 0.5; // 0-1, where 0.5 is balanced (for vertical ratio line)
	export let mirrored = false; // Horizontally mirror the entire chart
	export let width = 800;
	export let height = 500;

	let svg: SVGSVGElement;
	let mounted = false;

	// Stable price range tracking - smoothly adjusts instead of jumping
	let stableMinPrice: number | null = null;
	let stableMaxPrice: number | null = null;
	const smoothingFactor = 0.05; // How fast to adjust (lower = more stable)

	const margin = { top: 45, right: 60, bottom: 40, left: 60 };
	const innerWidth = width - margin.left - margin.right;
	const innerHeight = height - margin.top - margin.bottom;

	// Divide into eighths: outer eighths for bars, inner six eighths for tension field
	const eighthWidth = innerWidth / 8;
	const sideWidth = eighthWidth; // Each side bar area is 1/8 of width
	const centerWidth = eighthWidth * 6; // Center tension area is 6/8 of width
	const maxBarWidth = eighthWidth; // Bars can use full eighth

	// Force reactivity on all data props
	$: if (mounted && svg) {
		// Reference all props to trigger re-render when they change
		bids;
		asks;
		quantileLines;
		tradeDensity;
		bidCom;
		askCom;
		vwap;
		lastTradePrice;
		trendSupport;
		trendResistance;
		supportHistory;
		resistanceHistory;
		supIntersectHistory;
		resIntersectHistory;
		lastIntersectHistory;
		candles;
		currentCandle;
		bidDominance;
		mirrored;
		render();
	}

	onMount(() => {
		mounted = true;
	});

	function render() {
		if (!svg) return;

		const root = d3.select(svg);
		root.selectAll('*').remove();

		const g = root
			.append('g')
			.attr('transform', `translate(${margin.left}, ${margin.top})`);

		// Calculate price range
		const allPrices = [
			...bids.map((b) => b.price),
			...asks.map((a) => a.price)
		];

		if (allPrices.length === 0) {
			g.append('text')
				.attr('x', innerWidth / 2)
				.attr('y', innerHeight / 2)
				.attr('text-anchor', 'middle')
				.attr('fill', '#888')
				.text('Waiting for data...');
			return;
		}

		const currentMin = Math.min(...allPrices);
		const currentMax = Math.max(...allPrices);

		// Smoothly adjust stable price range
		if (stableMinPrice === null || stableMaxPrice === null) {
			// Initialize on first render
			stableMinPrice = currentMin;
			stableMaxPrice = currentMax;
		} else {
			// Expand immediately if price goes outside range, contract slowly
			if (currentMin < stableMinPrice) {
				stableMinPrice = currentMin; // Expand immediately
			} else {
				stableMinPrice = stableMinPrice + (currentMin - stableMinPrice) * smoothingFactor;
			}

			if (currentMax > stableMaxPrice) {
				stableMaxPrice = currentMax; // Expand immediately
			} else {
				stableMaxPrice = stableMaxPrice + (currentMax - stableMaxPrice) * smoothingFactor;
			}
		}

		const minPrice = stableMinPrice;
		const maxPrice = stableMaxPrice;
		const priceRange = maxPrice - minPrice || 1;

		// Scales
		const yScale = d3
			.scaleLinear()
			.domain([minPrice - priceRange * 0.05, maxPrice + priceRange * 0.05])
			.range([innerHeight, 0]);

		const maxBidQty = Math.max(...bids.map((b) => b.quantity), 1);
		const maxAskQty = Math.max(...asks.map((a) => a.quantity), 1);
		const maxQty = Math.max(maxBidQty, maxAskQty);

		// Bid bars grow from right edge of left quarter toward left
		// Ask bars grow from left edge of right quarter toward right
		const bidXScale = d3.scaleLinear().domain([0, maxQty]).range([sideWidth, 0]);
		const askXScale = d3.scaleLinear().domain([0, maxQty]).range([0, sideWidth]);

		// Background
		g.append('rect')
			.attr('width', innerWidth)
			.attr('height', innerHeight)
			.attr('fill', '#0a0a0f');

		// Create defs for gradients and clip paths
		const defs = g.append('defs');

		// Divider lines at tension field boundaries (1/8 and 7/8)
		[1, 7].forEach((i) => {
			g.append('line')
				.attr('x1', eighthWidth * i)
				.attr('y1', 0)
				.attr('x2', eighthWidth * i)
				.attr('y2', innerHeight)
				.attr('stroke', '#333')
				.attr('stroke-width', 1);
		});

		// Draw bid bars (left side)
		const bidGroup = g.append('g').attr('class', 'bids');

		bidGroup
			.selectAll('rect')
			.data(bids)
			.join('rect')
			.attr('x', (d) => bidXScale(d.quantity))
			.attr('y', (d) => yScale(d.price) - 2)
			.attr('width', (d) => sideWidth - bidXScale(d.quantity))
			.attr('height', 4)
			.attr('fill', '#22c55e')
			.attr('opacity', 0.5);

		// Draw ask bars (right side)
		const askGroup = g
			.append('g')
			.attr('class', 'asks')
			.attr('transform', `translate(${sideWidth + centerWidth}, 0)`);

		askGroup
			.selectAll('rect')
			.data(asks)
			.join('rect')
			.attr('x', 0)
			.attr('y', (d) => yScale(d.price) - 2)
			.attr('width', (d) => askXScale(d.quantity))
			.attr('height', 4)
			.attr('fill', '#ef4444')
			.attr('opacity', 0.5);

		// Draw trade histogram centered in the tension field
		// This is a simple histogram of the last 100 trade prices (30 bins)
		const tradeGroup = g
			.append('g')
			.attr('class', 'trades')
			.attr('transform', `translate(${sideWidth}, 0)`);

		if (tradeDensity.length > 0) {
			// Max bar width scales with density (densities sum to 1)
			// So if one price has 50% of trades, it gets 50% of maxBarWidth
			const maxBarWidth = centerWidth * 0.8;

			// Bar height - thin bars at each unique price level
			const barH = 4;

			// Draw histogram bars at each unique price
			for (const td of tradeDensity) {
				const y = yScale(td.price);
				// Skip if outside visible range
				if (y < 0 || y > innerHeight) continue;

				// Width is proportional to density (percentage of trades at this price)
				const barWidth = td.density * maxBarWidth;

				tradeGroup
					.append('rect')
					.attr('x', centerWidth / 2 - barWidth / 2)
					.attr('y', y - barH / 2)
					.attr('width', Math.max(barWidth, 2))
					.attr('height', barH)
					.attr('fill', '#f59e0b')
					.attr('opacity', 0.8)
					.attr('rx', 1);
			}
		}

		// Draw last trade price line (where the market is right now)
		if (lastTradePrice && lastTradePrice >= minPrice && lastTradePrice <= maxPrice) {
			g.append('line')
				.attr('x1', 0)
				.attr('y1', yScale(lastTradePrice))
				.attr('x2', innerWidth)
				.attr('y2', yScale(lastTradePrice))
				.attr('stroke', '#fff')
				.attr('stroke-width', 2);

			g.append('text')
				.attr('x', innerWidth + 5)
				.attr('y', yScale(lastTradePrice) + 4)
				.attr('fill', '#fff')
				.attr('font-size', '10px')
				.text('LAST');
		}

		// Draw support/resistance history histogram (on left side of chart)
		const srHistogramWidth = 30;
		const srGroup = g.append('g').attr('class', 'sr-histogram');

		// Count occurrences at each price level for support
		if (supportHistory.length > 0) {
			const supportCounts = new Map<number, number>();
			for (const price of supportHistory) {
				const rounded = Math.round(price * 100) / 100; // Round to cents
				supportCounts.set(rounded, (supportCounts.get(rounded) || 0) + 1);
			}
			const maxSupportCount = Math.max(...supportCounts.values());

			for (const [price, count] of supportCounts) {
				if (price >= minPrice && price <= maxPrice) {
					const barWidth = (count / maxSupportCount) * srHistogramWidth;
					srGroup
						.append('rect')
						.attr('x', -barWidth - 5)
						.attr('y', yScale(price) - 2)
						.attr('width', barWidth)
						.attr('height', 4)
						.attr('fill', '#22c55e')
						.attr('opacity', 0.6);
				}
			}
		}

		// Count occurrences at each price level for resistance
		if (resistanceHistory.length > 0) {
			const resistanceCounts = new Map<number, number>();
			for (const price of resistanceHistory) {
				const rounded = Math.round(price * 100) / 100;
				resistanceCounts.set(rounded, (resistanceCounts.get(rounded) || 0) + 1);
			}
			const maxResistanceCount = Math.max(...resistanceCounts.values());

			for (const [price, count] of resistanceCounts) {
				if (price >= minPrice && price <= maxPrice) {
					const barWidth = (count / maxResistanceCount) * srHistogramWidth;
					srGroup
						.append('rect')
						.attr('x', innerWidth + 5)
						.attr('y', yScale(price) - 2)
						.attr('width', barWidth)
						.attr('height', 4)
						.attr('fill', '#ef4444')
						.attr('opacity', 0.6);
				}
			}
		}

		// Draw current trend support line
		if (trendSupport && trendSupport >= minPrice && trendSupport <= maxPrice) {
			g.append('line')
				.attr('x1', 0)
				.attr('y1', yScale(trendSupport))
				.attr('x2', innerWidth)
				.attr('y2', yScale(trendSupport))
				.attr('stroke', '#22c55e')
				.attr('stroke-width', 2)
				.attr('stroke-dasharray', '8,4');

			g.append('text')
				.attr('x', -40)
				.attr('y', yScale(trendSupport) + 4)
				.attr('fill', '#22c55e')
				.attr('font-size', '10px')
				.text('SUP');
		}

		// Draw current trend resistance line
		if (trendResistance && trendResistance >= minPrice && trendResistance <= maxPrice) {
			g.append('line')
				.attr('x1', 0)
				.attr('y1', yScale(trendResistance))
				.attr('x2', innerWidth)
				.attr('y2', yScale(trendResistance))
				.attr('stroke', '#ef4444')
				.attr('stroke-width', 2)
				.attr('stroke-dasharray', '8,4');

			g.append('text')
				.attr('x', innerWidth + 40)
				.attr('y', yScale(trendResistance) + 4)
				.attr('fill', '#ef4444')
				.attr('font-size', '10px')
				.text('RES');
		}

		// Draw pressure corner indicators in tension field
		const cornerRadius = 240;
		const cornerGroup = g.append('g').attr('class', 'pressure-corners');

		// Top left corner - UP PRESSURE (green)
		const upPath = d3.path();
		upPath.moveTo(sideWidth, 0);
		upPath.lineTo(sideWidth + cornerRadius, 0);
		upPath.arc(sideWidth + cornerRadius, cornerRadius, cornerRadius, -Math.PI / 2, Math.PI, true);
		upPath.lineTo(sideWidth, 0);
		upPath.closePath();

		cornerGroup
			.append('path')
			.attr('d', upPath.toString())
			.attr('fill', '#22c55e')
			.attr('opacity', 0.3);

		cornerGroup
			.append('text')
			.attr('x', sideWidth + 25)
			.attr('y', 45)
			.attr('fill', '#22c55e')
			.attr('font-size', '22px')
			.attr('opacity', 0.7)
			.text('UP');

		cornerGroup
			.append('text')
			.attr('x', sideWidth + 25)
			.attr('y', 72)
			.attr('fill', '#22c55e')
			.attr('font-size', '22px')
			.attr('opacity', 0.7)
			.text('PRESSURE');

		// Bottom right corner - DOWN PRESSURE (red)
		const downPath = d3.path();
		downPath.moveTo(sideWidth + centerWidth, innerHeight);
		downPath.lineTo(sideWidth + centerWidth - cornerRadius, innerHeight);
		downPath.arc(sideWidth + centerWidth - cornerRadius, innerHeight - cornerRadius, cornerRadius, Math.PI / 2, 0, true);
		downPath.lineTo(sideWidth + centerWidth, innerHeight);
		downPath.closePath();

		cornerGroup
			.append('path')
			.attr('d', downPath.toString())
			.attr('fill', '#ef4444')
			.attr('opacity', 0.3);

		cornerGroup
			.append('text')
			.attr('x', sideWidth + centerWidth - 25)
			.attr('y', innerHeight - 52)
			.attr('text-anchor', 'end')
			.attr('fill', '#ef4444')
			.attr('font-size', '22px')
			.attr('opacity', 0.7)
			.text('DOWN');

		cornerGroup
			.append('text')
			.attr('x', sideWidth + centerWidth - 25)
			.attr('y', innerHeight - 25)
			.attr('text-anchor', 'end')
			.attr('fill', '#ef4444')
			.attr('font-size', '22px')
			.attr('opacity', 0.7)
			.text('PRESSURE');

		// Draw quantile density as shaded area (Gaussian-like intensity)
		const densityGroup = g.append('g').attr('class', 'quantile-density');

		// Sort quantile lines by percentile
		const sortedLines = [...quantileLines]
			.filter((q) => q.bidPrice >= minPrice && q.askPrice <= maxPrice)
			.sort((a, b) => a.percentile - b.percentile);

		// Create filled bands between adjacent quantile lines
		for (let i = 0; i < sortedLines.length - 1; i++) {
			const lower = sortedLines[i];
			const upper = sortedLines[i + 1];

			// Calculate density/opacity based on distance from median (50%)
			// Highest density near 50%, lowest near 0% and 100%
			const midPercentile = (lower.percentile + upper.percentile) / 2;
			const distFromMedian = Math.abs(midPercentile - 50) / 50; // 0 at median, 1 at edges
			const density = 1 - distFromMedian; // 1 at median, 0 at edges

			// Higher contrast: near-white glow at center, very dim at edges
			const opacity = 0.05 + density * density * 0.9; // Quadratic falloff, range 0.05 to 0.95

			// Interpolate colors toward white at high density
			const centerColor = density > 0.7
				? d3.interpolateRgb('#888888', '#ffffff')(density)
				: '#888888';
			const bidColor = density > 0.5
				? d3.interpolateRgb('#22c55e', '#aaffaa')(density)
				: '#22c55e';
			const askColor = density > 0.5
				? d3.interpolateRgb('#ef4444', '#ffaaaa')(density)
				: '#ef4444';

			// Create gradient for this band (green on left, white center, red on right)
			const gradientId = `density-grad-${i}`;
			const gradient = defs
				.append('linearGradient')
				.attr('id', gradientId)
				.attr('x1', '0%')
				.attr('x2', '100%');

			gradient.append('stop').attr('offset', '0%').attr('stop-color', bidColor).attr('stop-opacity', opacity);
			gradient.append('stop').attr('offset', '50%').attr('stop-color', centerColor).attr('stop-opacity', opacity);
			gradient.append('stop').attr('offset', '100%').attr('stop-color', askColor).attr('stop-opacity', opacity);

			// Draw filled polygon between the two quantile lines
			const path = d3.path();
			path.moveTo(sideWidth, yScale(lower.bidPrice));
			path.lineTo(sideWidth + centerWidth, yScale(lower.askPrice));
			path.lineTo(sideWidth + centerWidth, yScale(upper.askPrice));
			path.lineTo(sideWidth, yScale(upper.bidPrice));
			path.closePath();

			densityGroup
				.append('path')
				.attr('d', path.toString())
				.attr('fill', `url(#${gradientId})`);
		}

		// Extend density to edges of chart (beyond the tension field)
		// Left side (bid) - extend from leftmost edge to sideWidth
		if (sortedLines.length > 0) {
			for (let i = 0; i < sortedLines.length - 1; i++) {
				const lower = sortedLines[i];
				const upper = sortedLines[i + 1];
				const midPercentile = (lower.percentile + upper.percentile) / 2;
				const distFromMedian = Math.abs(midPercentile - 50) / 50;
				const density = 1 - distFromMedian;
				const opacity = 0.02 + density * density * 0.4; // Quadratic falloff for edges too

				// Brighten colors toward white at high density
				const bidEdgeColor = density > 0.5
					? d3.interpolateRgb('#22c55e', '#aaffaa')(density)
					: '#22c55e';
				const askEdgeColor = density > 0.5
					? d3.interpolateRgb('#ef4444', '#ffaaaa')(density)
					: '#ef4444';

				// Bid side extension
				densityGroup
					.append('rect')
					.attr('x', 0)
					.attr('y', Math.min(yScale(lower.bidPrice), yScale(upper.bidPrice)))
					.attr('width', sideWidth)
					.attr('height', Math.abs(yScale(upper.bidPrice) - yScale(lower.bidPrice)))
					.attr('fill', bidEdgeColor)
					.attr('opacity', opacity);

				// Ask side extension
				densityGroup
					.append('rect')
					.attr('x', sideWidth + centerWidth)
					.attr('y', Math.min(yScale(lower.askPrice), yScale(upper.askPrice)))
					.attr('width', sideWidth)
					.attr('height', Math.abs(yScale(upper.askPrice) - yScale(lower.askPrice)))
					.attr('fill', askEdgeColor)
					.attr('opacity', opacity);
			}
		}

		// Draw subtle quantile boundary lines for reference
		const lineGroup = g.append('g').attr('class', 'quantile-lines');
		for (const qline of sortedLines) {
			if (qline.percentile % 20 === 0 && qline.percentile !== 0 && qline.percentile !== 100) {
				const y1 = yScale(qline.bidPrice);
				const y2 = yScale(qline.askPrice);

				lineGroup
					.append('line')
					.attr('x1', sideWidth)
					.attr('y1', y1)
					.attr('x2', sideWidth + centerWidth)
					.attr('y2', y2)
					.attr('stroke', '#ffffff')
					.attr('stroke-width', 0.5)
					.attr('opacity', 0.3);

				// Label
				lineGroup
					.append('text')
					.attr('x', sideWidth + centerWidth / 2)
					.attr('y', (y1 + y2) / 2 - 5)
					.attr('text-anchor', 'middle')
					.attr('fill', '#ffffff')
					.attr('font-size', '9px')
					.attr('opacity', 0.5)
					.text(`${qline.percentile}%`);
			}
		}

		// Draw COM markers
		if (bidCom && bidCom.price >= minPrice && bidCom.price <= maxPrice) {
			g.append('circle')
				.attr('cx', sideWidth - 10)
				.attr('cy', yScale(bidCom.price))
				.attr('r', 6)
				.attr('fill', '#22c55e')
				.attr('stroke', '#fff')
				.attr('stroke-width', 2);

			g.append('text')
				.attr('x', sideWidth - 20)
				.attr('y', yScale(bidCom.price) + 4)
				.attr('text-anchor', 'end')
				.attr('fill', '#22c55e')
				.attr('font-size', '10px')
				.attr('font-weight', 'bold')
				.text('BID COM');
		}

		if (askCom && askCom.price >= minPrice && askCom.price <= maxPrice) {
			g.append('circle')
				.attr('cx', sideWidth + centerWidth + 10)
				.attr('cy', yScale(askCom.price))
				.attr('r', 6)
				.attr('fill', '#ef4444')
				.attr('stroke', '#fff')
				.attr('stroke-width', 2);

			g.append('text')
				.attr('x', sideWidth + centerWidth + 20)
				.attr('y', yScale(askCom.price) + 4)
				.attr('text-anchor', 'start')
				.attr('fill', '#ef4444')
				.attr('font-size', '10px')
				.attr('font-weight', 'bold')
				.text('ASK COM');
		}

		// Draw tension line between COMs
		if (bidCom && askCom) {
			const comX1 = sideWidth - 10;
			const comY1 = yScale(bidCom.price);
			const comX2 = sideWidth + centerWidth + 10;
			const comY2 = yScale(askCom.price);

			g.append('line')
				.attr('x1', comX1)
				.attr('y1', comY1)
				.attr('x2', comX2)
				.attr('y2', comY2)
				.attr('stroke', '#fff')
				.attr('stroke-width', 3)
				.attr('opacity', 0.8);

			// Helper function to find x where horizontal line at price intersects COM line
			const findIntersectionX = (price: number): number | null => {
				const y = yScale(price);
				// Check if y is within the COM line's y range
				const yMin = Math.min(comY1, comY2);
				const yMax = Math.max(comY1, comY2);
				if (y < yMin || y > yMax) return null;

				// Linear interpolation: find t where y = comY1 + t * (comY2 - comY1)
				if (comY2 === comY1) return (comX1 + comX2) / 2; // Horizontal line
				const t = (y - comY1) / (comY2 - comY1);
				return comX1 + t * (comX2 - comX1);
			};

			// Draw vertical line where SUP intersects COM line
			if (trendSupport && trendSupport >= minPrice && trendSupport <= maxPrice) {
				const supX = findIntersectionX(trendSupport);
				if (supX !== null) {
					g.append('line')
						.attr('x1', supX)
						.attr('y1', 0)
						.attr('x2', supX)
						.attr('y2', innerHeight)
						.attr('stroke', '#22c55e')
						.attr('stroke-width', 1)
						.attr('stroke-dasharray', '4,4')
						.attr('opacity', 0.6);
				}
			}

			// Draw vertical line where RES intersects COM line
			if (trendResistance && trendResistance >= minPrice && trendResistance <= maxPrice) {
				const resX = findIntersectionX(trendResistance);
				if (resX !== null) {
					g.append('line')
						.attr('x1', resX)
						.attr('y1', 0)
						.attr('x2', resX)
						.attr('y2', innerHeight)
						.attr('stroke', '#ef4444')
						.attr('stroke-width', 1)
						.attr('stroke-dasharray', '4,4')
						.attr('opacity', 0.6);
				}
			}

			// Draw vertical line where LAST price intersects COM line
			if (lastTradePrice && lastTradePrice >= minPrice && lastTradePrice <= maxPrice) {
				const lastX = findIntersectionX(lastTradePrice);
				if (lastX !== null) {
					g.append('line')
						.attr('x1', lastX)
						.attr('y1', 0)
						.attr('x2', lastX)
						.attr('y2', innerHeight)
						.attr('stroke', '#ffffff')
						.attr('stroke-width', 1.5)
						.attr('stroke-dasharray', '2,2')
						.attr('opacity', 0.8);
				}
			}
		}

		// Draw vertical line position histograms
		const histogramHeight = 25;
		const histogramBins = 30;

		// Helper to create histogram from normalized positions (0-1)
		const createHistogram = (positions: number[], numBins: number): number[] => {
			const bins = new Array(numBins).fill(0);
			for (const pos of positions) {
				// Clamp to 0-1 range
				const clamped = Math.max(0, Math.min(1, pos));
				const binIndex = Math.min(numBins - 1, Math.floor(clamped * numBins));
				bins[binIndex]++;
			}
			return bins;
		};

		// Top histogram - Support vertical line positions (green)
		if (supIntersectHistory.length > 0) {
			const supBins = createHistogram(supIntersectHistory, histogramBins);
			const maxSupCount = Math.max(...supBins, 1);
			const binWidth = centerWidth / histogramBins;

			const supHistGroup = g.append('g')
				.attr('class', 'sup-histogram')
				.attr('transform', `translate(${sideWidth}, 0)`);

			supBins.forEach((count, i) => {
				if (count > 0) {
					const barHeight = (count / maxSupCount) * histogramHeight;
					supHistGroup
						.append('rect')
						.attr('x', i * binWidth)
						.attr('y', -barHeight)
						.attr('width', binWidth - 1)
						.attr('height', barHeight)
						.attr('fill', '#22c55e')
						.attr('opacity', 0.6);
				}
			});
		}

		// Bottom histogram - Resistance vertical line positions (red)
		if (resIntersectHistory.length > 0) {
			const resBins = createHistogram(resIntersectHistory, histogramBins);
			const maxResCount = Math.max(...resBins, 1);
			const binWidth = centerWidth / histogramBins;

			const resHistGroup = g.append('g')
				.attr('class', 'res-histogram')
				.attr('transform', `translate(${sideWidth}, ${innerHeight})`);

			resBins.forEach((count, i) => {
				if (count > 0) {
					const barHeight = (count / maxResCount) * histogramHeight;
					resHistGroup
						.append('rect')
						.attr('x', i * binWidth)
						.attr('y', 0)
						.attr('width', binWidth - 1)
						.attr('height', barHeight)
						.attr('fill', '#ef4444')
						.attr('opacity', 0.6);
				}
			});
		}

		// Center histogram - Last price vertical line positions (white), centered on LAST line
		if (lastIntersectHistory.length > 0 && lastTradePrice && lastTradePrice >= minPrice && lastTradePrice <= maxPrice) {
			const lastBins = createHistogram(lastIntersectHistory, histogramBins);
			const maxLastCount = Math.max(...lastBins, 1);
			const binWidth = centerWidth / histogramBins;
			const lastY = yScale(lastTradePrice);
			const halfHistHeight = histogramHeight / 2;

			const lastHistGroup = g.append('g')
				.attr('class', 'last-histogram')
				.attr('transform', `translate(${sideWidth}, ${lastY})`);

			lastBins.forEach((count, i) => {
				if (count > 0) {
					const barHeight = (count / maxLastCount) * halfHistHeight;
					// Draw bars extending both up and down from center
					lastHistGroup
						.append('rect')
						.attr('x', i * binWidth)
						.attr('y', -barHeight)
						.attr('width', binWidth - 1)
						.attr('height', barHeight * 2)
						.attr('fill', '#ffffff')
						.attr('opacity', 0.4);
				}
			});
		}

		// Draw 30-second candlesticks on top in tension field
		const allCandles = [...candles];
		if (currentCandle) allCandles.push(currentCandle);

		if (allCandles.length > 0) {
			// Create clip path for tension field area
			const clipId = 'candle-clip';
			defs.append('clipPath')
				.attr('id', clipId)
				.append('rect')
				.attr('x', 0)
				.attr('y', 0)
				.attr('width', centerWidth)
				.attr('height', innerHeight);

			const candleGroup = g.append('g')
				.attr('class', 'candles-top')
				.attr('transform', `translate(${sideWidth}, 0)`)
				.attr('clip-path', `url(#${clipId})`);

			// Show last N candles that fit in the center area
			const candleWidth = 12;
			const candleGap = 4;
			const maxCandles = Math.floor(centerWidth / (candleWidth + candleGap));
			const visibleCandles = allCandles.slice(-maxCandles);

			// Position candles from right to left (newest on right)
			visibleCandles.forEach((candle, i) => {
				const x = centerWidth - (visibleCandles.length - i) * (candleWidth + candleGap);
				const isBullish = candle.close >= candle.open;
				// Blue/purple colors: purple for bullish, blue for bearish
				const color = isBullish ? '#a855f7' : '#6366f1';

				// Clamp Y values to visible range
				const highY = yScale(Math.min(Math.max(candle.high, minPrice), maxPrice));
				const lowY = yScale(Math.min(Math.max(candle.low, minPrice), maxPrice));
				const openY = yScale(Math.min(Math.max(candle.open, minPrice), maxPrice));
				const closeY = yScale(Math.min(Math.max(candle.close, minPrice), maxPrice));

				const bodyTop = Math.min(openY, closeY);
				const bodyHeight = Math.max(2, Math.abs(closeY - openY));

				// Wick (high-low line)
				candleGroup
					.append('line')
					.attr('x1', x + candleWidth / 2)
					.attr('y1', highY)
					.attr('x2', x + candleWidth / 2)
					.attr('y2', lowY)
					.attr('stroke', color)
					.attr('stroke-width', 2)
					.attr('opacity', 0.8);

				// Body (open-close rectangle)
				candleGroup
					.append('rect')
					.attr('x', x)
					.attr('y', bodyTop)
					.attr('width', candleWidth)
					.attr('height', bodyHeight)
					.attr('fill', isBullish ? color : '#0a0a0f')
					.attr('stroke', color)
					.attr('stroke-width', 2)
					.attr('opacity', 0.8);
			});
		}

		// Draw vertical BEARS/BULLS ratio line
		// bidDominance: 0 = all bears, 0.5 = balanced, 1 = all bulls
		// Map to x position within the tension field (sideWidth to sideWidth + centerWidth)
		const ratioX = sideWidth + (bidDominance * centerWidth);
		const bearsPct = Math.round((1 - bidDominance) * 100);
		const bullsPct = Math.round(bidDominance * 100);

		const ratioGroup = g.append('g').attr('class', 'ratio-line');

		// Main vertical ratio line
		ratioGroup
			.append('line')
			.attr('x1', ratioX)
			.attr('y1', 0)
			.attr('x2', ratioX)
			.attr('y2', innerHeight)
			.attr('stroke', bidDominance > 0.5 ? '#22c55e' : bidDominance < 0.5 ? '#ef4444' : '#888')
			.attr('stroke-width', 3)
			.attr('opacity', 0.8);

		// Top triangle marker
		const triangleSize = 10;
		ratioGroup
			.append('path')
			.attr('d', `M${ratioX},0 L${ratioX - triangleSize},${-triangleSize * 1.5} L${ratioX + triangleSize},${-triangleSize * 1.5} Z`)
			.attr('fill', bidDominance > 0.5 ? '#22c55e' : bidDominance < 0.5 ? '#ef4444' : '#888');

		// Bottom triangle marker
		ratioGroup
			.append('path')
			.attr('d', `M${ratioX},${innerHeight} L${ratioX - triangleSize},${innerHeight + triangleSize * 1.5} L${ratioX + triangleSize},${innerHeight + triangleSize * 1.5} Z`)
			.attr('fill', bidDominance > 0.5 ? '#22c55e' : bidDominance < 0.5 ? '#ef4444' : '#888');

		// Scale bar at the top of the tension field
		const scaleY = -25;
		const scaleHeight = 8;

		// Background scale bar
		ratioGroup
			.append('rect')
			.attr('x', sideWidth)
			.attr('y', scaleY)
			.attr('width', centerWidth)
			.attr('height', scaleHeight)
			.attr('fill', '#1a1a24')
			.attr('stroke', '#333')
			.attr('rx', 2);

		// Bulls portion (green, from center to right based on bulls %)
		const centerX = sideWidth + centerWidth / 2;
		const bullsWidth = (bidDominance - 0.5) * centerWidth;
		if (bullsWidth > 0) {
			ratioGroup
				.append('rect')
				.attr('x', centerX)
				.attr('y', scaleY)
				.attr('width', bullsWidth)
				.attr('height', scaleHeight)
				.attr('fill', '#22c55e')
				.attr('opacity', 0.7);
		}

		// Bears portion (red, from center to left based on bears %)
		const bearsWidth = (0.5 - bidDominance) * centerWidth;
		if (bearsWidth > 0) {
			ratioGroup
				.append('rect')
				.attr('x', centerX - bearsWidth)
				.attr('y', scaleY)
				.attr('width', bearsWidth)
				.attr('height', scaleHeight)
				.attr('fill', '#ef4444')
				.attr('opacity', 0.7);
		}

		// Center marker (50% line)
		ratioGroup
			.append('line')
			.attr('x1', centerX)
			.attr('y1', scaleY)
			.attr('x2', centerX)
			.attr('y2', scaleY + scaleHeight)
			.attr('stroke', '#fff')
			.attr('stroke-width', 2);

		// Labels
		ratioGroup
			.append('text')
			.attr('x', sideWidth + 5)
			.attr('y', scaleY - 3)
			.attr('fill', '#ef4444')
			.attr('font-size', '10px')
			.attr('font-weight', 'bold')
			.text(`BEARS ${bearsPct}%`);

		ratioGroup
			.append('text')
			.attr('x', sideWidth + centerWidth - 5)
			.attr('y', scaleY - 3)
			.attr('text-anchor', 'end')
			.attr('fill', '#22c55e')
			.attr('font-size', '10px')
			.attr('font-weight', 'bold')
			.text(`${bullsPct}% BULLS`);

		// Y-axis (price)
		const yAxis = d3.axisLeft(yScale).ticks(10).tickFormat(d3.format('$.2f'));

		g.append('g').attr('class', 'y-axis').call(yAxis).selectAll('text').attr('fill', '#888');

		g.selectAll('.y-axis path, .y-axis line').attr('stroke', '#333');

		// Legend - interpretation guide (small text at borders)
		const legendGroup = g.append('g').attr('class', 'legend');
		const legendFontSize = '8px';
		const legendOpacity = 0.5;

		// Bottom left - BID COM interpretation
		legendGroup
			.append('text')
			.attr('x', 5)
			.attr('y', innerHeight + 25)
			.attr('fill', '#22c55e')
			.attr('font-size', legendFontSize)
			.attr('opacity', legendOpacity)
			.text('← LAST here = Buyers market');

		// Bottom right - ASK COM interpretation
		legendGroup
			.append('text')
			.attr('x', innerWidth - 5)
			.attr('y', innerHeight + 25)
			.attr('text-anchor', 'end')
			.attr('fill', '#ef4444')
			.attr('font-size', legendFontSize)
			.attr('opacity', legendOpacity)
			.text('LAST here = Sellers market →');

		// Left side - SUP interpretation
		legendGroup
			.append('text')
			.attr('x', -50)
			.attr('y', innerHeight / 2 - 20)
			.attr('fill', '#22c55e')
			.attr('font-size', legendFontSize)
			.attr('opacity', legendOpacity)
			.attr('transform', `rotate(-90, -50, ${innerHeight / 2 - 20})`)
			.text('SUP left = strong floor');

		// Right side - RES interpretation
		legendGroup
			.append('text')
			.attr('x', innerWidth + 50)
			.attr('y', innerHeight / 2 + 20)
			.attr('fill', '#ef4444')
			.attr('font-size', legendFontSize)
			.attr('opacity', legendOpacity)
			.attr('transform', `rotate(90, ${innerWidth + 50}, ${innerHeight / 2 + 20})`)
			.text('RES right = strong ceiling');

		// Labels - centered in each section
		g.append('text')
			.attr('x', sideWidth / 2)
			.attr('y', -5)
			.attr('text-anchor', 'middle')
			.attr('fill', '#22c55e')
			.attr('font-size', '10px')
			.attr('font-weight', 'bold')
			.text('BID');

		g.append('text')
			.attr('x', sideWidth + centerWidth + sideWidth / 2)
			.attr('y', -5)
			.attr('text-anchor', 'middle')
			.attr('fill', '#ef4444')
			.attr('font-size', '10px')
			.attr('font-weight', 'bold')
			.text('ASK');

		g.append('text')
			.attr('x', sideWidth + centerWidth / 2)
			.attr('y', -5)
			.attr('text-anchor', 'middle')
			.attr('fill', '#888')
			.attr('font-size', '12px')
			.attr('font-weight', 'bold')
			.text('TENSION FIELD');
	}
</script>

<svg bind:this={svg} {width} {height} class="tug-of-war" class:mirrored />

<style>
	.tug-of-war {
		background: #0a0a0f;
		border-radius: 8px;
	}

	.tug-of-war.mirrored {
		transform: scaleX(-1);
	}

	.tug-of-war.mirrored :global(text) {
		transform: scaleX(-1);
		transform-box: fill-box;
		transform-origin: center;
	}
</style>
