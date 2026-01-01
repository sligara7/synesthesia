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
	export let width = 800;
	export let height = 500;

	let svg: SVGSVGElement;
	let mounted = false;

	const margin = { top: 20, right: 60, bottom: 40, left: 60 };
	const innerWidth = width - margin.left - margin.right;
	const innerHeight = height - margin.top - margin.bottom;

	// Divide into quarters: outer quarters for bars, inner two quarters for tension lines
	const quarterWidth = innerWidth / 4;
	const sideWidth = quarterWidth; // Each side bar area is 1/4 of width
	const centerWidth = quarterWidth * 2; // Center tension area is 2/4 (half) of width
	const maxBarWidth = quarterWidth; // Bars can use full quarter

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

		const minPrice = Math.min(...allPrices);
		const maxPrice = Math.max(...allPrices);
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

		// Quarter divider lines
		[1, 2, 3].forEach((i) => {
			g.append('line')
				.attr('x1', quarterWidth * i)
				.attr('y1', 0)
				.attr('x2', quarterWidth * i)
				.attr('y2', innerHeight)
				.attr('stroke', '#222')
				.attr('stroke-width', 1)
				.attr('stroke-dasharray', i === 2 ? 'none' : '4,4');
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

		// Draw quantile density as shaded area (Gaussian-like intensity)
		const densityGroup = g.append('g').attr('class', 'quantile-density');

		// Sort quantile lines by percentile
		const sortedLines = [...quantileLines]
			.filter((q) => q.bidPrice >= minPrice && q.askPrice <= maxPrice)
			.sort((a, b) => a.percentile - b.percentile);

		// Create gradient definitions for bid and ask sides
		const defs = g.append('defs');

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

		// Y-axis (price)
		const yAxis = d3.axisLeft(yScale).ticks(10).tickFormat(d3.format('$.2f'));

		g.append('g').attr('class', 'y-axis').call(yAxis).selectAll('text').attr('fill', '#888');

		g.selectAll('.y-axis path, .y-axis line').attr('stroke', '#333');

		// Labels - centered in each quarter
		g.append('text')
			.attr('x', quarterWidth / 2)
			.attr('y', -5)
			.attr('text-anchor', 'middle')
			.attr('fill', '#22c55e')
			.attr('font-size', '12px')
			.attr('font-weight', 'bold')
			.text('BID VOLUME');

		g.append('text')
			.attr('x', quarterWidth * 3.5)
			.attr('y', -5)
			.attr('text-anchor', 'middle')
			.attr('fill', '#ef4444')
			.attr('font-size', '12px')
			.attr('font-weight', 'bold')
			.text('ASK VOLUME');

		g.append('text')
			.attr('x', quarterWidth * 2)
			.attr('y', -5)
			.attr('text-anchor', 'middle')
			.attr('fill', '#888')
			.attr('font-size', '12px')
			.attr('font-weight', 'bold')
			.text('TENSION FIELD');
	}
</script>

<svg bind:this={svg} {width} {height} class="tug-of-war" />

<style>
	.tug-of-war {
		background: #0a0a0f;
		border-radius: 8px;
	}
</style>
