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

		// Draw current price line (price with most trades)
		if (tradeDensity.length > 0) {
			const maxDensity = Math.max(...tradeDensity.map(d => d.density));
			const currentPrice = tradeDensity.find(d => d.density === maxDensity)?.price;

			if (currentPrice && currentPrice >= minPrice && currentPrice <= maxPrice) {
				g.append('line')
					.attr('x1', 0)
					.attr('y1', yScale(currentPrice))
					.attr('x2', innerWidth)
					.attr('y2', yScale(currentPrice))
					.attr('stroke', '#f59e0b')
					.attr('stroke-width', 2)
					.attr('stroke-dasharray', '5,5');

				g.append('text')
					.attr('x', innerWidth + 5)
					.attr('y', yScale(currentPrice) + 4)
					.attr('fill', '#f59e0b')
					.attr('font-size', '10px')
					.text('ACTIVE');
			}
		}

		// Draw quantile lines connecting bid to ask
		const lineGroup = g.append('g').attr('class', 'quantile-lines');

		// Filter and sort quantile lines for processing
		const validLines = quantileLines
			.filter((q) => q.percentile !== 50)
			.filter((q) => q.bidPrice >= minPrice && q.askPrice <= maxPrice)
			.sort((a, b) => a.percentile - b.percentile);

		for (const qline of validLines) {
			const y1 = yScale(qline.bidPrice);
			const y2 = yScale(qline.askPrice);

			// Color gradient from green (low percentile) to red (high percentile)
			const t = qline.percentile / 100;
			const color =
				t < 0.5
					? d3.interpolateRgb('#22c55e', '#888888')(t * 2)
					: d3.interpolateRgb('#888888', '#ef4444')((t - 0.5) * 2);

			// Draw the quantile line
			lineGroup
				.append('line')
				.attr('x1', sideWidth)
				.attr('y1', y1)
				.attr('x2', sideWidth + centerWidth)
				.attr('y2', y2)
				.attr('stroke', color)
				.attr('stroke-width', 1.5)
				.attr('opacity', 0.6);

			// Percentile label at 20%, 40%, 60%, 80%
			if (qline.percentile % 20 === 0) {
				lineGroup
					.append('text')
					.attr('x', sideWidth + centerWidth / 2)
					.attr('y', (y1 + y2) / 2 - 8)
					.attr('text-anchor', 'middle')
					.attr('fill', color)
					.attr('font-size', '11px')
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
			g.append('line')
				.attr('x1', sideWidth - 10)
				.attr('y1', yScale(bidCom.price))
				.attr('x2', sideWidth + centerWidth + 10)
				.attr('y2', yScale(askCom.price))
				.attr('stroke', '#fff')
				.attr('stroke-width', 3)
				.attr('opacity', 0.8);
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
