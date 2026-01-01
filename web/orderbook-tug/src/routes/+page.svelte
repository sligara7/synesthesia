<script lang="ts">
	import { onMount, onDestroy } from 'svelte';
	import TugOfWar from '$lib/components/TugOfWar.svelte';
	import {
		binanceStore,
		orderBook,
		bidCom,
		askCom,
		tensionField,
		tradeDensity,
		vwap,
		bidDominance,
		currentSpread,
		connected,
		patternMatches,
		tensionHistory
	} from '$lib/stores/binanceStore';

	let symbol = 'BTCUSDT';
	let symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT'];

	onMount(() => {
		binanceStore.connect(symbol);
	});

	onDestroy(() => {
		binanceStore.disconnect();
	});

	function handleSymbolChange() {
		binanceStore.connect(symbol);
	}

	function formatPrice(price: number | null | undefined): string {
		if (price === null || price === undefined) return '-';
		return `$${price.toFixed(2)}`;
	}

	function formatPercent(value: number): string {
		return `${(value * 100).toFixed(1)}%`;
	}

	$: dominanceClass =
		$bidDominance > 0.55 ? 'bulls' : $bidDominance < 0.45 ? 'bears' : 'neutral';
	$: dominanceLabel =
		$bidDominance > 0.55 ? 'BULLS' : $bidDominance < 0.45 ? 'BEARS' : 'BALANCED';

	// Debug: count non-zero trade density bins
	$: nonZeroBins = $tradeDensity.filter(d => d.density > 0);
	$: tradeDebug = nonZeroBins.length > 0
		? `${nonZeroBins.length} bins, prices: ${nonZeroBins.slice(0,3).map(d => d.price.toFixed(0)).join(', ')}...`
		: 'no data';
</script>

<svelte:head>
	<title>Order Book Tug of War - {symbol}</title>
</svelte:head>

<div class="container">
	<header class="header">
		<div class="title">
			<h1>Order Book Tug of War</h1>
			<span class="subtitle">Real-time order book analysis</span>
		</div>

		<div class="controls">
			<select bind:value={symbol} on:change={handleSymbolChange}>
				{#each symbols as s}
					<option value={s}>{s}</option>
				{/each}
			</select>

			<span class="connection-status" class:connected={$connected}>
				{$connected ? 'LIVE' : 'DISCONNECTED'}
			</span>
		</div>
	</header>

	<div class="stats-row">
		<div class="stat-card">
			<div class="stat-value {dominanceClass}">
				{formatPercent($bidDominance)}
			</div>
			<div class="stat-label">{dominanceLabel}</div>
		</div>

		<div class="stat-card">
			<div class="stat-value">
				{formatPrice($bidCom?.price)}
			</div>
			<div class="stat-label bulls">BID COM</div>
		</div>

		<div class="stat-card">
			<div class="stat-value">
				{formatPrice($askCom?.price)}
			</div>
			<div class="stat-label bears">ASK COM</div>
		</div>

		<div class="stat-card">
			<div class="stat-value">
				{formatPrice($vwap)}
			</div>
			<div class="stat-label" style="color: #a855f7">VWAP</div>
		</div>

		<div class="stat-card">
			<div class="stat-value">
				${$currentSpread.toFixed(4)}
			</div>
			<div class="stat-label">SPREAD</div>
		</div>

		<div class="stat-card">
			<div class="stat-value">
				{$patternMatches.length}
			</div>
			<div class="stat-label" style="color: #3b82f6">PATTERNS</div>
		</div>

		<div class="stat-card">
			<div class="stat-value" style="font-size: 0.8rem">
				{tradeDebug}
			</div>
			<div class="stat-label" style="color: #f59e0b">TRADES</div>
		</div>
	</div>

	<div class="main-viz">
		<TugOfWar
			bids={$orderBook?.bids || []}
			asks={$orderBook?.asks || []}
			quantileLines={$tensionField?.lines || []}
			tradeDensity={$tradeDensity}
			bidCom={$bidCom}
			askCom={$askCom}
			vwap={$vwap}
			width={900}
			height={500}
		/>
	</div>

	{#if $tensionField}
		<div class="panel">
			<div class="panel-header">Tension Field Analysis</div>
			<div class="grid-3">
				<div class="stat">
					<div class="stat-value">{$tensionField.shape.toUpperCase()}</div>
					<div class="stat-label">Shape</div>
				</div>
				<div class="stat">
					<div class="stat-value">{$tensionField.convergenceRatio.toFixed(2)}</div>
					<div class="stat-label">Convergence</div>
				</div>
				<div class="stat">
					<div class="stat-value">
						{($tensionField.bidTotal / 1000).toFixed(1)}K / {($tensionField.askTotal / 1000).toFixed(1)}K
					</div>
					<div class="stat-label">Bid / Ask Volume</div>
				</div>
			</div>
		</div>
	{/if}

	{#if $patternMatches.length > 0}
		<div class="panel">
			<div class="panel-header">Pattern Matches</div>
			<div class="pattern-list">
				{#each $patternMatches.slice(0, 5) as match}
					<div class="pattern-item">
						<span class="pattern-score">{(match.similarityScore * 100).toFixed(0)}%</span>
						<span class="pattern-tick">Tick #{match.historicalTick}</span>
						<span class="pattern-angle">{match.angleDifference.toFixed(1)}° diff</span>
					</div>
				{/each}
			</div>
		</div>
	{/if}
</div>

<style>
	.header {
		display: flex;
		justify-content: space-between;
		align-items: center;
		padding: 1rem 0;
		border-bottom: 1px solid var(--border-color);
		margin-bottom: 1rem;
	}

	.title h1 {
		font-size: 1.5rem;
		font-weight: 700;
		margin: 0;
	}

	.subtitle {
		font-size: 0.875rem;
		color: var(--text-secondary);
	}

	.controls {
		display: flex;
		align-items: center;
		gap: 1rem;
	}

	select {
		background: var(--bg-card);
		border: 1px solid var(--border-color);
		color: var(--text-primary);
		padding: 0.5rem 1rem;
		border-radius: 4px;
		font-family: inherit;
		cursor: pointer;
	}

	.connection-status {
		padding: 0.25rem 0.75rem;
		border-radius: 4px;
		font-size: 0.75rem;
		font-weight: 600;
		background: var(--ask-color-dim);
		color: var(--ask-color);
	}

	.connection-status.connected {
		background: var(--bid-color-dim);
		color: var(--bid-color);
	}

	.stats-row {
		display: flex;
		gap: 1rem;
		margin-bottom: 1rem;
		flex-wrap: wrap;
	}

	.stat-card {
		flex: 1;
		min-width: 120px;
		background: var(--bg-panel);
		border: 1px solid var(--border-color);
		border-radius: 8px;
		padding: 1rem;
		text-align: center;
	}

	.stat-card .stat-value {
		font-size: 1.25rem;
		font-weight: 700;
	}

	.stat-card .stat-label {
		font-size: 0.75rem;
		text-transform: uppercase;
		color: var(--text-secondary);
		margin-top: 0.25rem;
	}

	.main-viz {
		background: var(--bg-panel);
		border: 1px solid var(--border-color);
		border-radius: 8px;
		padding: 1rem;
		margin-bottom: 1rem;
		display: flex;
		justify-content: center;
	}

	.pattern-list {
		display: flex;
		flex-direction: column;
		gap: 0.5rem;
	}

	.pattern-item {
		display: flex;
		align-items: center;
		gap: 1rem;
		padding: 0.5rem;
		background: var(--bg-card);
		border-radius: 4px;
	}

	.pattern-score {
		font-weight: 700;
		color: var(--pattern-match);
		min-width: 50px;
	}

	.pattern-tick {
		color: var(--text-secondary);
	}

	.pattern-angle {
		color: var(--text-muted);
		font-size: 0.875rem;
	}
</style>
