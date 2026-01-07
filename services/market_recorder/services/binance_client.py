"""
Binance WebSocket client for streaming market data.

Implements the CanStreamMarketData protocol for receiving real-time
order book depth and aggregated trade data from Binance.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import AsyncGenerator, Callable

import websockets
import orjson

from ..protocols import DepthUpdate, TradeUpdate
from ..config import get_settings

logger = logging.getLogger(__name__)


class BinanceWebSocketClient:
    """
    Async WebSocket client for Binance market data streams.

    Implements CanStreamMarketData protocol.

    Usage:
        client = BinanceWebSocketClient()

        async for update in client.stream_depth("BTCUSDT"):
            print(update)
    """

    def __init__(
        self,
        base_url: str | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ):
        """
        Initialize the Binance WebSocket client.

        Args:
            base_url: WebSocket base URL (defaults to config setting)
            on_error: Optional error callback
        """
        settings = get_settings()
        self.base_url = base_url or settings.binance_ws_url
        self.on_error = on_error
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def stream_depth(
        self,
        symbol: str,
        depth_levels: int = 20,
        update_speed_ms: int = 100,
    ) -> AsyncGenerator[DepthUpdate, None]:
        """
        Stream order book depth updates.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")
            depth_levels: Number of price levels (5, 10, or 20)
            update_speed_ms: Update frequency (100 or 1000)

        Yields:
            DepthUpdate objects with bids/asks
        """
        symbol_lower = symbol.lower()
        url = f"{self.base_url}/{symbol_lower}@depth{depth_levels}@{update_speed_ms}ms"

        logger.info(f"Connecting to depth stream: {url}")
        self._running = True

        while self._running:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    logger.info(f"Connected to depth stream for {symbol}")

                    while self._running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                            data = orjson.loads(msg)

                            # Parse bids and asks
                            bids = [
                                (float(price), float(qty))
                                for price, qty in data.get("bids", [])
                            ]
                            asks = [
                                (float(price), float(qty))
                                for price, qty in data.get("asks", [])
                            ]

                            yield DepthUpdate(
                                timestamp=datetime.now(timezone.utc),
                                symbol=symbol.upper(),
                                last_update_id=data.get("lastUpdateId", 0),
                                bids=bids,
                                asks=asks,
                            )

                        except asyncio.TimeoutError:
                            # Send ping to keep connection alive
                            logger.debug("Depth stream timeout, sending ping")
                            continue
                        except websockets.ConnectionClosed:
                            logger.warning("Depth stream connection closed, reconnecting...")
                            break

            except Exception as e:
                logger.error(f"Depth stream error: {e}")
                if self.on_error:
                    self.on_error(e)
                if self._running:
                    await asyncio.sleep(5)  # Wait before reconnecting

    async def stream_trades(
        self,
        symbol: str,
    ) -> AsyncGenerator[TradeUpdate, None]:
        """
        Stream aggregated trades.

        Args:
            symbol: Trading pair (e.g., "BTCUSDT")

        Yields:
            TradeUpdate objects
        """
        symbol_lower = symbol.lower()
        url = f"{self.base_url}/{symbol_lower}@aggTrade"

        logger.info(f"Connecting to trade stream: {url}")
        self._running = True

        while self._running:
            try:
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    logger.info(f"Connected to trade stream for {symbol}")

                    while self._running:
                        try:
                            msg = await asyncio.wait_for(ws.recv(), timeout=30.0)
                            data = orjson.loads(msg)

                            # Parse trade data
                            # T = trade timestamp in ms
                            # a = aggregate trade ID
                            # p = price
                            # q = quantity
                            # m = is buyer maker (true = sell aggressor)

                            yield TradeUpdate(
                                timestamp=datetime.fromtimestamp(
                                    data["T"] / 1000, tz=timezone.utc
                                ),
                                symbol=symbol.upper(),
                                trade_id=data["a"],
                                price=float(data["p"]),
                                quantity=float(data["q"]),
                                is_buyer_maker=data["m"],
                            )

                        except asyncio.TimeoutError:
                            logger.debug("Trade stream timeout, sending ping")
                            continue
                        except websockets.ConnectionClosed:
                            logger.warning("Trade stream connection closed, reconnecting...")
                            break

            except Exception as e:
                logger.error(f"Trade stream error: {e}")
                if self.on_error:
                    self.on_error(e)
                if self._running:
                    await asyncio.sleep(5)  # Wait before reconnecting

    def stop(self) -> None:
        """Stop all active streams."""
        logger.info("Stopping Binance WebSocket client")
        self._running = False

        # Cancel any running tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
        self._tasks.clear()

    @property
    def is_running(self) -> bool:
        """Check if client is running."""
        return self._running


async def demo():
    """Demo the Binance client."""
    client = BinanceWebSocketClient()

    print("Streaming BTCUSDT depth for 10 seconds...")

    count = 0
    async for update in client.stream_depth("BTCUSDT"):
        print(f"[{update.timestamp}] Best bid: {update.bids[0][0]:.2f}, Best ask: {update.asks[0][0]:.2f}")
        count += 1
        if count >= 100:  # ~10 seconds at 100ms
            break

    client.stop()
    print("Done!")


if __name__ == "__main__":
    asyncio.run(demo())
