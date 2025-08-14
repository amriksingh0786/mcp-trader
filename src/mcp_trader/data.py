import asyncio
import os
from datetime import datetime, timedelta
from typing import Any

import aiohttp
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

try:
    from fastmcp import FastMCP

    USE_FASTMCP = True
except ImportError:
    USE_FASTMCP = False

load_dotenv()

# Initialize FastMCP if available
if USE_FASTMCP:
    mcp = FastMCP("mcp-trader-resources")


def normalize_indian_symbol(symbol: str, exchange: str = "NSE") -> str:
    """
    Normalize Indian stock symbol to proper format.
    
    Args:
        symbol: Stock symbol (e.g., 'RELIANCE', 'RELIANCE.NS')
        exchange: Exchange code ('NSE' or 'BSE')
    
    Returns:
        Normalized symbol (e.g., 'RELIANCE.NS')
    """
    symbol_upper = symbol.upper()
    
    # If already has exchange suffix, return as-is
    if symbol_upper.endswith('.NS') or symbol_upper.endswith('.BO'):
        return symbol_upper
    
    # Add appropriate suffix based on exchange
    suffix = '.NS' if exchange.upper() == 'NSE' else '.BO'
    return f"{symbol_upper}{suffix}"


def is_indian_symbol(symbol: str) -> bool:
    """
    Check if symbol is an Indian stock symbol.
    
    Args:
        symbol: Stock symbol to check
    
    Returns:
        True if symbol appears to be Indian stock
    """
    symbol_upper = symbol.upper()
    return symbol_upper.endswith('.NS') or symbol_upper.endswith('.BO')


class MarketData:
    """Handles all market data fetching operations."""

    def __init__(self):
        self.api_key = os.getenv("TIINGO_API_KEY")
        # Don't require API key if using other providers
        self.has_tiingo = bool(self.api_key)
        
        if self.has_tiingo:
            self.headers = {
                "Content-Type": "application/json",
                "Authorization": f"Token {self.api_key}",
            }

    async def get_crypto_historical_data(
        self,
        symbol: str,
        lookback_days: int = 365,
        provider: str = "tiingo",
        quote_currency: str = "usd",
    ) -> pd.DataFrame:
        """
        Fetch historical daily data for a given crypto asset.

        Args:
            symbol (str): The crypto symbol (e.g., 'BTC', 'ETH', or 'BTCUSDT' for Binance).
            lookback_days (int): Number of days to look back from today.
            provider (str): 'tiingo' or 'binance'.
            quote_currency (str): Quote currency (default 'usd' for Tiingo, 'USDT' for Binance).

        Returns:
            pd.DataFrame: DataFrame containing historical crypto market data.

        Raises:
            ValueError: If the symbol is invalid or no data is returned.
            Exception: For other unexpected issues during the fetch operation.
        """
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)

        if provider.lower() == "tiingo":
            # Tiingo expects symbols like 'btcusd'
            pair = f"{symbol.lower()}{quote_currency.lower()}"
            url = (
                f"https://api.tiingo.com/tiingo/crypto/prices?"
                f"tickers={pair}&"
                f"startDate={start_date.strftime('%Y-%m-%d')}&"
                f"endDate={end_date.strftime('%Y-%m-%d')}&"
                f"resampleFreq=1day"
            )
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as session:
                    async with session.get(url, headers=self.headers) as response:
                        if response.status == 404:
                            raise ValueError(f"Crypto symbol not found: {symbol}")
                        response.raise_for_status()
                        data = await response.json()

                if not data or not data[0].get("priceData"):
                    raise ValueError(f"No data returned for {symbol} on Tiingo")

                df = pd.DataFrame(data[0]["priceData"])
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                df["open"] = df["open"].astype(float)
                df["high"] = df["high"].astype(float)
                df["low"] = df["low"].astype(float)
                df["close"] = df["close"].astype(float)
                df["volume"] = df["volume"].astype(float)
                df["symbol"] = pair.upper()
                return df

            except aiohttp.ClientError as e:
                raise ConnectionError(
                    f"Network error while fetching crypto data for {symbol} (Tiingo): {e}"
                ) from e
            except ValueError as ve:
                raise ve
            except Exception as e:
                raise Exception(
                    f"Unexpected error fetching crypto data for {symbol} (Tiingo): {e}"
                ) from e

        elif provider.lower() == "binance":
            # Binance expects symbols like 'BTCUSDT'
            binance_symbol = symbol.upper()
            interval = "1d"
            limit = min(lookback_days, 1000)  # Binance max 1000
            url = (
                f"https://api.binance.com/api/v3/klines?"
                f"symbol={binance_symbol}&interval={interval}&limit={limit}"
            )
            try:
                async with aiohttp.ClientSession(
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as session:
                    async with session.get(url) as response:
                        if response.status == 404:
                            raise ValueError(f"Crypto symbol not found: {symbol}")
                        response.raise_for_status()
                        data = await response.json()

                if not data:
                    raise ValueError(f"No data returned for {symbol} on Binance")

                # Binance kline columns:
                # 0: open time, 1: open, 2: high, 3: low, 4: close, 5: volume, 6: close time, ...
                df = pd.DataFrame(
                    data,
                    columns=[
                        "open_time",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "close_time",
                        "quote_asset_volume",
                        "number_of_trades",
                        "taker_buy_base_asset_volume",
                        "taker_buy_quote_asset_volume",
                        "ignore",
                    ],
                )
                df["date"] = pd.to_datetime(df["open_time"], unit="ms")
                df.set_index("date", inplace=True)
                df["open"] = df["open"].astype(float)
                df["high"] = df["high"].astype(float)
                df["low"] = df["low"].astype(float)
                df["close"] = df["close"].astype(float)
                df["volume"] = df["volume"].astype(float)
                df["symbol"] = binance_symbol
                return df[["open", "high", "low", "close", "volume", "symbol"]]

            except aiohttp.ClientError as e:
                raise ConnectionError(
                    f"Network error while fetching crypto data for {symbol} (Binance): {e}"
                ) from e
            except ValueError as ve:
                raise ve
            except Exception as e:
                raise Exception(
                    f"Unexpected error fetching crypto data for {symbol} (Binance): {e}"
                ) from e

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def get_historical_data(
        self, 
        symbol: str, 
        lookback_days: int = 365, 
        provider: str = "tiingo", 
        market: str = "us"
    ) -> pd.DataFrame:
        """
        Fetch historical daily data for a given symbol.

        Args:
            symbol (str): The stock symbol to fetch data for.
            lookback_days (int): Number of days to look back from today.
            provider (str): Data provider ('tiingo' or 'yfinance').
            market (str): Market region ('us' or 'in' for India).

        Returns:
            pd.DataFrame: DataFrame containing historical market data.

        Raises:
            ValueError: If the symbol is invalid or no data is returned.
            Exception: For other unexpected issues during the fetch operation.
        """
        # Auto-detect market and provider based on symbol
        if is_indian_symbol(symbol) or market.lower() == "in":
            provider = "yfinance"  # Force yfinance for Indian stocks
            market = "in"
            symbol = normalize_indian_symbol(symbol)
        
        if provider.lower() == "tiingo":
            return await self._fetch_tiingo_data(symbol, lookback_days)
        elif provider.lower() == "yfinance":
            return await self._fetch_yfinance_data(symbol, lookback_days)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    async def _fetch_tiingo_data(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        """
        Fetch data from Tiingo API.
        """
        if not self.has_tiingo:
            raise ValueError("TIINGO_API_KEY not found in environment")
            
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)

        url = (
            f"https://api.tiingo.com/tiingo/daily/{symbol}/prices?"
            f"startDate={start_date.strftime('%Y-%m-%d')}&"
            f"endDate={end_date.strftime('%Y-%m-%d')}"
        )

        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                async with session.get(url, headers=self.headers) as response:
                    if response.status == 404:
                        raise ValueError(f"Symbol not found: {symbol}")
                    response.raise_for_status()
                    data = await response.json()

            if not data:
                raise ValueError(f"No data returned for {symbol}")

            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

            df[["open", "high", "low", "close"]] = df[
                ["adjOpen", "adjHigh", "adjLow", "adjClose"]
            ].round(2)
            df["volume"] = df["adjVolume"].astype(int)
            df["symbol"] = symbol.upper()

            return df

        except aiohttp.ClientError as e:
            raise ConnectionError(f"Network error while fetching data for {symbol} (Tiingo): {e}") from e
        except ValueError as ve:
            raise ve
        except Exception as e:
            raise Exception(f"Unexpected error fetching data for {symbol} (Tiingo): {e}") from e
    
    async def _fetch_yfinance_data(self, symbol: str, lookback_days: int) -> pd.DataFrame:
        """
        Fetch data from Yahoo Finance via yfinance library.
        """
        try:
            def fetch_data():
                """
                Synchronous data fetch function to run in executor.
                """
                ticker = yf.Ticker(symbol)
                # Calculate period string for yfinance
                if lookback_days <= 7:
                    period = "7d"
                elif lookback_days <= 30:
                    period = "1mo"
                elif lookback_days <= 90:
                    period = "3mo"
                elif lookback_days <= 180:
                    period = "6mo"
                elif lookback_days <= 365:
                    period = "1y"
                elif lookback_days <= 730:
                    period = "2y"
                elif lookback_days <= 1825:
                    period = "5y"
                else:
                    period = "max"
                
                # Fetch historical data
                data = ticker.history(period=period)
                return data
            
            # Run synchronous yfinance call in executor
            loop = asyncio.get_event_loop()
            df = await loop.run_in_executor(None, fetch_data)
            
            if df.empty:
                raise ValueError(f"No data returned for {symbol}")
            
            # Standardize column names to match Tiingo format
            df.columns = df.columns.str.lower()
            df = df.rename(columns={
                'adj close': 'close'  # Use adjusted close as close
            })
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Round prices to 2 decimal places
            df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].round(2)
            df['volume'] = df['volume'].astype(int)
            df['symbol'] = symbol.upper()
            
            # Ensure index is named 'date'
            df.index.name = 'date'
            
            return df[['open', 'high', 'low', 'close', 'volume', 'symbol']]
            
        except Exception as e:
            if "No data found" in str(e) or "No timezone found" in str(e):
                raise ValueError(f"Symbol not found or invalid: {symbol}")
            raise Exception(f"Unexpected error fetching data for {symbol} (yfinance): {e}") from e


# FastMCP Resources Implementation
if USE_FASTMCP:
    # Create a global instance for the resources to use
    _market_data = MarketData()

    # Cache for resource data with TTL
    _resource_cache: dict[str, dict[str, Any]] = {}
    _cache_ttl = 300  # 5 minutes cache TTL

    def _is_cache_valid(cache_key: str) -> bool:
        """Check if cache entry is still valid."""
        if cache_key not in _resource_cache:
            return False
        cached = _resource_cache[cache_key]
        return (datetime.now() - cached["timestamp"]).total_seconds() < _cache_ttl

    @mcp.resource("stock://{symbol}")
    async def get_stock_price(symbol: str, market: str = "us") -> dict[str, Any]:
        """
        Get current and recent stock price data.

        Returns the latest price, daily change, and key statistics for a stock symbol.
        Supports both US and Indian markets.
        """
        cache_key = f"stock:{symbol}:{market}"

        # Check cache first
        if _is_cache_valid(cache_key):
            return _resource_cache[cache_key]["data"]

        try:
            # Auto-detect provider and normalize symbol
            provider = "yfinance" if is_indian_symbol(symbol) or market.lower() == "in" else "tiingo"
            if market.lower() == "in" or is_indian_symbol(symbol):
                symbol = normalize_indian_symbol(symbol)
                market = "in"
            
            # Fetch last 5 days of data
            df = await _market_data.get_historical_data(
                symbol, lookback_days=5, provider=provider, market=market
            )

            latest = df.iloc[-1]
            prev_close = df.iloc[-2]["close"] if len(df) > 1 else latest["close"]

            data = {
                "symbol": symbol.upper(),
                "market": market,
                "price": float(latest["close"]),
                "open": float(latest["open"]),
                "high": float(latest["high"]),
                "low": float(latest["low"]),
                "volume": int(latest["volume"]),
                "change": float(latest["close"] - prev_close),
                "change_percent": float((latest["close"] - prev_close) / prev_close * 100),
                "timestamp": datetime.now().isoformat(),
                "provider": provider,
            }

            # Cache the result
            _resource_cache[cache_key] = {"data": data, "timestamp": datetime.now()}

            return data

        except Exception as e:
            return {"error": str(e), "symbol": symbol, "market": market, "timestamp": datetime.now().isoformat()}

    @mcp.resource("stock://{symbol}/history")
    async def get_stock_history(symbol: str, days: int | None = 30, market: str = "us") -> dict[str, Any]:
        """
        Get historical stock price data.

        Returns OHLCV data for the specified number of days.
        Supports both US and Indian markets.
        """
        cache_key = f"stock:{symbol}:history:{days}:{market}"

        # Check cache first
        if _is_cache_valid(cache_key):
            return _resource_cache[cache_key]["data"]

        try:
            # Auto-detect provider and normalize symbol
            provider = "yfinance" if is_indian_symbol(symbol) or market.lower() == "in" else "tiingo"
            if market.lower() == "in" or is_indian_symbol(symbol):
                symbol = normalize_indian_symbol(symbol)
                market = "in"
            
            df = await _market_data.get_historical_data(
                symbol, lookback_days=days, provider=provider, market=market
            )

            # Convert DataFrame to list of dictionaries
            history = []
            for date, row in df.iterrows():
                history.append(
                    {
                        "date": date.isoformat(),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": int(row["volume"]),
                    }
                )

            data = {
                "symbol": symbol.upper(),
                "market": market,
                "days": days,
                "data": history,
                "timestamp": datetime.now().isoformat(),
                "provider": provider,
            }

            # Cache the result
            _resource_cache[cache_key] = {"data": data, "timestamp": datetime.now()}

            return data

        except Exception as e:
            return {"error": str(e), "symbol": symbol, "market": market, "timestamp": datetime.now().isoformat()}

    @mcp.resource("crypto://{symbol}")
    async def get_crypto_price(
        symbol: str, provider: str = "tiingo", quote: str = "usd"
    ) -> dict[str, Any]:
        """
        Get current and recent cryptocurrency price data.

        Supports both Tiingo and Binance providers.
        """
        cache_key = f"crypto:{symbol}:{provider}:{quote}"

        # Check cache first
        if _is_cache_valid(cache_key):
            return _resource_cache[cache_key]["data"]

        try:
            # Fetch last 5 days of data
            df = await _market_data.get_crypto_historical_data(
                symbol, lookback_days=5, provider=provider, quote_currency=quote
            )

            latest = df.iloc[-1]
            prev_close = df.iloc[-2]["close"] if len(df) > 1 else latest["close"]

            data = {
                "symbol": symbol.upper(),
                "quote_currency": quote.upper(),
                "price": float(latest["close"]),
                "open": float(latest["open"]),
                "high": float(latest["high"]),
                "low": float(latest["low"]),
                "volume": float(latest["volume"]),
                "change": float(latest["close"] - prev_close),
                "change_percent": float((latest["close"] - prev_close) / prev_close * 100),
                "timestamp": datetime.now().isoformat(),
                "provider": provider,
            }

            # Cache the result
            _resource_cache[cache_key] = {"data": data, "timestamp": datetime.now()}

            return data

        except Exception as e:
            return {
                "error": str(e),
                "symbol": symbol,
                "provider": provider,
                "timestamp": datetime.now().isoformat(),
            }

    @mcp.resource("crypto://{symbol}/history")
    async def get_crypto_history(
        symbol: str, days: int | None = 30, provider: str = "tiingo", quote: str = "usd"
    ) -> dict[str, Any]:
        """
        Get historical cryptocurrency price data.

        Returns OHLCV data for the specified number of days.
        """
        cache_key = f"crypto:{symbol}:history:{days}:{provider}:{quote}"

        # Check cache first
        if _is_cache_valid(cache_key):
            return _resource_cache[cache_key]["data"]

        try:
            df = await _market_data.get_crypto_historical_data(
                symbol, lookback_days=days, provider=provider, quote_currency=quote
            )

            # Convert DataFrame to list of dictionaries
            history = []
            for date, row in df.iterrows():
                history.append(
                    {
                        "date": date.isoformat(),
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                    }
                )

            data = {
                "symbol": symbol.upper(),
                "quote_currency": quote.upper(),
                "days": days,
                "data": history,
                "timestamp": datetime.now().isoformat(),
                "provider": provider,
            }

            # Cache the result
            _resource_cache[cache_key] = {"data": data, "timestamp": datetime.now()}

            return data

        except Exception as e:
            return {
                "error": str(e),
                "symbol": symbol,
                "provider": provider,
                "timestamp": datetime.now().isoformat(),
            }

    @mcp.resource("market://cache/clear")
    async def clear_cache() -> dict[str, Any]:
        """
        Clear the resource cache.

        Useful for forcing fresh data retrieval.
        """
        _resource_cache.clear()
        return {
            "status": "success",
            "message": "Cache cleared",
            "timestamp": datetime.now().isoformat(),
        }

    @mcp.resource("indian-stock://{symbol}")
    async def get_indian_stock_price(symbol: str, exchange: str = "NSE") -> dict[str, Any]:
        """
        Get current and recent Indian stock price data.
        
        Returns the latest price, daily change, and key statistics for an Indian stock symbol.
        Automatically formats symbol with appropriate exchange suffix (.NS or .BO).
        """
        normalized_symbol = normalize_indian_symbol(symbol, exchange)
        cache_key = f"indian-stock:{normalized_symbol}"
        
        # Check cache first
        if _is_cache_valid(cache_key):
            return _resource_cache[cache_key]["data"]
        
        try:
            # Fetch last 5 days of data
            df = await _market_data.get_historical_data(
                normalized_symbol, lookback_days=5, provider="yfinance", market="in"
            )
            
            latest = df.iloc[-1]
            prev_close = df.iloc[-2]["close"] if len(df) > 1 else latest["close"]
            
            data = {
                "symbol": normalized_symbol,
                "exchange": exchange.upper(),
                "market": "in",
                "price": float(latest["close"]),
                "open": float(latest["open"]),
                "high": float(latest["high"]),
                "low": float(latest["low"]),
                "volume": int(latest["volume"]),
                "change": float(latest["close"] - prev_close),
                "change_percent": float((latest["close"] - prev_close) / prev_close * 100),
                "timestamp": datetime.now().isoformat(),
                "provider": "yfinance",
            }
            
            # Cache the result
            _resource_cache[cache_key] = {"data": data, "timestamp": datetime.now()}
            
            return data
            
        except Exception as e:
            return {
                "error": str(e), 
                "symbol": normalized_symbol, 
                "exchange": exchange,
                "timestamp": datetime.now().isoformat()
            }

    @mcp.resource("market://cache/status")
    async def cache_status() -> dict[str, Any]:
        """
        Get current cache status and statistics.
        """
        entries = []
        for key, value in _resource_cache.items():
            age = (datetime.now() - value["timestamp"]).total_seconds()
            entries.append({"key": key, "age_seconds": age, "expired": age >= _cache_ttl})

        return {
            "total_entries": len(_resource_cache),
            "ttl_seconds": _cache_ttl,
            "entries": entries,
            "timestamp": datetime.now().isoformat(),
        }
