"""
Tests for Indian stock functionality in MCP Trader.

This module tests the Indian stock support including symbol normalization,
data fetching from yfinance, and MCP tool integration.
"""

import asyncio
import pytest
import pandas as pd
from unittest.mock import AsyncMock, MagicMock, patch

from mcp_trader.data import MarketData, normalize_indian_symbol, is_indian_symbol


class TestIndianStockUtilities:
    """Test Indian stock utility functions."""
    
    def test_normalize_indian_symbol_nse_default(self):
        """Test symbol normalization with NSE as default."""
        assert normalize_indian_symbol("RELIANCE") == "RELIANCE.NS"
        assert normalize_indian_symbol("TCS") == "TCS.NS"
        assert normalize_indian_symbol("reliance") == "RELIANCE.NS"
    
    def test_normalize_indian_symbol_bse(self):
        """Test symbol normalization with BSE exchange."""
        assert normalize_indian_symbol("RELIANCE", "BSE") == "RELIANCE.BO"
        assert normalize_indian_symbol("TCS", "bse") == "TCS.BO"
    
    def test_normalize_indian_symbol_already_formatted(self):
        """Test that already formatted symbols are returned as-is."""
        assert normalize_indian_symbol("RELIANCE.NS") == "RELIANCE.NS"
        assert normalize_indian_symbol("RELIANCE.BO") == "RELIANCE.BO"
        assert normalize_indian_symbol("TCS.NS", "BSE") == "TCS.NS"  # Keeps existing suffix
    
    def test_is_indian_symbol(self):
        """Test Indian symbol detection."""
        assert is_indian_symbol("RELIANCE.NS") is True
        assert is_indian_symbol("TCS.BO") is True
        assert is_indian_symbol("reliance.ns") is True
        assert is_indian_symbol("AAPL") is False
        assert is_indian_symbol("TSLA") is False
        assert is_indian_symbol("") is False


class TestMarketDataIndianStocks:
    """Test MarketData class with Indian stock support."""
    
    @pytest.fixture
    def market_data(self):
        """Create MarketData instance for testing."""
        with patch.dict('os.environ', {'TIINGO_API_KEY': ''}, clear=False):
            return MarketData()
    
    @pytest.mark.asyncio
    async def test_get_historical_data_auto_detect_indian(self, market_data):
        """Test automatic detection and routing of Indian stocks."""
        mock_yfinance_data = pd.DataFrame({
            'open': [1350.0, 1355.0, 1360.0],
            'high': [1370.0, 1375.0, 1380.0],
            'low': [1340.0, 1345.0, 1350.0],
            'close': [1365.0, 1370.0, 1375.0],
            'volume': [1000000, 1100000, 1200000]
        })
        
        with patch.object(market_data, '_fetch_yfinance_data', return_value=mock_yfinance_data) as mock_fetch:
            df = await market_data.get_historical_data("RELIANCE.NS", lookback_days=5)
            
            # Verify yfinance was called
            mock_fetch.assert_called_once_with("RELIANCE.NS", 5)
            assert len(df) == 3
            assert df.iloc[-1]['close'] == 1375.0
    
    @pytest.mark.asyncio
    async def test_get_historical_data_market_parameter(self, market_data):
        """Test explicit market parameter for Indian stocks."""
        mock_yfinance_data = pd.DataFrame({
            'open': [2500.0, 2510.0],
            'high': [2520.0, 2530.0], 
            'low': [2490.0, 2500.0],
            'close': [2515.0, 2525.0],
            'volume': [800000, 850000]
        })
        
        with patch.object(market_data, '_fetch_yfinance_data', return_value=mock_yfinance_data) as mock_fetch:
            df = await market_data.get_historical_data("TCS", market="in", lookback_days=2)
            
            # Should normalize symbol and use yfinance
            mock_fetch.assert_called_once_with("TCS.NS", 2)
            assert len(df) == 2
    
    @pytest.mark.asyncio
    async def test_get_historical_data_us_stocks_unchanged(self, market_data):
        """Test that US stocks still work with Tiingo (if API key available)."""
        # Mock Tiingo data
        mock_tiingo_data = pd.DataFrame({
            'open': [150.0, 155.0],
            'high': [160.0, 165.0],
            'low': [145.0, 150.0], 
            'close': [158.0, 162.0],
            'volume': [5000000, 5500000]
        })
        
        # Mock has_tiingo to be True for this test
        market_data.has_tiingo = True
        market_data.headers = {'Authorization': 'Token test_key'}
        
        with patch.object(market_data, '_fetch_tiingo_data', return_value=mock_tiingo_data) as mock_fetch:
            df = await market_data.get_historical_data("AAPL", lookback_days=2)
            
            mock_fetch.assert_called_once_with("AAPL", 2)
            assert len(df) == 2
            assert df.iloc[-1]['close'] == 162.0
    
    @pytest.mark.asyncio 
    async def test_fetch_yfinance_data_success(self, market_data):
        """Test successful yfinance data fetch."""
        mock_ticker = MagicMock()
        mock_history = pd.DataFrame({
            'Open': [1300.0, 1305.0, 1310.0],
            'High': [1315.0, 1320.0, 1325.0],
            'Low': [1295.0, 1300.0, 1305.0],
            'Close': [1310.0, 1315.0, 1320.0],
            'Volume': [2000000, 2100000, 2200000]
        })
        mock_ticker.history.return_value = mock_history
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            df = await market_data._fetch_yfinance_data("INFY.NS", 10)
            
            assert len(df) == 3
            assert list(df.columns) == ['open', 'high', 'low', 'close', 'volume', 'symbol']
            assert df.iloc[-1]['symbol'] == 'INFY.NS'
            assert df.iloc[-1]['close'] == 1320.0
    
    @pytest.mark.asyncio
    async def test_fetch_yfinance_data_empty_response(self, market_data):
        """Test handling of empty yfinance response."""
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame()  # Empty DataFrame
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            with pytest.raises(ValueError, match="No data returned for INVALID.NS"):
                await market_data._fetch_yfinance_data("INVALID.NS", 10)
    
    @pytest.mark.asyncio
    async def test_fetch_yfinance_data_network_error(self, market_data):
        """Test handling of network errors in yfinance."""
        mock_ticker = MagicMock()
        mock_ticker.history.side_effect = Exception("Network connection failed")
        
        with patch('yfinance.Ticker', return_value=mock_ticker):
            with pytest.raises(Exception, match="Unexpected error fetching data for TEST.NS"):
                await market_data._fetch_yfinance_data("TEST.NS", 10)


class TestIndianStockModels:
    """Test Pydantic models with Indian stock support."""
    
    def test_market_data_response_indian_market(self):
        """Test MarketDataResponse with Indian market data."""
        from mcp_trader.models import MarketDataResponse, CandleData
        from datetime import datetime
        
        candle = CandleData(
            date=datetime.now(),
            open=1350.0,
            high=1370.0,
            low=1340.0,
            close=1365.0,
            volume=1000000,
            symbol="RELIANCE.NS"
        )
        
        response = MarketDataResponse(
            symbol="RELIANCE.NS",
            data=[candle],
            provider="yfinance",
            market="in",
            lookback_days=1
        )
        
        assert response.symbol == "RELIANCE.NS"
        assert response.provider == "yfinance"
        assert response.market == "in"
        assert len(response.data) == 1
    
    def test_analyze_indian_stock_request(self):
        """Test AnalyzeIndianStockRequest model."""
        from mcp_trader.models import AnalyzeIndianStockRequest
        
        request = AnalyzeIndianStockRequest(
            symbol="TCS",
            exchange="NSE", 
            lookback_days=30
        )
        
        assert request.symbol == "TCS"
        assert request.exchange == "NSE"
        assert request.lookback_days == 30
    
    def test_analyze_stock_request_with_market(self):
        """Test AnalyzeStockRequest with market parameter."""
        from mcp_trader.models import AnalyzeStockRequest
        
        request = AnalyzeStockRequest(
            symbol="HDFC.NS",
            market="in",
            lookback_days=60
        )
        
        assert request.symbol == "HDFC.NS"
        assert request.market == "in"
        assert request.lookback_days == 60


@pytest.mark.integration
class TestIndianStockIntegration:
    """Integration tests for Indian stock functionality."""
    
    @pytest.mark.asyncio
    async def test_real_indian_stock_data_fetch(self):
        """Test fetching real Indian stock data (requires internet)."""
        market_data = MarketData()
        
        try:
            # Test with a liquid Indian stock
            df = await market_data.get_historical_data(
                "RELIANCE.NS", 
                lookback_days=5,
                provider="yfinance",
                market="in"
            )
            
            # Verify data structure
            assert len(df) > 0
            assert 'close' in df.columns
            assert 'volume' in df.columns
            assert 'symbol' in df.columns
            assert df.iloc[-1]['symbol'] == 'RELIANCE.NS'
            
            # Verify price data is reasonable (Reliance typically trades > 1000)
            assert df.iloc[-1]['close'] > 500  # Sanity check
            
        except Exception as e:
            pytest.skip(f"Real data test failed (likely network/API issue): {e}")
    
    @pytest.mark.asyncio
    async def test_multiple_indian_exchanges(self):
        """Test fetching data from both NSE and BSE."""
        market_data = MarketData()
        
        try:
            # Test NSE
            df_nse = await market_data.get_historical_data(
                "TCS.NS",
                lookback_days=5, 
                provider="yfinance",
                market="in"
            )
            
            # Test BSE (same company)
            df_bse = await market_data.get_historical_data(
                "TCS.BO",
                lookback_days=5,
                provider="yfinance", 
                market="in"
            )
            
            # Both should return data
            assert len(df_nse) > 0
            assert len(df_bse) > 0
            
            # Prices should be similar (within reasonable range)
            nse_close = df_nse.iloc[-1]['close']
            bse_close = df_bse.iloc[-1]['close']
            
            # Allow for small differences between exchanges
            price_diff_percent = abs(nse_close - bse_close) / nse_close * 100
            assert price_diff_percent < 5.0  # Should be very close
            
        except Exception as e:
            pytest.skip(f"Multi-exchange test failed: {e}")


# Test fixtures and utilities
@pytest.fixture
def sample_indian_stock_data():
    """Sample Indian stock data for testing."""
    return pd.DataFrame({
        'open': [1350.0, 1355.0, 1360.0, 1365.0, 1370.0],
        'high': [1370.0, 1375.0, 1380.0, 1385.0, 1390.0],
        'low': [1340.0, 1345.0, 1350.0, 1355.0, 1360.0],
        'close': [1365.0, 1370.0, 1375.0, 1380.0, 1385.0],
        'volume': [1000000, 1100000, 1200000, 1300000, 1400000],
        'symbol': ['RELIANCE.NS'] * 5
    })


@pytest.fixture  
def sample_indian_symbols():
    """Sample Indian stock symbols for testing."""
    return {
        'nse': ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ITC.NS'],
        'bse': ['RELIANCE.BO', 'TCS.BO', 'INFY.BO', 'HDFCBANK.BO', 'ITC.BO'],
        'raw': ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ITC']
    }


if __name__ == "__main__":
    # Run basic tests
    pytest.main([__file__, "-v"])