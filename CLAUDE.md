# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MCP Trader is a Model Context Protocol server providing technical analysis tools for stocks, cryptocurrency, and **Indian stocks**. It uses FastMCP framework with decorator-based tools and async operations throughout. The system supports multiple data providers including Tiingo (US stocks/crypto), Binance (crypto), and Yahoo Finance (Indian stocks).

## Essential Commands

```bash
# Install dependencies
uv sync

# Run the server
uv run mcp-trader

# Run tests with coverage  
pytest --cov=src/mcp_trader --cov-report=term-missing

# Run Indian stock specific tests
pytest src/mcp_trader/tests/test_indian_stocks.py -v

# Debug with MCP Inspector
npx @modelcontextprotocol/inspector uv --directory . run mcp-trader

# Development mode
uv run python -m mcp_trader.server

# Test Indian stock functionality
uv run python -c "
import asyncio
from src.mcp_trader.data import MarketData
async def test(): 
    md = MarketData()
    df = await md.get_historical_data('RELIANCE.NS', provider='yfinance', market='in')
    print(f'Latest RELIANCE.NS price: ₹{df.iloc[-1][\"close\"]:.2f}')
asyncio.run(test())
"
```

## Code Architecture

### Core Components
- **server.py**: Main MCP server using FastMCP decorators (@mcp.tool()) with US and Indian stock tools
- **data.py**: Multi-provider async data fetching (Tiingo/Binance/Yahoo Finance) with Indian stock support
- **indicators.py**: Technical analysis calculations using pandas-ta
- **models.py**: Pydantic models with Indian market support
- **config.py**: Simple dataclass configuration management

### Key Design Patterns
- **FastMCP decorators**: All tools use @mcp.tool() with context-aware operations
- **Multi-provider support**: Auto-detection and routing based on symbol format and market parameter
- **Async/await**: All external API calls and I/O operations are async
- **Type safety**: Pydantic models validate all input/output data
- **Resource URIs**: Support for stock://, crypto://, indian-stock://, market:// resource access

### API Integration & Provider Selection
- **Tiingo API**: US stocks and crypto (requires TIINGO_API_KEY in .env)
- **Yahoo Finance (yfinance)**: Indian stocks NSE (.NS) and BSE (.BO) - no API key required
- **Binance API**: Extended crypto pairs, no API key required
- **Auto-detection**: Indian symbols (ending with .NS/.BO) automatically use yfinance
- **Environment**: TIINGO_API_KEY optional (system works with yfinance for Indian stocks)

## Development Guidelines

### Testing
- Target: 80% code coverage (configured in pytest.ini)
- Test files in src/mcp_trader/tests/ mirror source structure
- Use pytest-asyncio for async test functions
- Mock external API calls using pytest-mock

### Code Quality
- **Linting**: Ruff with pycodestyle, pyflakes, isort rules
- **Type checking**: MyPy with strict configuration
- **Line length**: 100 characters maximum
- **Python version**: 3.11+ required

### Adding New Features
1. Define Pydantic models in models.py for data structures (include market parameter for multi-region support)
2. Implement async data fetching in data.py with provider selection logic
3. Add analysis logic to indicators.py for technical calculations
4. Create MCP tool in server.py using @mcp.tool() decorator with market parameter support
5. Write comprehensive tests maintaining 80% coverage (see test_indian_stocks.py for examples)
6. Update type hints and ensure MyPy passes

### Indian Stock Development
- **Symbol formats**: Support both raw symbols (RELIANCE) and formatted (RELIANCE.NS/RELIANCE.BO)
- **Auto-normalization**: Use `normalize_indian_symbol()` utility function
- **Provider routing**: Indian symbols automatically use yfinance provider
- **Currency display**: Use ₹ for Indian stocks, $ for US stocks
- **Exchange support**: Both NSE (.NS) and BSE (.BO) exchanges
- **Benchmarks**: Use ^NSEI (Nifty 50) instead of SPY for Indian relative strength

### Error Handling
- Use context.show_progress() for long-running operations
- Log errors with context.report_error() in MCP tools
- Raise descriptive exceptions with context for debugging
- Handle API rate limits and network errors gracefully

## Dependencies

### Core Runtime
- fastmcp>=2.8.0 (MCP framework)
- pandas>=2.3.0 with pandas-ta>=0.3.14b (analysis)
- aiohttp>=3.12.13 (async HTTP for Tiingo/Binance)
- yfinance>=0.2.63 (Yahoo Finance for Indian stocks)
- numpy~=1.26.4 (locked to v1.x for compatibility)

### Optional
- ta-lib>=0.6.4: Enhanced performance (see INSTALL.md for setup)  
- redis>=6.2.0: Caching support

## Available MCP Tools

### Stock Analysis (Multi-region)
- `analyze_stock(symbol, market="us")`: Enhanced to support Indian stocks with market parameter
- `analyze_indian_stock(symbol, exchange="NSE")`: Dedicated Indian stock analysis tool
- `relative_strength(symbol, benchmark="SPY", market="us")`: Relative strength with region-specific benchmarks
- `volume_profile(symbol, lookback_days=60, market="us")`: Volume analysis for US and Indian stocks
- `detect_patterns(symbol, market="us")`: Pattern detection with regional support
- `position_size(symbol, stop_price, risk_amount, account_size, price=0, market="us")`: Position sizing
- `suggest_stops(symbol, market="us")`: Stop level suggestions

### Crypto Analysis  
- `analyze_crypto(symbol, provider="tiingo", lookback_days=365, quote_currency="usd")`: Crypto technical analysis

### Usage Examples
```python
# Indian stocks - all equivalent
analyze_stock("RELIANCE", market="in")      # Auto-normalized to RELIANCE.NS
analyze_stock("RELIANCE.NS")                # Auto-detected as Indian
analyze_indian_stock("RELIANCE", "NSE")     # Explicit Indian stock tool

# US stocks (existing functionality)
analyze_stock("AAPL")                       # Default US market

# Indian relative strength vs Nifty
relative_strength("TCS.NS", "^NSEI", "in")  # TCS vs Nifty 50
```