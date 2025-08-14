# Indian Stocks Implementation Plan

## 🎯 Objective
Add comprehensive Indian stock market support to MCP Trader using Yahoo Finance (yfinance) while maintaining existing architecture patterns.

## 📊 Research Summary

### Selected Solution: Yahoo Finance (yfinance)
- **Rationale**: Already in dependencies, supports NSE (.NS) and BSE (.BO), zero API keys needed
- **Coverage**: 10+ years historical data for Indian stocks
- **Integration**: Perfect pandas compatibility with existing codebase
- **Exchanges**: NSE (National Stock Exchange) and BSE (Bombay Stock Exchange)

### Alternative Options Evaluated
- **Zerodha Kite**: Professional but requires account setup
- **NSEPython**: Free but synchronous, limited data range
- **Alpha Vantage**: Good but requires API key

## 🏗️ Architecture Design

### Current Architecture Analysis
- **Method-based provider switching** (not class-based)
- **Single MarketData class** with conditional logic
- **Provider parameter** for crypto: `"tiingo"` vs `"binance"`
- **Unified DataFrame return format**: `[open, high, low, close, volume, symbol]`
- **FastMCP caching** with 5-minute TTL

### Integration Approach
Extend existing patterns by:
1. Adding `market` and `provider` parameters to `get_historical_data()`
2. Symbol normalization for Indian format (RELIANCE → RELIANCE.NS)
3. Provider routing logic similar to crypto implementation
4. Maintaining existing cache key structure

## 📝 Implementation Plan

### Phase 1: Core Data Layer (data.py)
- [ ] Add symbol normalization utility
- [ ] Extend `get_historical_data()` method signature
- [ ] Implement yfinance provider logic
- [ ] Add Indian market symbol validation
- [ ] Integrate with existing error handling

### Phase 2: Models & Config (models.py, config.py)
- [ ] Update MarketDataResponse model for Indian markets
- [ ] Add Indian market configuration options
- [ ] Update Pydantic models with new provider types

### Phase 3: MCP Tools (server.py)
- [ ] Add `analyze_indian_stock()` MCP tool
- [ ] Update cache key generation for Indian stocks
- [ ] Add FastMCP resources for Indian stocks
- [ ] Maintain existing tool patterns

### Phase 4: Testing & Documentation
- [ ] Write comprehensive tests for Indian stock functionality
- [ ] Mock yfinance responses for testing
- [ ] Update CLAUDE.md with Indian stock commands
- [ ] Add usage examples and documentation

## 🔧 Technical Implementation Details

### Symbol Format Handling
```python
# Input formats supported:
"RELIANCE"     → "RELIANCE.NS" (NSE default)
"RELIANCE.NS"  → "RELIANCE.NS" (NSE explicit)
"RELIANCE.BO"  → "RELIANCE.BO" (BSE explicit)
"reliance"     → "RELIANCE.NS" (case-insensitive)
```

### Method Signature Updates
```python
# Current:
async def get_historical_data(symbol: str, lookback_days: int = 365) -> pd.DataFrame

# New:
async def get_historical_data(
    symbol: str, 
    lookback_days: int = 365,
    provider: str = "tiingo",
    market: str = "us"
) -> pd.DataFrame
```

### MCP Tool Design
```python
@mcp.tool()
async def analyze_indian_stock(
    ctx: Context,
    symbol: str,
    exchange: str = "NSE",
    lookback_days: int = 365
) -> str:
    """Analyze Indian stocks from NSE or BSE"""
```

### Cache Key Structure
```python
# Current: f"stock:{symbol}:history:{days}"
# New:     f"stock:{symbol}:history:{days}:{provider}:{market}"
```

## 🚀 File Modification List

### 1. `/src/mcp_trader/data.py`
- Add `normalize_indian_symbol()` utility
- Extend `get_historical_data()` with provider logic
- Add yfinance integration with async wrapper
- Update error handling for Indian symbols

### 2. `/src/mcp_trader/models.py`  
- Update `MarketDataResponse` with Indian provider support
- Add market type enum: `us`, `crypto`, `in`
- Extend Pydantic models for validation

### 3. `/src/mcp_trader/server.py`
- Add `analyze_indian_stock()` MCP tool
- Update FastMCP resources for Indian stocks
- Extend cache key generation logic

### 4. `/src/mcp_trader/config.py`
- Add `default_indian_provider = "yfinance"`
- Add `default_indian_exchange = "NSE"`
- Add yfinance-specific configuration

### 5. `/pyproject.toml`
- Ensure yfinance dependency is explicit
- Update version constraints if needed

### 6. `/src/mcp_trader/tests/`
- Add `test_indian_stocks.py`
- Mock yfinance responses
- Test symbol normalization
- Test error handling

## 📋 Success Criteria

### Functional Requirements
- [ ] Support NSE and BSE stocks with proper symbol formatting
- [ ] Historical data retrieval (1 day to 10+ years)
- [ ] Technical analysis integration with existing indicators
- [ ] Proper error handling for invalid Indian symbols
- [ ] Cache integration with existing FastMCP system

### Non-Functional Requirements
- [ ] Maintain existing performance characteristics
- [ ] 80%+ test coverage for new code
- [ ] Zero breaking changes to existing functionality
- [ ] Async/await compatibility throughout
- [ ] Type safety with Pydantic validation

### Integration Requirements
- [ ] Works with existing MCP tools and resources
- [ ] Compatible with current caching mechanism
- [ ] Follows existing code patterns and conventions
- [ ] No additional API keys or authentication required

## 🎯 Popular Indian Stocks for Testing

### NSE Large Cap
- RELIANCE.NS (Reliance Industries)
- TCS.NS (Tata Consultancy Services)
- INFY.NS (Infosys)
- HDFCBANK.NS (HDFC Bank)
- ITC.NS (ITC Limited)

### NSE Mid Cap
- ADANIPORTS.NS (Adani Ports)
- BAJFINANCE.NS (Bajaj Finance)
- MARUTI.NS (Maruti Suzuki)

### BSE Examples
- RELIANCE.BO (Reliance Industries on BSE)
- TCS.BO (TCS on BSE)

## 🚦 Implementation Priority

1. **HIGH**: Core data.py implementation with yfinance
2. **HIGH**: Symbol normalization and validation
3. **MEDIUM**: MCP tool integration
4. **MEDIUM**: Comprehensive testing
5. **LOW**: Documentation updates

## 🔄 Testing Strategy

### Unit Tests
- Symbol normalization edge cases
- Provider selection logic
- Error handling for invalid symbols
- Data format validation

### Integration Tests
- End-to-end MCP tool functionality
- Cache behavior with Indian stocks
- Performance with large datasets

### Manual Testing
- Test with popular Indian stocks
- Verify technical analysis accuracy
- Check error messages for user experience

This plan provides a comprehensive roadmap for implementing Indian stock support while maintaining the high code quality and architecture patterns of the existing MCP Trader system.