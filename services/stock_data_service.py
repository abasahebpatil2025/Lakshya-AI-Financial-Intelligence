"""
Stock Data Service module for Lakshya AI Multi-Agent System.

This module provides an abstraction layer over yfinance for stock data retrieval,
including historical prices, current information, and quarterly results. It implements
caching to reduce API calls and comprehensive error handling.

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6
"""

import time
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
from utils.exceptions import (
    InvalidTickerError,
    DataUnavailableError,
    InsufficientDataError
)
from utils.logger import setup_logger


class StockDataService:
    """
    Stock data retrieval service with caching and error handling.
    
    This class provides methods to fetch stock data from yfinance with:
    - Historical price data with configurable time periods
    - Current stock information (price, market cap, P/E ratio, etc.)
    - Quarterly financial results
    - Caching mechanism (5-minute TTL) to reduce API calls
    - Comprehensive error handling and logging
    
    Attributes:
        logger: Logger instance for data operations
        _cache: Dictionary storing cached data with timestamps
        _cache_ttl: Cache time-to-live in seconds (default: 300 = 5 minutes)
    
    Requirements:
        - 2.1: Fetch historical price data for valid tickers
        - 2.2: Return error for invalid tickers
        - 2.3: Retrieve current stock information
        - 2.4: Fetch quarterly financial results
        - 2.5: Handle data unavailability gracefully
        - 2.6: Support configurable time periods
    """
    
    # Cache TTL: 5 minutes (300 seconds)
    DEFAULT_CACHE_TTL = 300
    
    def __init__(self, cache_ttl: int = DEFAULT_CACHE_TTL):
        """
        Initialize the stock data service.
        
        Args:
            cache_ttl: Cache time-to-live in seconds (default: 300)
        """
        self.logger = setup_logger(__name__)
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = cache_ttl
        
        self.logger.info(
            f"StockDataService initialized | Cache TTL: {cache_ttl}s"
        )
    
    def _get_cache_key(self, ticker: str, data_type: str) -> str:
        """
        Generate cache key for a ticker and data type.
        
        Args:
            ticker: Stock ticker symbol
            data_type: Type of data (e.g., 'historical', 'info', 'quarterly')
        
        Returns:
            Cache key string
        """
        return f"{ticker.upper()}:{data_type}"
    
    def _get_from_cache(self, cache_key: str) -> Optional[Any]:
        """
        Retrieve data from cache if not expired.
        
        Args:
            cache_key: Cache key to retrieve
        
        Returns:
            Cached data if valid, None if expired or not found
        """
        if cache_key in self._cache:
            cached_entry = self._cache[cache_key]
            timestamp = cached_entry.get('timestamp', 0)
            
            # Check if cache is still valid
            if time.time() - timestamp < self._cache_ttl:
                self.logger.debug(f"Cache hit: {cache_key}")
                return cached_entry.get('data')
            else:
                # Cache expired, remove it
                self.logger.debug(f"Cache expired: {cache_key}")
                del self._cache[cache_key]
        
        self.logger.debug(f"Cache miss: {cache_key}")
        return None
    
    def _store_in_cache(self, cache_key: str, data: Any) -> None:
        """
        Store data in cache with current timestamp.
        
        Args:
            cache_key: Cache key to store under
            data: Data to cache
        """
        self._cache[cache_key] = {
            'data': data,
            'timestamp': time.time()
        }
        self.logger.debug(f"Cached: {cache_key}")
    
    def validate_ticker(self, ticker: str) -> bool:
        """
        Check if a ticker symbol is valid.
        
        This method attempts to fetch basic info for the ticker to validate
        its existence. It does not cache the result.
        
        Args:
            ticker: Stock ticker symbol to validate
        
        Returns:
            True if ticker is valid, False otherwise
        
        Requirements:
            - 2.2: Validate ticker symbols
        """
        try:
            self.logger.debug(f"Validating ticker: {ticker}")
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Check if we got valid data
            # yfinance returns an empty dict or dict with 'trailingPegRatio' only for invalid tickers
            if not info or len(info) <= 1:
                self.logger.warning(f"Invalid ticker: {ticker}")
                return False
            
            # Check for symbol in info as additional validation
            if 'symbol' not in info and 'shortName' not in info:
                self.logger.warning(f"Invalid ticker (no symbol/name): {ticker}")
                return False
            
            self.logger.debug(f"Ticker validated: {ticker}")
            return True
            
        except Exception as e:
            self.logger.warning(
                f"Ticker validation failed for {ticker}: {str(e)}"
            )
            return False
    
    def fetch_historical_data(
        self,
        ticker: str,
        period: str = "1y"
    ) -> pd.DataFrame:
        """
        Fetch historical price data for a stock.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period for historical data. Valid values:
                   "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
        
        Returns:
            DataFrame with columns: Date (index), Open, High, Low, Close, Volume
        
        Raises:
            InvalidTickerError: If ticker symbol is invalid
            DataUnavailableError: If data cannot be retrieved
            InsufficientDataError: If not enough data is available
        
        Requirements:
            - 2.1: Fetch historical price data for valid tickers
            - 2.2: Return error for invalid tickers
            - 2.5: Handle data unavailability gracefully
            - 2.6: Support configurable time periods
        """
        ticker = ticker.upper()
        cache_key = self._get_cache_key(ticker, f"historical:{period}")
        
        # Check cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        self.logger.info(
            f"Fetching historical data | Ticker: {ticker} | Period: {period}"
        )
        
        try:
            stock = yf.Ticker(ticker)
            hist_data = stock.history(period=period)
            
            # Validate that we got data
            if hist_data is None or hist_data.empty:
                self.logger.error(
                    f"No historical data available | Ticker: {ticker} | Period: {period}"
                )
                raise DataUnavailableError(
                    f"No historical data available for ticker '{ticker}' "
                    f"with period '{period}'"
                )
            
            # Check if we have sufficient data (at least 2 data points)
            if len(hist_data) < 2:
                self.logger.error(
                    f"Insufficient historical data | Ticker: {ticker} | "
                    f"Data points: {len(hist_data)}"
                )
                raise InsufficientDataError(
                    f"Insufficient historical data for ticker '{ticker}': "
                    f"only {len(hist_data)} data points available"
                )
            
            # Ensure required columns are present
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in hist_data.columns]
            
            if missing_columns:
                self.logger.error(
                    f"Missing required columns | Ticker: {ticker} | "
                    f"Missing: {missing_columns}"
                )
                raise DataUnavailableError(
                    f"Historical data for '{ticker}' is missing required columns: "
                    f"{', '.join(missing_columns)}"
                )
            
            # Store in cache
            self._store_in_cache(cache_key, hist_data)
            
            self.logger.info(
                f"Historical data fetched successfully | Ticker: {ticker} | "
                f"Data points: {len(hist_data)} | Period: {period}"
            )
            
            return hist_data
            
        except (InvalidTickerError, DataUnavailableError, InsufficientDataError):
            # Re-raise our custom exceptions
            raise
            
        except Exception as e:
            self.logger.error(
                f"Failed to fetch historical data | Ticker: {ticker} | "
                f"Period: {period} | Error: {str(e)}",
                exc_info=True
            )
            
            # Check if it's a ticker validation issue
            if "No data found" in str(e) or "404" in str(e):
                raise InvalidTickerError(
                    f"Invalid ticker symbol: '{ticker}'"
                ) from e
            
            raise DataUnavailableError(
                f"Failed to fetch historical data for '{ticker}': {str(e)}"
            ) from e
    
    def fetch_current_info(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch current stock information.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            Dictionary containing:
                - price: Current stock price (float)
                - market_cap: Market capitalization (float)
                - pe_ratio: Price-to-earnings ratio (float or None)
                - volume: Trading volume (int)
                - name: Company name (str)
        
        Raises:
            InvalidTickerError: If ticker symbol is invalid
            DataUnavailableError: If data cannot be retrieved
        
        Requirements:
            - 2.3: Retrieve current stock information including price, market cap, P/E ratio
            - 2.2: Return error for invalid tickers
            - 2.5: Handle data unavailability gracefully
        """
        ticker = ticker.upper()
        cache_key = self._get_cache_key(ticker, "info")
        
        # Check cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        self.logger.info(f"Fetching current info | Ticker: {ticker}")
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Validate that we got data
            if not info or len(info) <= 1:
                self.logger.error(f"Invalid ticker or no data | Ticker: {ticker}")
                raise InvalidTickerError(f"Invalid ticker symbol: '{ticker}'")
            
            # Extract required fields with fallbacks
            current_info = {
                'price': info.get('currentPrice') or info.get('regularMarketPrice') or info.get('previousClose'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('trailingPE') or info.get('forwardPE'),
                'volume': info.get('volume') or info.get('regularMarketVolume'),
                'name': info.get('shortName') or info.get('longName') or ticker
            }
            
            # Validate that we have at least price and name
            if current_info['price'] is None:
                self.logger.error(
                    f"Missing critical data (price) | Ticker: {ticker}"
                )
                raise DataUnavailableError(
                    f"Current price not available for ticker '{ticker}'"
                )
            
            # Log warnings for missing optional fields
            if current_info['market_cap'] is None:
                self.logger.warning(
                    f"Market cap not available | Ticker: {ticker}"
                )
            
            if current_info['pe_ratio'] is None:
                self.logger.warning(
                    f"P/E ratio not available | Ticker: {ticker}"
                )
            
            if current_info['volume'] is None:
                self.logger.warning(
                    f"Volume not available | Ticker: {ticker}"
                )
                current_info['volume'] = 0  # Default to 0 if not available
            
            # Store in cache
            self._store_in_cache(cache_key, current_info)
            
            self.logger.info(
                f"Current info fetched successfully | Ticker: {ticker} | "
                f"Price: ${current_info['price']:.2f} | "
                f"Market Cap: ${current_info['market_cap']:,.0f}" if current_info['market_cap'] else "Market Cap: N/A"
            )
            
            return current_info
            
        except (InvalidTickerError, DataUnavailableError):
            # Re-raise our custom exceptions
            raise
            
        except Exception as e:
            self.logger.error(
                f"Failed to fetch current info | Ticker: {ticker} | "
                f"Error: {str(e)}",
                exc_info=True
            )
            
            # Check if it's a ticker validation issue
            if "No data found" in str(e) or "404" in str(e):
                raise InvalidTickerError(
                    f"Invalid ticker symbol: '{ticker}'"
                ) from e
            
            raise DataUnavailableError(
                f"Failed to fetch current info for '{ticker}': {str(e)}"
            ) from e
    
    def fetch_quarterly_results(self, ticker: str) -> pd.DataFrame:
        """
        Fetch quarterly financial results for a stock.
        
        Args:
            ticker: Stock ticker symbol
        
        Returns:
            DataFrame with quarterly earnings data
        
        Raises:
            InvalidTickerError: If ticker symbol is invalid
            DataUnavailableError: If data cannot be retrieved
        
        Requirements:
            - 2.4: Fetch quarterly financial results
            - 2.2: Return error for invalid tickers
            - 2.5: Handle data unavailability gracefully
        """
        ticker = ticker.upper()
        cache_key = self._get_cache_key(ticker, "quarterly")
        
        # Check cache first
        cached_data = self._get_from_cache(cache_key)
        if cached_data is not None:
            return cached_data
        
        self.logger.info(f"Fetching quarterly results | Ticker: {ticker}")
        
        try:
            stock = yf.Ticker(ticker)
            quarterly_data = stock.quarterly_financials
            
            # Validate that we got data
            if quarterly_data is None or quarterly_data.empty:
                self.logger.warning(
                    f"No quarterly data available | Ticker: {ticker}"
                )
                raise DataUnavailableError(
                    f"No quarterly financial data available for ticker '{ticker}'"
                )
            
            # Store in cache
            self._store_in_cache(cache_key, quarterly_data)
            
            self.logger.info(
                f"Quarterly results fetched successfully | Ticker: {ticker} | "
                f"Quarters available: {len(quarterly_data.columns)}"
            )
            
            return quarterly_data
            
        except (InvalidTickerError, DataUnavailableError):
            # Re-raise our custom exceptions
            raise
            
        except Exception as e:
            self.logger.error(
                f"Failed to fetch quarterly results | Ticker: {ticker} | "
                f"Error: {str(e)}",
                exc_info=True
            )
            
            # Check if it's a ticker validation issue
            if "No data found" in str(e) or "404" in str(e):
                raise InvalidTickerError(
                    f"Invalid ticker symbol: '{ticker}'"
                ) from e
            
            raise DataUnavailableError(
                f"Failed to fetch quarterly results for '{ticker}': {str(e)}"
            ) from e
    
    def clear_cache(self, ticker: Optional[str] = None) -> None:
        """
        Clear cached data.
        
        Args:
            ticker: If provided, clear cache only for this ticker.
                   If None, clear entire cache.
        """
        if ticker:
            ticker = ticker.upper()
            keys_to_remove = [
                key for key in self._cache.keys()
                if key.startswith(f"{ticker}:")
            ]
            for key in keys_to_remove:
                del self._cache[key]
            self.logger.info(f"Cache cleared for ticker: {ticker}")
        else:
            self._cache.clear()
            self.logger.info("Entire cache cleared")
