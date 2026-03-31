"""
Technical Agent module for Lakshya AI Multi-Agent System.

This module implements the Technical Agent responsible for analyzing stock price
movements, calculating technical indicators (moving averages, RSI), identifying
trading signals, and generating LLM-powered insights using Amazon Bedrock.

Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7
"""

import time
from typing import Dict, List, Any
import pandas as pd
from utils.aws_helper import AWSHelper
from services.stock_data_service import StockDataService
from models.data_models import TechnicalAnalysisOutput
from utils.exceptions import AnalysisError, InsufficientDataError
from utils.logger import setup_logger, log_agent_start, log_agent_complete


class TechnicalAgent:
    """
    Technical analysis agent for stock price and momentum analysis.
    
    This agent performs technical analysis by:
    - Calculating 50-day and 200-day moving averages
    - Computing 14-day Relative Strength Index (RSI)
    - Identifying technical signals (Golden Cross, Death Cross, Overbought, Oversold)
    - Generating LLM-powered natural language insights via Bedrock
    
    Attributes:
        aws_helper: AWS helper for Bedrock LLM invocation
        stock_service: Stock data service for fetching historical prices
        logger: Logger instance for technical agent operations
    
    Requirements:
        - 3.1: Calculate 50-day and 200-day moving averages
        - 3.2: Calculate 14-day RSI
        - 3.3: Identify overbought condition (RSI > 70)
        - 3.4: Identify oversold condition (RSI < 30)
        - 3.5: Use Bedrock for natural language analysis
        - 3.6: Return structured TechnicalAnalysisOutput
        - 3.7: Handle insufficient data errors
    """
    
    # Technical indicator parameters
    MA_SHORT_PERIOD = 50   # 50-day moving average
    MA_LONG_PERIOD = 200   # 200-day moving average
    RSI_PERIOD = 14        # 14-day RSI
    RSI_OVERBOUGHT = 70    # RSI threshold for overbought
    RSI_OVERSOLD = 30      # RSI threshold for oversold
    
    def __init__(self, aws_helper: AWSHelper, stock_service: StockDataService):
        """
        Initialize Technical Agent with dependencies.
        
        Args:
            aws_helper: AWS helper for Bedrock LLM access
            stock_service: Stock data service for historical data
        """
        self.aws_helper = aws_helper
        self.stock_service = stock_service
        self.logger = setup_logger(__name__)
        
        self.logger.info("TechnicalAgent initialized")
    
    def calculate_moving_averages(self, prices: pd.Series) -> Dict[str, float]:
        """
        Calculate 50-day and 200-day simple moving averages.
        
        Args:
            prices: Series of closing prices
        
        Returns:
            Dictionary with 'ma_50' and 'ma_200' keys
        
        Raises:
            InsufficientDataError: If not enough data for calculations
        
        Requirements:
            - 3.1: Calculate 50-day and 200-day moving averages
            - 3.7: Handle insufficient data errors
        """
        if len(prices) < self.MA_SHORT_PERIOD:
            raise InsufficientDataError(
                f"Insufficient data for 50-day MA: need {self.MA_SHORT_PERIOD} "
                f"data points, got {len(prices)}"
            )
        
        if len(prices) < self.MA_LONG_PERIOD:
            self.logger.warning(
                f"Insufficient data for 200-day MA: need {self.MA_LONG_PERIOD} "
                f"data points, got {len(prices)}. Will calculate 50-day MA only."
            )
            ma_200 = None
        else:
            ma_200 = prices.rolling(window=self.MA_LONG_PERIOD).mean().iloc[-1]
        
        ma_50 = prices.rolling(window=self.MA_SHORT_PERIOD).mean().iloc[-1]
        
        ma_200_str = f"{ma_200:.2f}" if ma_200 is not None else "N/A"
        self.logger.debug(
            f"Moving averages calculated | MA-50: {ma_50:.2f} | "
            f"MA-200: {ma_200_str}"
        )
        
        return {
            'ma_50': float(ma_50),
            'ma_200': float(ma_200) if ma_200 is not None else None
        }
    
    def calculate_rsi(self, prices: pd.Series, period: int = RSI_PERIOD) -> float:
        """
        Calculate Relative Strength Index (RSI).
        
        RSI is calculated using the standard formula:
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss over the period
        
        Args:
            prices: Series of closing prices
            period: RSI period (default: 14 days)
        
        Returns:
            RSI value (0-100)
        
        Raises:
            InsufficientDataError: If not enough data for calculation
        
        Requirements:
            - 3.2: Calculate 14-day RSI
            - 3.7: Handle insufficient data errors
        """
        if len(prices) < period + 1:
            raise InsufficientDataError(
                f"Insufficient data for RSI calculation: need {period + 1} "
                f"data points, got {len(prices)}"
            )
        
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0.0)
        losses = -delta.where(delta < 0, 0.0)
        
        # Calculate average gains and losses using exponential moving average
        avg_gains = gains.rolling(window=period, min_periods=period).mean()
        avg_losses = losses.rolling(window=period, min_periods=period).mean()
        
        # Calculate RS (Relative Strength)
        rs = avg_gains / avg_losses
        
        # Calculate RSI
        rsi = 100.0 - (100.0 / (1.0 + rs))
        
        # Get the most recent RSI value
        current_rsi = rsi.iloc[-1]
        
        # Handle edge cases
        if pd.isna(current_rsi):
            # If all losses are zero, RSI = 100
            if avg_losses.iloc[-1] == 0:
                current_rsi = 100.0
            # If all gains are zero, RSI = 0
            elif avg_gains.iloc[-1] == 0:
                current_rsi = 0.0
            else:
                raise AnalysisError("Failed to calculate RSI: result is NaN")
        
        self.logger.debug(f"RSI calculated: {current_rsi:.2f}")
        
        return float(current_rsi)
    
    def identify_signals(
        self,
        current_price: float,
        ma_50: float,
        ma_200: float,
        rsi: float,
        historical_prices: pd.Series
    ) -> List[str]:
        """
        Identify technical trading signals.
        
        Signals identified:
        - Golden Cross: 50-day MA crosses above 200-day MA (bullish)
        - Death Cross: 50-day MA crosses below 200-day MA (bearish)
        - Overbought: RSI > 70
        - Oversold: RSI < 30
        
        Args:
            current_price: Current stock price
            ma_50: 50-day moving average
            ma_200: 200-day moving average (can be None)
            rsi: Current RSI value
            historical_prices: Historical price series for cross detection
        
        Returns:
            List of signal strings
        
        Requirements:
            - 3.3: Identify overbought condition (RSI > 70)
            - 3.4: Identify oversold condition (RSI < 30)
        """
        signals = []
        
        # RSI-based signals
        if rsi > self.RSI_OVERBOUGHT:
            signals.append("Overbought")
            self.logger.debug(f"Signal detected: Overbought (RSI={rsi:.2f})")
        elif rsi < self.RSI_OVERSOLD:
            signals.append("Oversold")
            self.logger.debug(f"Signal detected: Oversold (RSI={rsi:.2f})")
        
        # Moving average cross signals (only if we have 200-day MA)
        if ma_200 is not None and len(historical_prices) >= self.MA_LONG_PERIOD:
            # Calculate previous MAs to detect crosses
            ma_50_series = historical_prices.rolling(window=self.MA_SHORT_PERIOD).mean()
            ma_200_series = historical_prices.rolling(window=self.MA_LONG_PERIOD).mean()
            
            # Get current and previous values
            if len(ma_50_series) >= 2 and len(ma_200_series) >= 2:
                ma_50_current = ma_50_series.iloc[-1]
                ma_50_previous = ma_50_series.iloc[-2]
                ma_200_current = ma_200_series.iloc[-1]
                ma_200_previous = ma_200_series.iloc[-2]
                
                # Golden Cross: 50-day crosses above 200-day
                if ma_50_previous <= ma_200_previous and ma_50_current > ma_200_current:
                    signals.append("Golden Cross")
                    self.logger.debug("Signal detected: Golden Cross")
                
                # Death Cross: 50-day crosses below 200-day
                elif ma_50_previous >= ma_200_previous and ma_50_current < ma_200_current:
                    signals.append("Death Cross")
                    self.logger.debug("Signal detected: Death Cross")
        
        # Additional trend signals based on price vs MAs
        if current_price > ma_50:
            signals.append("Price Above MA-50")
        else:
            signals.append("Price Below MA-50")
        
        if ma_200 is not None:
            if current_price > ma_200:
                signals.append("Price Above MA-200")
            else:
                signals.append("Price Below MA-200")
        
        if not signals:
            signals.append("Neutral")
        
        self.logger.info(f"Technical signals identified: {', '.join(signals)}")
        
        return signals
    
    def generate_llm_insights(self, metrics: Dict[str, Any]) -> str:
        """
        Generate natural language technical analysis using Gemini LLM.
        
        Args:
            metrics: Dictionary containing technical metrics and signals
        
        Returns:
            LLM-generated technical analysis text
        
        Raises:
            AnalysisError: If LLM invocation fails
        
        Requirements:
            - 3.5: Use Gemini for natural language analysis
        """
        ticker = metrics['ticker']
        current_price = metrics['current_price']
        ma_50 = metrics['ma_50']
        ma_200 = metrics['ma_200']
        rsi = metrics['rsi']
        signals = metrics['signals']

        # Compute 7-day linear forecast from last 14 closing prices
        forecast_line = ""
        try:
            import numpy as np
            prices_series = metrics.get('prices_series')
            if prices_series is not None and len(prices_series) >= 14:
                recent = prices_series.tail(14).values
                x = np.arange(len(recent))
                slope, intercept = np.polyfit(x, recent, 1)
                day7_price = round(intercept + slope * (len(recent) + 6), 2)
                forecast_line = (
                    f"\n७-दिवस रेखीय अंदाज: {day7_price:.2f}"
                    f"\n(मागील १४ दिवसांच्या कलावर आधारित)"
                )
        except Exception:
            pass

        # Construct prompt for Claude - response in English
        prompt = f"""You are an expert technical analyst. Analyze the following technical indicators for {ticker}:

Current Price: {current_price:.2f}
50-Day Moving Average (MA-50): {ma_50:.2f}
200-Day Moving Average (MA-200): {f"{ma_200:.2f}" if ma_200 else "Unavailable (insufficient data)"}
RSI (14-day): {rsi:.2f}{forecast_line}

Detected Technical Signals:
{chr(10).join(f"- {signal}" for signal in signals)}

Please provide a concise 2-3 paragraph analysis covering:
1. Current trend based on moving averages and price position
2. Momentum assessment based on RSI
3. Key support/resistance levels indicated by moving averages
4. Short-term outlook (next few weeks)
5. If a 7-day forecast is available, state: "The price is projected to reach X over the next 7 days"

Keep the analysis objective, data-driven, and professional. Write entirely in English."""

        try:
            self.logger.debug(f"Generating LLM insights for {ticker}")
            
            analysis_text = self.aws_helper.invoke_claude(
                prompt=prompt,
                max_tokens=1500,
                temperature=0.7
            )
            
            if not analysis_text or len(analysis_text.strip()) == 0:
                raise AnalysisError("LLM returned empty analysis")
            
            self.logger.info(
                f"LLM insights generated | Length: {len(analysis_text)} chars"
            )
            
            return analysis_text.strip()
            
        except Exception as e:
            self.logger.error(
                f"Failed to generate LLM insights: {str(e)}",
                exc_info=True
            )
            raise AnalysisError(
                f"Failed to generate technical analysis insights: {str(e)}"
            ) from e
    
    def analyze(self, ticker: str) -> TechnicalAnalysisOutput:
        """
        Perform complete technical analysis for a stock.
        
        This method orchestrates all technical analysis steps:
        1. Fetch historical price data
        2. Calculate moving averages (50-day, 200-day)
        3. Calculate RSI (14-day)
        4. Identify technical signals
        5. Generate LLM-powered insights
        6. Return structured output
        
        Args:
            ticker: Stock ticker symbol to analyze
        
        Returns:
            TechnicalAnalysisOutput with all metrics and insights
        
        Raises:
            AnalysisError: If analysis fails
            InsufficientDataError: If not enough historical data
        
        Requirements:
            - 3.1: Calculate moving averages
            - 3.2: Calculate RSI
            - 3.3: Identify overbought condition
            - 3.4: Identify oversold condition
            - 3.5: Use Bedrock for insights
            - 3.6: Return structured output
            - 3.7: Handle insufficient data errors
        """
        start_time = time.time()
        ticker = ticker.upper()
        
        log_agent_start(self.logger, "TechnicalAgent", ticker)
        
        try:
            # Step 1: Fetch historical data (1 year for 200-day MA)
            self.logger.info(f"Fetching historical data for {ticker}")
            historical_data = self.stock_service.fetch_historical_data(
                ticker=ticker,
                period="1y"
            )
            
            # Extract closing prices
            closing_prices = historical_data['Close']
            current_price = float(closing_prices.iloc[-1])
            
            self.logger.info(
                f"Historical data retrieved | Data points: {len(closing_prices)} | "
                f"Current price: {current_price:.2f}"
            )
            
            # Step 2: Calculate moving averages
            self.logger.info("Calculating moving averages")
            ma_dict = self.calculate_moving_averages(closing_prices)
            ma_50 = ma_dict['ma_50']
            ma_200 = ma_dict['ma_200']
            
            # Step 3: Calculate RSI
            self.logger.info("Calculating RSI")
            rsi = self.calculate_rsi(closing_prices)
            
            # Step 4: Identify signals
            self.logger.info("Identifying technical signals")
            signals = self.identify_signals(
                current_price=current_price,
                ma_50=ma_50,
                ma_200=ma_200,
                rsi=rsi,
                historical_prices=closing_prices
            )
            
            # Step 5: Generate LLM insights
            self.logger.info("Generating LLM-powered insights")
            metrics = {
                'ticker': ticker,
                'current_price': current_price,
                'ma_50': ma_50,
                'ma_200': ma_200,
                'rsi': rsi,
                'signals': signals,
                'prices_series': closing_prices   # for 7-day forecast in prompt
            }
            analysis_text = self.generate_llm_insights(metrics)
            
            # Step 6: Create structured output
            # Handle case where ma_200 is None (insufficient data)
            if ma_200 is None:
                # Use ma_50 as fallback for ma_200 to satisfy dataclass validation
                # The analysis text will note the insufficient data
                ma_200 = ma_50
                self.logger.warning(
                    "Using MA-50 as fallback for MA-200 due to insufficient data"
                )
            
            output = TechnicalAnalysisOutput(
                ticker=ticker,
                current_price=current_price,
                ma_50=ma_50,
                ma_200=ma_200,
                rsi=rsi,
                signals=signals,
                analysis_text=analysis_text
            )
            
            execution_time = time.time() - start_time
            log_agent_complete(
                self.logger,
                "TechnicalAgent",
                execution_time,
                success=True,
                additional_info=f"Signals: {len(signals)}"
            )
            
            return output
            
        except (InsufficientDataError, AnalysisError):
            # Re-raise our custom exceptions
            execution_time = time.time() - start_time
            log_agent_complete(
                self.logger,
                "TechnicalAgent",
                execution_time,
                success=False
            )
            raise
            
        except Exception as e:
            execution_time = time.time() - start_time
            log_agent_complete(
                self.logger,
                "TechnicalAgent",
                execution_time,
                success=False
            )
            
            self.logger.error(
                f"Technical analysis failed for {ticker}: {str(e)}",
                exc_info=True
            )
            raise AnalysisError(
                f"Technical analysis failed for {ticker}: {str(e)}"
            ) from e
