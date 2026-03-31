"""
Fundamental Agent module for Lakshya AI Multi-Agent System.

This module implements the Fundamental Agent responsible for analyzing company
financials, evaluating fundamental metrics (P/E ratio, market cap, earnings trends),
and generating LLM-powered insights using Amazon Bedrock.

Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6
"""

import time
from typing import Dict, Any
import pandas as pd
from utils.aws_helper import AWSHelper
from services.stock_data_service import StockDataService
from models.data_models import FundamentalAnalysisOutput
from utils.exceptions import AnalysisError, DataUnavailableError
from utils.logger import setup_logger, log_agent_start, log_agent_complete


class FundamentalAgent:
    """
    Fundamental analysis agent for company financial and valuation analysis.
    
    This agent performs fundamental analysis by:
    - Categorizing market capitalization (Large/Mid/Small cap)
    - Analyzing P/E ratio for valuation assessment
    - Evaluating quarterly earnings trends
    - Generating LLM-powered natural language insights via Bedrock
    
    Attributes:
        aws_helper: AWS helper for Bedrock LLM invocation
        stock_service: Stock data service for fetching fundamental data
        logger: Logger instance for fundamental agent operations
    
    Requirements:
        - 4.1: Analyze P/E ratio and compare to industry averages
        - 4.2: Evaluate market capitalization to determine company size
        - 4.3: Analyze quarterly earnings results and trends
        - 4.4: Use Bedrock for natural language insights
        - 4.5: Return structured FundamentalAnalysisOutput
        - 4.6: Handle missing fundamental data errors
    """
    
    # Market cap thresholds (in dollars)
    LARGE_CAP_THRESHOLD = 10_000_000_000   # $10 billion
    MID_CAP_THRESHOLD = 2_000_000_000      # $2 billion
    
    # P/E ratio thresholds (general market averages)
    PE_UNDERVALUED_THRESHOLD = 15.0
    PE_OVERVALUED_THRESHOLD = 25.0
    
    def __init__(self, aws_helper: AWSHelper, stock_service: StockDataService):
        """
        Initialize Fundamental Agent with dependencies.
        
        Args:
            aws_helper: AWS helper for Bedrock LLM access
            stock_service: Stock data service for fundamental data
        """
        self.aws_helper = aws_helper
        self.stock_service = stock_service
        self.logger = setup_logger(__name__)
        
        self.logger.info("FundamentalAgent initialized")
    
    def categorize_market_cap(self, market_cap: float) -> str:
        """
        Categorize market capitalization as Large/Mid/Small cap.
        
        Categories:
        - Large Cap: > $10 billion
        - Mid Cap: $2 billion - $10 billion
        - Small Cap: < $2 billion
        
        Args:
            market_cap: Market capitalization in dollars
        
        Returns:
            Category string: "Large", "Mid", or "Small"
        
        Requirements:
            - 4.2: Evaluate market capitalization to determine company size
        """
        if market_cap >= self.LARGE_CAP_THRESHOLD:
            category = "Large"
        elif market_cap >= self.MID_CAP_THRESHOLD:
            category = "Mid"
        else:
            category = "Small"
        
        self.logger.debug(
            f"Market cap categorized | Value: ${market_cap:,.0f} | "
            f"Category: {category}"
        )
        
        return category
    
    def analyze_pe_ratio(self, pe_ratio: float) -> str:
        """
        Assess P/E ratio for valuation (Undervalued/Fair/Overvalued).
        
        Assessment criteria (general market averages):
        - Undervalued: P/E < 15
        - Fair: P/E 15-25
        - Overvalued: P/E > 25
        
        Args:
            pe_ratio: Price-to-Earnings ratio
        
        Returns:
            Assessment string: "Undervalued", "Fair", or "Overvalued"
        
        Requirements:
            - 4.1: Analyze P/E ratio and compare to industry averages
        """
        if pe_ratio < self.PE_UNDERVALUED_THRESHOLD:
            assessment = "Undervalued"
        elif pe_ratio <= self.PE_OVERVALUED_THRESHOLD:
            assessment = "Fair"
        else:
            assessment = "Overvalued"
        
        self.logger.debug(
            f"P/E ratio assessed | Value: {pe_ratio:.2f} | "
            f"Assessment: {assessment}"
        )
        
        return assessment
    
    def analyze_earnings_trend(self, quarterly_data: pd.DataFrame) -> str:
        """
        Identify earnings growth/decline trends from quarterly data.
        
        Analyzes the most recent quarters to determine if earnings are:
        - Growing: Consistent increase in recent quarters
        - Stable: Relatively flat earnings
        - Declining: Consistent decrease in recent quarters
        
        Args:
            quarterly_data: DataFrame with quarterly financial results
        
        Returns:
            Trend string: "Growing", "Stable", or "Declining"
        
        Requirements:
            - 4.3: Analyze quarterly earnings results and trends
        """
        try:
            # Try to find revenue or net income rows
            # yfinance uses different row names, so we'll try multiple options
            earnings_row = None
            possible_row_names = [
                'Total Revenue',
                'Net Income',
                'Operating Income',
                'Gross Profit'
            ]
            
            for row_name in possible_row_names:
                if row_name in quarterly_data.index:
                    earnings_row = quarterly_data.loc[row_name]
                    self.logger.debug(f"Using '{row_name}' for earnings trend analysis")
                    break
            
            if earnings_row is None or earnings_row.empty:
                self.logger.warning(
                    "No suitable earnings data found in quarterly results. "
                    "Defaulting to 'Stable' trend."
                )
                return "Stable"
            
            # Get the most recent quarters (up to 4)
            recent_quarters = earnings_row.head(min(4, len(earnings_row)))
            
            # Remove any NaN values
            recent_quarters = recent_quarters.dropna()
            
            if len(recent_quarters) < 2:
                self.logger.warning(
                    f"Insufficient quarterly data for trend analysis: "
                    f"{len(recent_quarters)} quarters. Defaulting to 'Stable'."
                )
                return "Stable"
            
            # Calculate quarter-over-quarter changes
            # Note: yfinance returns quarters in reverse chronological order (newest first)
            # But we use .head() which gets the first N columns as they appear
            # So we need to reverse to get chronological order (oldest to newest)
            # However, the actual order depends on how yfinance structures the data
            # To be safe, we'll assume the data is already in the order it appears
            # and calculate changes from left to right (first to last column)
            quarters_values = recent_quarters.values
            
            # Calculate percentage changes from one quarter to the next
            changes = []
            for i in range(1, len(quarters_values)):
                if quarters_values[i-1] != 0:
                    change = (quarters_values[i] - quarters_values[i-1]) / abs(quarters_values[i-1])
                    changes.append(change)
            
            if len(changes) == 0:
                return "Stable"
            
            avg_change = sum(changes) / len(changes)
            
            if len(changes) == 0:
                return "Stable"
            
            # Thresholds for trend determination
            GROWTH_THRESHOLD = 0.05   # 5% average growth
            DECLINE_THRESHOLD = -0.05  # 5% average decline
            
            if avg_change > GROWTH_THRESHOLD:
                trend = "Growing"
            elif avg_change < DECLINE_THRESHOLD:
                trend = "Declining"
            else:
                trend = "Stable"
            
            self.logger.debug(
                f"Earnings trend analyzed | Quarters: {len(recent_quarters)} | "
                f"Avg change: {avg_change:.2%} | Trend: {trend}"
            )
            
            return trend
            
        except Exception as e:
            self.logger.warning(
                f"Error analyzing earnings trend: {str(e)}. "
                "Defaulting to 'Stable'."
            )
            return "Stable"
    
    def generate_llm_insights(self, metrics: Dict[str, Any]) -> str:
        """
        Generate natural language fundamental analysis using Bedrock LLM.
        
        Args:
            metrics: Dictionary containing fundamental metrics
        
        Returns:
            LLM-generated fundamental analysis text
        
        Raises:
            AnalysisError: If LLM invocation fails
        
        Requirements:
            - 4.4: Use Bedrock for natural language insights
        """
        ticker = metrics['ticker']
        company_name = metrics['company_name']
        market_cap = metrics['market_cap']
        market_cap_category = metrics['market_cap_category']
        pe_ratio = metrics['pe_ratio']
        pe_assessment = metrics['pe_assessment']
        earnings_trend = metrics['earnings_trend']
        
        # Format market cap for display
        if market_cap >= 1_000_000_000:
            market_cap_display = f"${market_cap / 1_000_000_000:.2f}B"
        elif market_cap >= 1_000_000:
            market_cap_display = f"${market_cap / 1_000_000:.2f}M"
        else:
            market_cap_display = f"${market_cap:,.0f}"
        
        # Construct prompt for Claude - response in English
        prompt = f"""You are an expert fundamental analyst. Analyze the following fundamental indicators for {ticker} ({company_name}):

Company: {company_name}
Market Cap: {market_cap_display} ({market_cap_category} Cap)
P/E Ratio: {pe_ratio:.2f}
Valuation: {pe_assessment}
Quarterly Earnings Trend: {earnings_trend}

Please provide a concise 2-3 paragraph analysis covering:
1. Valuation based on P/E ratio — is the stock fairly priced relative to its earnings?
2. Company size and market position — what does the market cap indicate about stability?
3. Earnings growth — what does the earnings trend suggest about financial health?
4. Long-term investment potential — what is the outlook based on these fundamentals?

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
                f"Failed to generate fundamental analysis insights: {str(e)}"
            ) from e
    
    def analyze(self, ticker: str) -> FundamentalAnalysisOutput:
        """
        Perform complete fundamental analysis for a stock.
        
        This method orchestrates all fundamental analysis steps:
        1. Fetch current stock info (price, market cap, P/E ratio)
        2. Fetch quarterly financial results
        3. Categorize market capitalization
        4. Assess P/E ratio valuation
        5. Analyze earnings trends
        6. Generate LLM-powered insights
        7. Return structured output
        
        Args:
            ticker: Stock ticker symbol to analyze
        
        Returns:
            FundamentalAnalysisOutput with all metrics and insights
        
        Raises:
            AnalysisError: If analysis fails
            DataUnavailableError: If fundamental data is unavailable
        
        Requirements:
            - 4.1: Analyze P/E ratio
            - 4.2: Evaluate market capitalization
            - 4.3: Analyze quarterly earnings
            - 4.4: Use Bedrock for insights
            - 4.5: Return structured output
            - 4.6: Handle missing fundamental data
        """
        start_time = time.time()
        ticker = ticker.upper()
        
        log_agent_start(self.logger, "FundamentalAgent", ticker)
        
        try:
            # Step 1: Fetch current stock info
            self.logger.info(f"Fetching current stock info for {ticker}")
            current_info = self.stock_service.fetch_current_info(ticker)
            
            company_name = current_info['name']
            market_cap = current_info['market_cap']
            pe_ratio = current_info['pe_ratio']
            
            # Validate required fundamental data
            if market_cap is None:
                self.logger.error(
                    f"Market cap not available for {ticker}"
                )
                raise DataUnavailableError(
                    f"Market cap data not available for ticker '{ticker}'. "
                    "Cannot perform fundamental analysis."
                )
            
            if pe_ratio is None:
                self.logger.warning(
                    f"P/E ratio not available for {ticker}. "
                    "This may indicate negative earnings or missing data. "
                    "Using default value of 0.0."
                )
                pe_ratio = 0.0
            
            self.logger.info(
                f"Stock info retrieved | Company: {company_name} | "
                f"Market Cap: ${market_cap:,.0f} | P/E: {pe_ratio:.2f}"
            )
            
            # Step 2: Fetch quarterly results
            self.logger.info(f"Fetching quarterly results for {ticker}")
            try:
                quarterly_data = self.stock_service.fetch_quarterly_results(ticker)
                self.logger.info(
                    f"Quarterly data retrieved | Quarters: {len(quarterly_data.columns)}"
                )
            except DataUnavailableError as e:
                self.logger.warning(
                    f"Quarterly data not available for {ticker}: {str(e)}. "
                    "Will use 'Stable' as default earnings trend."
                )
                quarterly_data = pd.DataFrame()  # Empty DataFrame
            
            # Step 3: Categorize market cap
            self.logger.info("Categorizing market capitalization")
            market_cap_category = self.categorize_market_cap(market_cap)
            
            # Step 4: Assess P/E ratio
            self.logger.info("Assessing P/E ratio")
            if pe_ratio > 0:
                pe_assessment = self.analyze_pe_ratio(pe_ratio)
            else:
                pe_assessment = "Fair"  # Default for negative or zero P/E
                self.logger.warning(
                    f"P/E ratio is {pe_ratio}, using 'Fair' as default assessment"
                )
            
            # Step 5: Analyze earnings trend
            self.logger.info("Analyzing earnings trend")
            if not quarterly_data.empty:
                earnings_trend = self.analyze_earnings_trend(quarterly_data)
            else:
                earnings_trend = "Stable"
                self.logger.warning(
                    "No quarterly data available, using 'Stable' as default trend"
                )
            
            # Step 6: Generate LLM insights
            self.logger.info("Generating LLM-powered insights")
            metrics = {
                'ticker': ticker,
                'company_name': company_name,
                'market_cap': market_cap,
                'market_cap_category': market_cap_category,
                'pe_ratio': pe_ratio,
                'pe_assessment': pe_assessment,
                'earnings_trend': earnings_trend
            }
            analysis_text = self.generate_llm_insights(metrics)
            
            # Step 7: Create structured output
            output = FundamentalAnalysisOutput(
                ticker=ticker,
                company_name=company_name,
                market_cap=market_cap,
                market_cap_category=market_cap_category,
                pe_ratio=pe_ratio,
                pe_assessment=pe_assessment,
                earnings_trend=earnings_trend,
                analysis_text=analysis_text
            )
            
            execution_time = time.time() - start_time
            log_agent_complete(
                self.logger,
                "FundamentalAgent",
                execution_time,
                success=True,
                additional_info=f"Market Cap: {market_cap_category}, P/E: {pe_assessment}, Trend: {earnings_trend}"
            )
            
            return output
            
        except (DataUnavailableError, AnalysisError):
            # Re-raise our custom exceptions
            execution_time = time.time() - start_time
            log_agent_complete(
                self.logger,
                "FundamentalAgent",
                execution_time,
                success=False
            )
            raise
            
        except Exception as e:
            execution_time = time.time() - start_time
            log_agent_complete(
                self.logger,
                "FundamentalAgent",
                execution_time,
                success=False
            )
            
            self.logger.error(
                f"Fundamental analysis failed for {ticker}: {str(e)}",
                exc_info=True
            )
            raise AnalysisError(
                f"Fundamental analysis failed for {ticker}: {str(e)}"
            ) from e
