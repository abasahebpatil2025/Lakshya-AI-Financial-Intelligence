"""
Risk Manager module for Lakshya AI Multi-Agent System.

This module implements the Risk Manager responsible for synthesizing outputs from
Technical, Fundamental, and Sentiment agents into a final investment recommendation.
It uses Amazon Bedrock (Claude 3.7 Sonnet) to generate Buy/Sell/Hold signals with
confidence scores and detailed reasoning.

Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7
"""

import re
import time
from typing import Optional, Dict, Any
from utils.aws_helper import AWSHelper
from models.data_models import (
    TechnicalAnalysisOutput,
    FundamentalAnalysisOutput,
    SentimentAnalysisOutput,
    Recommendation
)
from utils.exceptions import SynthesisError
from utils.logger import setup_logger, log_agent_start, log_agent_complete


class RiskManager:
    """
    Risk Manager agent for synthesizing multi-agent analysis into final recommendations.
    
    This agent performs decision synthesis by:
    - Collecting outputs from Technical, Fundamental, and Sentiment agents
    - Validating input data completeness
    - Using Bedrock LLM to synthesize all inputs into a final recommendation
    - Parsing LLM output to extract signal (Buy/Sell/Hold) and confidence (0-100)
    - Handling missing agent inputs gracefully
    - Generating structured Recommendation output
    
    Attributes:
        aws_helper: AWS helper for Bedrock LLM invocation
        logger: Logger instance for risk manager operations
    
    Requirements:
        - 6.1: Collect structured outputs from all agents
        - 6.2: Use Bedrock to synthesize inputs into final recommendation
        - 6.3: Generate Buy/Sell/Hold signal
        - 6.4: Provide confidence score (0-100)
        - 6.5: Include reasoning explaining the decision
        - 6.6: Handle missing agent inputs gracefully
        - 6.7: Return structured Recommendation object
    """
    
    # Signal validation
    VALID_SIGNALS = ["BUY", "SELL", "HOLD"]
    
    # Default values for missing inputs
    DEFAULT_SIGNAL = "HOLD"
    DEFAULT_CONFIDENCE = 50
    
    def __init__(self, aws_helper: AWSHelper):
        """
        Initialize Risk Manager with dependencies.
        
        Args:
            aws_helper: AWS helper for Bedrock LLM access
        
        Requirements:
            - 6.1: Collect structured outputs from all agents
        """
        self.aws_helper = aws_helper
        self.logger = setup_logger(__name__)
        
        self.logger.info("RiskManager initialized")
    
    def validate_inputs(
        self,
        technical: Optional[TechnicalAnalysisOutput],
        fundamental: Optional[FundamentalAnalysisOutput],
        sentiment: Optional[SentimentAnalysisOutput]
    ) -> Dict[str, bool]:
        """
        Validate that agent inputs are present and properly structured.
        
        This method checks which agent outputs are available and logs warnings
        for any missing inputs. The Risk Manager can proceed with partial data,
        but will note missing inputs in the final reasoning.
        
        Args:
            technical: Technical analysis output (can be None)
            fundamental: Fundamental analysis output (can be None)
            sentiment: Sentiment analysis output (can be None)
        
        Returns:
            Dictionary with availability status for each agent:
            {
                'technical_available': bool,
                'fundamental_available': bool,
                'sentiment_available': bool,
                'all_available': bool
            }
        
        Requirements:
            - 6.1: Collect structured outputs from all agents
            - 6.6: Handle missing agent inputs gracefully
        """
        availability = {
            'technical_available': technical is not None,
            'fundamental_available': fundamental is not None,
            'sentiment_available': sentiment is not None
        }
        
        availability['all_available'] = all(availability.values())
        
        # Log validation results
        if availability['all_available']:
            self.logger.info("All agent inputs available for synthesis")
        else:
            missing_agents = []
            if not availability['technical_available']:
                missing_agents.append("Technical")
            if not availability['fundamental_available']:
                missing_agents.append("Fundamental")
            if not availability['sentiment_available']:
                missing_agents.append("Sentiment")
            
            self.logger.warning(
                f"Missing agent inputs: {', '.join(missing_agents)}. "
                "Will proceed with available data."
            )
        
        return availability
    
    def generate_recommendation(
        self,
        ticker: str,
        technical: Optional[TechnicalAnalysisOutput],
        fundamental: Optional[FundamentalAnalysisOutput],
        sentiment: Optional[SentimentAnalysisOutput],
        availability: Dict[str, bool]
    ) -> Dict[str, Any]:
        """
        Generate final recommendation using Bedrock LLM.
        
        This method constructs a comprehensive prompt with all available agent
        analyses and uses Claude to synthesize them into a final recommendation.
        The LLM is instructed to provide a signal (BUY/SELL/HOLD), confidence
        score (0-100), and detailed reasoning.
        
        Args:
            ticker: Stock ticker symbol
            technical: Technical analysis output (can be None)
            fundamental: Fundamental analysis output (can be None)
            sentiment: Sentiment analysis output (can be None)
            availability: Dictionary indicating which inputs are available
        
        Returns:
            Dictionary containing raw LLM response text
        
        Raises:
            SynthesisError: If LLM invocation fails
        
        Requirements:
            - 6.2: Use Bedrock to synthesize inputs into final recommendation
            - 6.3: Generate Buy/Sell/Hold signal
            - 6.4: Provide confidence score (0-100)
            - 6.5: Include reasoning explaining the decision
            - 6.6: Handle missing agent inputs gracefully
        """
        # Construct prompt sections for each agent
        technical_section = self._format_technical_section(technical, availability, ticker)
        fundamental_section = self._format_fundamental_section(fundamental, availability, ticker)
        sentiment_section = self._format_sentiment_section(sentiment, availability)
        
        # Build comprehensive synthesis prompt - REASONING in English
        prompt = f"""You are an expert investment advisor synthesizing multiple analyses for {ticker}.

{technical_section}

{fundamental_section}

{sentiment_section}

Based on these analyses, provide a final investment recommendation. Consider:
- Technical signals for short-term momentum and trend
- Fundamental indicators for long-term value and financial health
- Sentiment indicators for market psychology
- Risk factors in the current market environment

Provide your recommendation in this exact format:

RECOMMENDATION: [BUY/SELL/HOLD]
CONFIDENCE: [0-100]
REASONING:
[Write 2-3 paragraphs in English explaining your decision. Cover how each type of analysis influenced your recommendation, what the key factors were, and what risks or opportunities you identified. Be professional, data-driven, and specific.]

Important rules:
- RECOMMENDATION must be exactly BUY, SELL, or HOLD
- CONFIDENCE must be an integer between 0 and 100
- REASONING must be written entirely in English
- If any analysis is unavailable, note its absence in the reasoning
- Be objective and data-driven"""

        try:
            self.logger.debug(f"Generating recommendation for {ticker}")
            
            llm_response = self.aws_helper.invoke_claude(
                prompt=prompt,
                max_tokens=2000,
                temperature=0.7
            )
            
            if not llm_response or len(llm_response.strip()) == 0:
                raise SynthesisError("LLM returned empty recommendation")
            
            self.logger.info(
                f"LLM recommendation generated | Length: {len(llm_response)} chars"
            )
            
            return {'llm_response': llm_response}
            
        except Exception as e:
            self.logger.error(
                f"Failed to generate recommendation: {str(e)}",
                exc_info=True
            )
            raise SynthesisError(
                f"Failed to generate recommendation for {ticker}: {str(e)}"
            ) from e
    
    @staticmethod
    def _currency(ticker: str) -> str:
        """Return ₹ for Indian tickers (.NS/.BO), $ otherwise."""
        t = ticker.upper()
        return "₹" if (t.endswith(".NS") or t.endswith(".BO")) else "$"

    @staticmethod
    def _fmt_market_cap(market_cap: float, ticker: str) -> str:
        """Format market cap with correct currency and unit for the ticker's market."""
        t = ticker.upper()
        is_indian = t.endswith(".NS") or t.endswith(".BO")
        if is_indian:
            cr = market_cap / 1e7  # convert to Crore
            if cr >= 1e5:
                return f"₹ {cr / 1e5:.2f} Lakh Cr"
            elif cr >= 1:
                return f"₹ {cr:,.0f} Cr"
            else:
                return f"₹ {market_cap:,.0f}"
        else:
            if market_cap >= 1e12:
                return f"${market_cap / 1e12:.2f}T"
            elif market_cap >= 1e9:
                return f"${market_cap / 1e9:.2f}B"
            else:
                return f"${market_cap / 1e6:.0f}M"

    def _format_technical_section(
        self,
        technical: Optional[TechnicalAnalysisOutput],
        availability: Dict[str, bool],
        ticker: str = ""
    ) -> str:
        """
        Format technical analysis section for LLM prompt.
        
        Args:
            technical: Technical analysis output (can be None)
            availability: Dictionary indicating input availability
        
        Returns:
            Formatted technical analysis section
        
        Requirements:
            - 6.6: Handle missing agent inputs gracefully
        """
        if not availability['technical_available'] or technical is None:
            return """TECHNICAL ANALYSIS:
[NOT AVAILABLE - Technical analysis could not be completed]

Note: Proceed with recommendation based on available fundamental and sentiment data."""
        
        signals_text = ", ".join(technical.signals) if technical.signals else "None"
        cur = self._currency(ticker)
        
        return f"""TECHNICAL ANALYSIS:
Current Price: {cur}{technical.current_price:.2f}
50-day Moving Average: {cur}{technical.ma_50:.2f}
200-day Moving Average: {cur}{technical.ma_200:.2f}
RSI (14-day): {technical.rsi:.2f}
Technical Signals: {signals_text}

Analysis:
{technical.analysis_text}"""
    
    def _format_fundamental_section(
        self,
        fundamental: Optional[FundamentalAnalysisOutput],
        availability: Dict[str, bool],
        ticker: str = ""
    ) -> str:
        """
        Format fundamental analysis section for LLM prompt.
        
        Args:
            fundamental: Fundamental analysis output (can be None)
            availability: Dictionary indicating input availability
        
        Returns:
            Formatted fundamental analysis section
        
        Requirements:
            - 6.6: Handle missing agent inputs gracefully
        """
        if not availability['fundamental_available'] or fundamental is None:
            return """FUNDAMENTAL ANALYSIS:
[NOT AVAILABLE - Fundamental analysis could not be completed]

Note: Proceed with recommendation based on available technical and sentiment data."""
        
        # Format market cap for display
        market_cap_display = self._fmt_market_cap(fundamental.market_cap, ticker)
        
        return f"""FUNDAMENTAL ANALYSIS:
Company: {fundamental.company_name}
Market Cap: {market_cap_display} ({fundamental.market_cap_category} Cap)
P/E Ratio: {fundamental.pe_ratio:.2f}
Valuation Assessment: {fundamental.pe_assessment}
Earnings Trend: {fundamental.earnings_trend}

Analysis:
{fundamental.analysis_text}"""
    
    def _format_sentiment_section(
        self,
        sentiment: Optional[SentimentAnalysisOutput],
        availability: Dict[str, bool]
    ) -> str:
        """
        Format sentiment analysis section for LLM prompt.
        
        Args:
            sentiment: Sentiment analysis output (can be None)
            availability: Dictionary indicating input availability
        
        Returns:
            Formatted sentiment analysis section
        
        Requirements:
            - 6.6: Handle missing agent inputs gracefully
        """
        if not availability['sentiment_available'] or sentiment is None:
            return """SENTIMENT ANALYSIS:
[NOT AVAILABLE - Sentiment analysis could not be completed]

Note: Proceed with recommendation based on available technical and fundamental data."""
        
        return f"""SENTIMENT ANALYSIS:
Sentiment Score: {sentiment.sentiment_score:.2f} (0=Negative, 1=Positive)
Sentiment Label: {sentiment.sentiment_label}

Analysis:
{sentiment.analysis_text}"""
    
    def parse_llm_response(
        self,
        llm_response: str,
        ticker: str
    ) -> Dict[str, Any]:
        """
        Parse LLM response to extract signal, confidence, and reasoning.
        
        This method uses regex patterns to extract structured data from the
        LLM's natural language response. It validates that the signal is one
        of the valid options (BUY/SELL/HOLD) and that the confidence score
        is in the valid range (0-100).
        
        Args:
            llm_response: Raw text response from LLM
            ticker: Stock ticker symbol (for error messages)
        
        Returns:
            Dictionary containing:
            {
                'signal': str,  # "BUY", "SELL", or "HOLD"
                'confidence': int,  # 0-100
                'reasoning': str  # Detailed reasoning text
            }
        
        Raises:
            SynthesisError: If parsing fails or values are invalid
        
        Requirements:
            - 6.3: Generate Buy/Sell/Hold signal
            - 6.4: Provide confidence score (0-100)
            - 6.5: Include reasoning explaining the decision
        """
        try:
            # Extract RECOMMENDATION using regex
            signal_match = re.search(
                r'RECOMMENDATION:\s*(BUY|SELL|HOLD)',
                llm_response,
                re.IGNORECASE
            )
            
            if not signal_match:
                self.logger.warning(
                    f"Could not parse RECOMMENDATION from LLM response. "
                    f"Using default: {self.DEFAULT_SIGNAL}"
                )
                signal = self.DEFAULT_SIGNAL
            else:
                signal = signal_match.group(1).upper()
            
            # Validate signal
            if signal not in self.VALID_SIGNALS:
                self.logger.warning(
                    f"Invalid signal '{signal}' parsed from LLM. "
                    f"Using default: {self.DEFAULT_SIGNAL}"
                )
                signal = self.DEFAULT_SIGNAL
            
            # Extract CONFIDENCE using regex
            confidence_match = re.search(
                r'CONFIDENCE:\s*(\d+)',
                llm_response,
                re.IGNORECASE
            )
            
            if not confidence_match:
                self.logger.warning(
                    f"Could not parse CONFIDENCE from LLM response. "
                    f"Using default: {self.DEFAULT_CONFIDENCE}"
                )
                confidence = self.DEFAULT_CONFIDENCE
            else:
                confidence = int(confidence_match.group(1))
            
            # Validate confidence range
            if not (0 <= confidence <= 100):
                self.logger.warning(
                    f"Confidence {confidence} out of range [0, 100]. "
                    f"Clamping to valid range."
                )
                confidence = max(0, min(100, confidence))
            
            # Extract REASONING
            reasoning_match = re.search(
                r'REASONING:\s*(.+)',
                llm_response,
                re.IGNORECASE | re.DOTALL
            )
            
            if not reasoning_match:
                self.logger.warning(
                    "Could not parse REASONING from LLM response. "
                    "Using full response as reasoning."
                )
                reasoning = llm_response.strip()
            else:
                reasoning = reasoning_match.group(1).strip()
            
            # Validate reasoning is not empty
            if not reasoning or len(reasoning) < 10:
                raise SynthesisError(
                    f"Reasoning is too short or empty for {ticker}"
                )
            
            self.logger.info(
                f"LLM response parsed | Signal: {signal} | "
                f"Confidence: {confidence} | Reasoning length: {len(reasoning)} chars"
            )
            
            return {
                'signal': signal,
                'confidence': confidence,
                'reasoning': reasoning
            }
            
        except Exception as e:
            self.logger.error(
                f"Failed to parse LLM response: {str(e)}",
                exc_info=True
            )
            raise SynthesisError(
                f"Failed to parse recommendation for {ticker}: {str(e)}"
            ) from e
    
    def synthesize(
        self,
        ticker: str,
        technical: Optional[TechnicalAnalysisOutput] = None,
        fundamental: Optional[FundamentalAnalysisOutput] = None,
        sentiment: Optional[SentimentAnalysisOutput] = None
    ) -> Recommendation:
        """
        Synthesize all agent inputs into final investment recommendation.
        
        This is the main entry point for the Risk Manager. It orchestrates:
        1. Input validation
        2. LLM-based recommendation generation
        3. Response parsing
        4. Structured output creation
        
        The method can handle missing agent inputs gracefully by proceeding
        with available data and noting the limitations in the reasoning.
        
        Args:
            ticker: Stock ticker symbol
            technical: Technical analysis output (optional)
            fundamental: Fundamental analysis output (optional)
            sentiment: Sentiment analysis output (optional)
        
        Returns:
            Recommendation object with signal, confidence, and reasoning
        
        Raises:
            SynthesisError: If synthesis fails
            ValueError: If all inputs are None
        
        Requirements:
            - 6.1: Collect structured outputs from all agents
            - 6.2: Use Bedrock to synthesize inputs
            - 6.3: Generate Buy/Sell/Hold signal
            - 6.4: Provide confidence score (0-100)
            - 6.5: Include reasoning
            - 6.6: Handle missing agent inputs gracefully
            - 6.7: Return structured Recommendation object
        """
        start_time = time.time()
        ticker = ticker.upper()
        
        log_agent_start(self.logger, "RiskManager", ticker)
        
        try:
            # Step 1: Validate inputs
            self.logger.info("Validating agent inputs")
            availability = self.validate_inputs(technical, fundamental, sentiment)
            
            # Check if we have at least one input
            if not any([availability['technical_available'],
                       availability['fundamental_available'],
                       availability['sentiment_available']]):
                raise ValueError(
                    f"Cannot synthesize recommendation for {ticker}: "
                    "all agent inputs are None. At least one agent output is required."
                )
            
            # Step 2: Generate recommendation using LLM
            self.logger.info("Generating LLM-based recommendation")
            llm_result = self.generate_recommendation(
                ticker=ticker,
                technical=technical,
                fundamental=fundamental,
                sentiment=sentiment,
                availability=availability
            )
            
            # Step 3: Parse LLM response
            self.logger.info("Parsing LLM response")
            parsed_result = self.parse_llm_response(
                llm_response=llm_result['llm_response'],
                ticker=ticker
            )
            
            # Step 4: Create summaries for each analysis type
            technical_summary = self._create_technical_summary(technical, availability, ticker)
            fundamental_summary = self._create_fundamental_summary(fundamental, availability, ticker)
            sentiment_summary = self._create_sentiment_summary(sentiment, availability)
            
            # Step 5: Create structured Recommendation output
            recommendation = Recommendation(
                ticker=ticker,
                signal=parsed_result['signal'],
                confidence_score=parsed_result['confidence'],
                reasoning=parsed_result['reasoning'],
                technical_summary=technical_summary,
                fundamental_summary=fundamental_summary,
                sentiment_summary=sentiment_summary
            )
            
            execution_time = time.time() - start_time
            log_agent_complete(
                self.logger,
                "RiskManager",
                execution_time,
                success=True,
                additional_info=f"Signal: {recommendation.signal}, Confidence: {recommendation.confidence_score}"
            )
            
            return recommendation
            
        except (SynthesisError, ValueError):
            # Re-raise our custom exceptions
            execution_time = time.time() - start_time
            log_agent_complete(
                self.logger,
                "RiskManager",
                execution_time,
                success=False
            )
            raise
            
        except Exception as e:
            execution_time = time.time() - start_time
            log_agent_complete(
                self.logger,
                "RiskManager",
                execution_time,
                success=False
            )
            
            self.logger.error(
                f"Synthesis failed for {ticker}: {str(e)}",
                exc_info=True
            )
            raise SynthesisError(
                f"Failed to synthesize recommendation for {ticker}: {str(e)}"
            ) from e
    
    def _create_technical_summary(
        self,
        technical: Optional[TechnicalAnalysisOutput],
        availability: Dict[str, bool],
        ticker: str = ""
    ) -> str:
        """
        Create concise summary of technical analysis.
        
        Args:
            technical: Technical analysis output (can be None)
            availability: Dictionary indicating input availability
        
        Returns:
            Summary string for technical analysis
        
        Requirements:
            - 6.6: Handle missing agent inputs gracefully
            - 6.7: Return structured Recommendation object
        """
        if not availability['technical_available'] or technical is None:
            return "Technical analysis not available"
        
        signals_text = ", ".join(technical.signals[:3]) if technical.signals else "None"
        cur = self._currency(ticker)
        
        return (
            f"Price: {cur}{technical.current_price:.2f} | "
            f"MA-50: {cur}{technical.ma_50:.2f} | "
            f"MA-200: {cur}{technical.ma_200:.2f} | "
            f"RSI: {technical.rsi:.2f} | "
            f"Signals: {signals_text}"
        )
    
    def _create_fundamental_summary(
        self,
        fundamental: Optional[FundamentalAnalysisOutput],
        availability: Dict[str, bool],
        ticker: str = ""
    ) -> str:
        """
        Create concise summary of fundamental analysis.
        
        Args:
            fundamental: Fundamental analysis output (can be None)
            availability: Dictionary indicating input availability
        
        Returns:
            Summary string for fundamental analysis
        
        Requirements:
            - 6.6: Handle missing agent inputs gracefully
            - 6.7: Return structured Recommendation object
        """
        if not availability['fundamental_available'] or fundamental is None:
            return "Fundamental analysis not available"
        
        # Format market cap with correct currency/unit
        market_cap_display = self._fmt_market_cap(fundamental.market_cap, ticker)
        
        return (
            f"{fundamental.company_name} | "
            f"Market Cap: {market_cap_display} ({fundamental.market_cap_category}) | "
            f"P/E: {fundamental.pe_ratio:.2f} ({fundamental.pe_assessment}) | "
            f"Earnings: {fundamental.earnings_trend}"
        )
    
    def _create_sentiment_summary(
        self,
        sentiment: Optional[SentimentAnalysisOutput],
        availability: Dict[str, bool]
    ) -> str:
        """
        Create concise summary of sentiment analysis.
        
        Args:
            sentiment: Sentiment analysis output (can be None)
            availability: Dictionary indicating input availability
        
        Returns:
            Summary string for sentiment analysis
        
        Requirements:
            - 6.6: Handle missing agent inputs gracefully
            - 6.7: Return structured Recommendation object
        """
        if not availability['sentiment_available'] or sentiment is None:
            return "Sentiment analysis not available"
        
        return (
            f"Sentiment: {sentiment.sentiment_label} | "
            f"Score: {sentiment.sentiment_score:.2f}"
        )
