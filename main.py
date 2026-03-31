"""
Main orchestrator for Lakshya AI Multi-Agent System.

This module provides the main entry point for stock analysis, coordinating
the execution of all agents (Technical, Fundamental, Sentiment) and the Risk
Manager to produce final investment recommendations.

Usage:
    python main.py AAPL
    python main.py MSFT --verbose
    python main.py TSLA --output json

Requirements: 7.1, 7.2, 7.3, 7.4, 7.5, 9.1, 9.4, 9.5
"""
import os                          
from dotenv import load_dotenv
import sys
import argparse
import time
from typing import Optional
from utils.aws_helper import AWSHelper
from utils.logger import setup_logger, log_system_event
from services.stock_data_service import StockDataService
from agents.technical_agent import TechnicalAgent
from agents.fundamental_agent import FundamentalAgent
from agents.sentiment_agent import SentimentAgent
from agents.risk_manager import RiskManager
from models.data_models import (
    TechnicalAnalysisOutput,
    FundamentalAnalysisOutput,
    SentimentAnalysisOutput,
    Recommendation
)
from utils.exceptions import (
    LakshyaAIError,
    InvalidTickerError,
    DataUnavailableError,
    AuthenticationError,
    BedrockError
)

load_dotenv()

def analyze_stock(ticker: str, profile_name: Optional[str] = None) -> Recommendation:
    """
    Main entry point for stock analysis.
    
    This function orchestrates the entire analysis pipeline:
    1. Initialize all components (AWS helper, logger, stock service, agents)
    2. Validate ticker input
    3. Execute Technical Agent and capture output
    4. Execute Fundamental Agent and capture output
    5. Execute Sentiment Agent and capture output
    6. Pass all outputs to Risk Manager
    7. Return final Recommendation
    
    The function implements graceful degradation - if individual agents fail,
    the system continues with available data and notes missing inputs in the
    final recommendation.
    
    Args:
        ticker: Stock ticker symbol to analyze
    
    Returns:
        Recommendation object with final investment signal and reasoning
    
    Raises:
        InvalidTickerError: If ticker symbol is invalid
        DataUnavailableError: If stock data cannot be retrieved
        AuthenticationError: If AWS credentials are invalid
        LakshyaAIError: For other system errors
    
    Requirements:
        - 7.1: Accept stock ticker symbol as input
        - 7.2: Execute Technical, Fundamental, and Sentiment agents in sequence
        - 7.3: Pass all agent outputs to Risk Manager
        - 7.4: Display final recommendation
        - 7.5: Log errors and display user-friendly messages
        - 9.1: Continue with remaining agents if one fails
        - 9.4: Log errors with full context
        - 9.5: Validate inputs before processing
    """
    # Initialize logger
    logger = setup_logger(__name__)
    
    log_system_event(
        logger,
        "ANALYSIS_START",
        f"Starting stock analysis for {ticker}"
    )
    
    start_time = time.time()
    
    # Step 1: Initialize components
    logger.info("Initializing system components")
    
    try:

        # Initialize AWS Helper
        logger.info("Initializing AWS Helper")
        # Use lakshya-ai profile if no profile specified
        if profile_name is None:
            profile_name = "lakshya-ai"
        aws_helper = AWSHelper(region="us-east-1", profile_name=profile_name)
        
        # Initialize Stock Data Service
        logger.info("Initializing Stock Data Service")
        stock_service = StockDataService()
        
        # Initialize agents
        logger.info("Initializing agents")
        technical_agent = TechnicalAgent(aws_helper, stock_service)
        fundamental_agent = FundamentalAgent(aws_helper, stock_service)
        sentiment_agent = SentimentAgent(aws_helper)
        risk_manager = RiskManager(aws_helper)
        
        logger.info("All components initialized successfully")
        
    except AuthenticationError as e:
        logger.error(
            f"AWS authentication failed: {str(e)}",
            exc_info=True
        )
        raise AuthenticationError(
            "Failed to authenticate with AWS. Please check your AWS credentials "
            "and ensure they are properly configured."
        ) from e
    
    except Exception as e:
        logger.error(
            f"Failed to initialize system components: {str(e)}",
            exc_info=True
        )
        raise LakshyaAIError(
            f"System initialization failed: {str(e)}"
        ) from e
    
    # Step 2: Validate ticker input
    logger.info(f"Validating ticker: {ticker}")
    
    try:
        ticker = ticker.upper().strip()
        
        # Basic validation
        if not ticker or len(ticker) == 0:
            raise InvalidTickerError("Ticker symbol cannot be empty")
        
        if len(ticker) > 10:
            raise InvalidTickerError(
                f"Ticker symbol '{ticker}' is too long (max 10 characters)"
            )
        
        # Validate with stock service
        if not stock_service.validate_ticker(ticker):
            raise InvalidTickerError(
                f"Invalid ticker symbol: '{ticker}'. "
                "Please check the symbol and try again."
            )
        
        logger.info(f"Ticker validated successfully: {ticker}")
        
    except InvalidTickerError:
        # Re-raise ticker validation errors
        raise
    
    except Exception as e:
        logger.error(
            f"Ticker validation failed: {str(e)}",
            exc_info=True
        )
        raise InvalidTickerError(
            f"Failed to validate ticker '{ticker}': {str(e)}"
        ) from e
    
    # Initialize variables to store agent outputs
    technical_output: Optional[TechnicalAnalysisOutput] = None
    fundamental_output: Optional[FundamentalAnalysisOutput] = None
    sentiment_output: Optional[SentimentAnalysisOutput] = None
    
    # Step 3: Execute Technical Agent
    logger.info("=" * 60)
    logger.info("EXECUTING TECHNICAL AGENT")
    logger.info("=" * 60)
    
    try:
        technical_output = technical_agent.analyze(ticker)
        logger.info("Technical Agent completed successfully")
        
    except Exception as e:
        logger.error(
            f"Technical Agent failed: {str(e)}",
            exc_info=True
        )
        logger.warning(
            "Continuing with remaining agents despite Technical Agent failure"
        )
        # Continue with other agents (graceful degradation)
    
    # Step 4: Execute Fundamental Agent
    logger.info("=" * 60)
    logger.info("EXECUTING FUNDAMENTAL AGENT")
    logger.info("=" * 60)
    
    try:
        fundamental_output = fundamental_agent.analyze(ticker)
        logger.info("Fundamental Agent completed successfully")
        
    except Exception as e:
        logger.error(
            f"Fundamental Agent failed: {str(e)}",
            exc_info=True
        )
        logger.warning(
            "Continuing with remaining agents despite Fundamental Agent failure"
        )
        # Continue with other agents (graceful degradation)
    
    # Step 5: Execute Sentiment Agent
    logger.info("=" * 60)
    logger.info("EXECUTING SENTIMENT AGENT")
    logger.info("=" * 60)
    
    try:
        sentiment_output = sentiment_agent.analyze(ticker)
        logger.info("Sentiment Agent completed successfully")
        
    except Exception as e:
        logger.error(
            f"Sentiment Agent failed: {str(e)}",
            exc_info=True
        )
        logger.warning(
            "Continuing with Risk Manager despite Sentiment Agent failure"
        )
        # Continue with Risk Manager (graceful degradation)
    
    # Step 6: Pass all outputs to Risk Manager
    logger.info("=" * 60)
    logger.info("EXECUTING RISK MANAGER")
    logger.info("=" * 60)
    
    try:
        # Check if we have at least one agent output
        if not any([technical_output, fundamental_output, sentiment_output]):
            raise LakshyaAIError(
                f"Cannot generate recommendation for {ticker}: "
                "all agents failed to produce output. Please check the logs "
                "for details and try again."
            )
        
        recommendation = risk_manager.synthesize(
            ticker=ticker,
            technical=technical_output,
            fundamental=fundamental_output,
            sentiment=sentiment_output
        )
        
        logger.info("Risk Manager completed successfully")
        
    except Exception as e:
        logger.error(
            f"Risk Manager failed: {str(e)}",
            exc_info=True
        )
        raise LakshyaAIError(
            f"Failed to generate recommendation for {ticker}: {str(e)}"
        ) from e
    
    # Step 7: Log completion and return
    total_time = time.time() - start_time
    
    log_system_event(
        logger,
        "ANALYSIS_COMPLETE",
        f"Stock analysis completed for {ticker} | "
        f"Total time: {total_time:.2f}s | "
        f"Signal: {recommendation.signal} | "
        f"Confidence: {recommendation.confidence_score}"
    )
    
    return recommendation


def display_recommendation(recommendation: Recommendation, output_format: str = "text") -> None:
    """
    Format and display recommendation to user.
    
    This function presents the final recommendation in a user-friendly format,
    including the investment signal, confidence score, detailed reasoning, and
    summaries from each agent.
    
    Args:
        recommendation: Recommendation object to display
        output_format: Output format - "text" for human-readable, "json" for JSON
    
    Requirements:
        - 7.4: Display final recommendation with confidence score and reasoning
    """
    if output_format == "json":
        # JSON output for programmatic consumption
        print(recommendation.to_json())
        return
    
    # Text output for human readability
    print("\n" + "=" * 80)
    print(f"INVESTMENT RECOMMENDATION FOR {recommendation.ticker}")
    print("=" * 80)
    print()
    
    # Display signal with visual indicator
    signal_emoji = {
        "BUY": "📈",
        "SELL": "📉",
        "HOLD": "⏸️"
    }
    
    emoji = signal_emoji.get(recommendation.signal, "")
    print(f"SIGNAL: {emoji} {recommendation.signal} {emoji}")
    print(f"CONFIDENCE: {recommendation.confidence_score}/100")
    print(f"TIMESTAMP: {recommendation.timestamp}")
    print()
    
    # Display reasoning
    print("-" * 80)
    print("REASONING:")
    print("-" * 80)
    print(recommendation.reasoning)
    print()
    
    # Display agent summaries
    print("-" * 80)
    print("ANALYSIS SUMMARIES:")
    print("-" * 80)
    print()
    
    print("📊 TECHNICAL ANALYSIS:")
    print(f"   {recommendation.technical_summary}")
    print()
    
    print("💼 FUNDAMENTAL ANALYSIS:")
    print(f"   {recommendation.fundamental_summary}")
    print()
    
    print("💭 SENTIMENT ANALYSIS:")
    print(f"   {recommendation.sentiment_summary}")
    print()
    
    print("=" * 80)
    print()


def main():
    """
    CLI entry point for the Lakshya AI Multi-Agent System.
    
    Parses command-line arguments and executes stock analysis with
    comprehensive error handling and user-friendly error messages.
    
    Usage:
        python main.py AAPL
        python main.py MSFT --verbose
        python main.py TSLA --output json
    
    Requirements:
        - 7.1: Accept stock ticker symbol as input
        - 7.4: Display final recommendation
        - 7.5: Handle errors gracefully with user-friendly messages
        - 9.5: Validate inputs before processing
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Lakshya AI Multi-Agent Stock Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py AAPL              Analyze Apple stock
  python main.py MSFT --verbose    Analyze Microsoft with verbose logging
  python main.py TSLA --output json  Output recommendation as JSON

The system analyzes stocks using multiple AI agents:
  - Technical Agent: Price movements, moving averages, RSI
  - Fundamental Agent: P/E ratio, market cap, earnings trends
  - Sentiment Agent: Market sentiment (placeholder)
  - Risk Manager: Synthesizes all inputs into final recommendation
        """
    )
    
    parser.add_argument(
        "ticker",
        type=str,
        help="Stock ticker symbol to analyze (e.g., AAPL, MSFT, TSLA)"
    )
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging (DEBUG level)"
    )
    
   # ... तुमचा आधीचा कोड
    parser.add_argument(
        "--region",
        "-r",
        type=str,
        default="us-east-1",
        help="AWS region for Bedrock (default: us-east-1)"
    )

    # नवीन profile आर्ग्युमेंट इथे जोडा:
    parser.add_argument(
        "--profile",
        "-p",
        type=str,
        default=None,
        help="AWS profile name (e.g., lakshya-ai)"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Output format: text or json (default: text)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    # ...

    # Set up logger with appropriate level
    log_level = "DEBUG" if args.verbose else "INFO"
    logger = setup_logger(__name__, log_level=log_level)
    
    # Display banner
    if args.output == "text":
        print()
        print("=" * 80)
        print("LAKSHYA AI MULTI-AGENT STOCK ANALYSIS SYSTEM")
        print("=" * 80)
        print(f"Analyzing: {args.ticker.upper()}")
        print(f"AWS Region: {args.region}")
        print(f"AWS Profile: {args.profile or 'default'}")
        print("=" * 80)
        print()
    
    try:
        # Execute analysis (इथे आपण args.profile पास करत आहोत)
        recommendation = analyze_stock(args.ticker, profile_name=args.profile)
        
        # Display results
        display_recommendation(recommendation, output_format=args.output)
        
        # Exit with success
        sys.exit(0)
        
    except InvalidTickerError as e:
        logger.error(f"Invalid ticker: {str(e)}")
        print(f"\n❌ ERROR: {str(e)}", file=sys.stderr)
        print("\nPlease provide a valid stock ticker symbol and try again.", file=sys.stderr)
        print("Example: python main.py AAPL\n", file=sys.stderr)
        sys.exit(1)
    
    except DataUnavailableError as e:
        logger.error(f"Data unavailable: {str(e)}")
        print(f"\n❌ ERROR: {str(e)}", file=sys.stderr)
        print("\nThe stock data service is temporarily unavailable or the ticker", file=sys.stderr)
        print("has insufficient data. Please try again later.\n", file=sys.stderr)
        sys.exit(1)
    
    except AuthenticationError as e:
        logger.error(f"Authentication error: {str(e)}")
        print(f"\n❌ ERROR: AWS Authentication Failed", file=sys.stderr)
        print(f"\n{str(e)}", file=sys.stderr)
        print("\nPlease ensure:", file=sys.stderr)
        print("  1. AWS credentials are configured (aws configure)", file=sys.stderr)
        print("  2. Credentials have access to Bedrock and Secrets Manager", file=sys.stderr)
        print("  3. AWS region is correct (use --region flag if needed)\n", file=sys.stderr)
        sys.exit(1)
    
    except BedrockError as e:
        logger.error(f"Bedrock error: {str(e)}")
        print(f"\n❌ ERROR: Amazon Bedrock Service Error", file=sys.stderr)
        print(f"\n{str(e)}", file=sys.stderr)
        print("\nThis may be due to:", file=sys.stderr)
        print("  1. API rate limits (try again in a few moments)", file=sys.stderr)
        print("  2. Service temporarily unavailable", file=sys.stderr)
        print("  3. Model access not enabled in your AWS account\n", file=sys.stderr)
        sys.exit(1)
    
    except LakshyaAIError as e:
        logger.error(f"System error: {str(e)}")
        print(f"\n❌ ERROR: {str(e)}", file=sys.stderr)
        print("\nPlease check the logs for more details and try again.\n", file=sys.stderr)
        sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        print("\n\n⚠️  Analysis interrupted by user. Exiting...\n", file=sys.stderr)
        sys.exit(130)
    
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        print(f"\n❌ UNEXPECTED ERROR: {str(e)}", file=sys.stderr)
        print("\nAn unexpected error occurred. Please check the logs for details.", file=sys.stderr)
        print("Log file: logs/lakshya_ai.log\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
