"""
Custom exception classes for the Lakshya AI Multi-Agent System.

This module defines a hierarchy of exceptions used throughout the system
to handle various error conditions in a structured and meaningful way.

Exception Hierarchy:
    LakshyaAIError (base)
    ├── AWSError
    │   ├── AuthenticationError
    │   ├── BedrockError
    │   └── SecretNotFoundError
    ├── DataError
    │   ├── InvalidTickerError
    │   ├── DataUnavailableError
    │   └── InsufficientDataError
    ├── AnalysisError
    └── SynthesisError
"""


class LakshyaAIError(Exception):
    """
    Base exception for all Lakshya AI errors.
    
    All custom exceptions in the system inherit from this base class,
    allowing for easy catching of any system-specific error.
    """
    pass


class AWSError(LakshyaAIError):
    """
    AWS service related errors.
    
    Raised when there are issues with AWS service interactions,
    including Bedrock and Secrets Manager operations.
    """
    pass


class AuthenticationError(AWSError):
    """
    Invalid or missing AWS credentials.
    
    Raised when AWS credentials are invalid, expired, or missing,
    preventing authentication with AWS services.
    
    Examples:
        - Missing AWS access key or secret key
        - Expired temporary credentials
        - Invalid IAM permissions
    """
    pass


class BedrockError(AWSError):
    """
    Bedrock API errors.
    
    Raised when there are issues with Amazon Bedrock API calls,
    including rate limiting, model errors, or service unavailability.
    
    Examples:
        - API rate limit exceeded
        - Model invocation failures
        - Invalid model parameters
        - Service temporarily unavailable
    """
    pass


class GeminiError(LakshyaAIError):
    """
    Google Gemini API errors.
    
    Raised when there are issues with Google Gemini API calls,
    including rate limiting, model errors, or service unavailability.
    
    Examples:
        - API rate limit exceeded
        - Model invocation failures
        - Invalid model parameters
        - Service temporarily unavailable
        - Authentication failures
    """
    pass


class SecretNotFoundError(AWSError):
    """
    Secret not found in AWS Secrets Manager.
    
    Raised when attempting to retrieve a secret that doesn't exist
    in AWS Secrets Manager or when access is denied.
    
    Examples:
        - Secret name doesn't exist
        - Insufficient permissions to access secret
        - Secret deleted or scheduled for deletion
    """
    pass


class DataError(LakshyaAIError):
    """
    Stock data related errors.
    
    Base class for all errors related to stock data retrieval
    and processing.
    """
    pass


class InvalidTickerError(DataError):
    """
    Invalid stock ticker symbol.
    
    Raised when a provided stock ticker symbol is not recognized
    or doesn't exist in the data source.
    
    Examples:
        - Ticker symbol doesn't exist
        - Malformed ticker symbol
        - Delisted stock
    """
    pass


class DataUnavailableError(DataError):
    """
    Data source temporarily unavailable.
    
    Raised when the stock data source (e.g., yfinance) is temporarily
    unavailable or experiencing issues.
    
    Examples:
        - Network connectivity issues
        - Data provider service outage
        - API rate limits from data provider
        - Timeout errors
    """
    pass


class InsufficientDataError(DataError):
    """
    Not enough historical data for analysis.
    
    Raised when there isn't enough historical data available to
    perform the requested analysis (e.g., calculating 200-day moving average
    requires at least 200 days of data).
    
    Examples:
        - Newly listed stock with limited history
        - Requested time period exceeds available data
        - Data gaps preventing calculation
    """
    pass


class AnalysisError(LakshyaAIError):
    """
    Agent analysis errors.
    
    Raised when an agent (Technical, Fundamental, or Sentiment)
    encounters an error during analysis that prevents it from
    completing its task.
    
    Examples:
        - Calculation errors in technical indicators
        - Missing required data fields
        - LLM generation failures
        - Invalid analysis parameters
    """
    pass


class SynthesisError(LakshyaAIError):
    """
    Risk Manager synthesis errors.
    
    Raised when the Risk Manager encounters an error while
    synthesizing agent outputs into a final recommendation.
    
    Examples:
        - Invalid agent output format
        - Missing required agent inputs
        - LLM synthesis failures
        - Unable to parse recommendation from LLM output
    """
    pass
