"""
AWS Helper module for Lakshya AI Multi-Agent System.

This module provides centralized AWS service integration for Amazon Bedrock
and AWS Secrets Manager. It handles client initialization, credential management,
and LLM invocation with retry logic and exponential backoff.

**PRODUCTION MODE**: Using real AWS Bedrock with Claude 3 Haiku.
Set MOCK_MODE=true in .env to use mock responses for testing.

Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 9.2, 9.3
"""

import json
import time
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv

# Load environment variables FIRST before reading MOCK_MODE
load_dotenv()

import boto3
from botocore.exceptions import ClientError, BotoCoreError
from utils.exceptions import (
    AuthenticationError,
    BedrockError,
    SecretNotFoundError
)
from utils.logger import setup_logger, log_error_with_context


# MOCK MODE CONFIGURATION
# Read from environment variable - set MOCK_MODE=true in .env for mock mode
# Default to "true" if not set (safer for development)
MOCK_MODE = os.getenv("MOCK_MODE", "true").lower() == "true"


class AWSHelper:
    """
    AWS service integration helper for Bedrock and Secrets Manager.
    
    This class provides methods to interact with AWS services, including:
    - Bedrock Runtime client initialization
    - AWS Secrets Manager secret retrieval
    - Amazon Nova Pro model invocation with retry logic via Converse API
    
    Attributes:
        region: AWS region for service clients
        model_id: Amazon Nova Pro model identifier
        logger: Logger instance for AWS operations
        _bedrock_client: Cached Bedrock Runtime client
        _secrets_client: Cached Secrets Manager client
    
    Requirements:
        - 1.1: Bedrock client with Amazon Nova Pro model
        - 1.2: Credentials from AWS Secrets Manager
        - 1.3: Error logging for AWS operations
        - 1.4: Proper region configuration
        - 1.5: Authentication error handling
        - 9.2: Exponential backoff retry logic
        - 9.3: Network error retry (up to 3 times)
    """
    
    # Using Claude 3 Haiku - lighter model to avoid token limits
    CLAUDE_HAIKU_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
    
    # Retry configuration
    MAX_RETRIES = 3
    INITIAL_RETRY_DELAY = 1.0  # seconds
    
    def __init__(self, region: str = "us-east-1",profile_name: Optional[str] = None):
        """
        Initialize AWS Helper with region configuration.
        
        Args:
            region: AWS region for service clients (default: us-east-1)
        
        Raises:
            AuthenticationError: If AWS credentials are invalid or missing
        
        Requirements:
            - 1.4: Initialize AWS clients with proper region configuration
            - 1.5: Raise authentication error for invalid/missing credentials
        """
        self.region = region
        self.profile_name = profile_name
        self.model_id = self.CLAUDE_HAIKU_MODEL_ID
        self.logger = setup_logger(__name__)
        self.mock_mode = MOCK_MODE
        
        # Cached clients (initialized on first use)
        self._bedrock_client: Optional[Any] = None
        self._secrets_client: Optional[Any] = None
        
        # Validate AWS credentials on initialization (skip in mock mode)
        if not self.mock_mode:
            self._validate_credentials()
            self.logger.info(f"AWSHelper initialized | Region: {region} | Model: Claude 3 Haiku | Production Mode")
        else:
            self.logger.warning("⚠️ MOCK MODE ENABLED - Using mock responses instead of real AWS Bedrock")
            self.logger.info(f"AWSHelper initialized | Region: {region} | Mock Mode: {self.mock_mode}")

    def _get_session(self):
        """Helper to create a session with the configured profile."""
        return boto3.Session(profile_name=self.profile_name)

    def _validate_credentials(self) -> None:
        """
        Validate that AWS credentials are available and valid using the session.
        
        Raises:
            AuthenticationError: If credentials are invalid or missing
        
        Requirements:
            - 1.5: Raise authentication error for invalid/missing credentials
        """
        try:
            # आता 'boto3.client' ऐवजी आपल्या 'self._get_session()' चा वापर करा
            session = self._get_session()
            sts_client = session.client('sts', region_name=self.region)
            
            # Identity validate करा
            sts_client.get_caller_identity()
            self.logger.debug(f"AWS credentials validated successfully using profile: {self.profile_name or 'default'}")
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code in ['InvalidClientTokenId', 'SignatureDoesNotMatch', 
                              'AccessDenied', 'UnrecognizedClientException']:
                log_error_with_context(
                    self.logger,
                    e,
                    "AWS credential validation",
                    {"region": self.region, "profile": self.profile_name}
                )
                raise AuthenticationError(
                    f"Invalid or missing AWS credentials: {str(e)}"
                ) from e
            raise
            
        except BotoCoreError as e:
            log_error_with_context(
                self.logger,
                e,
                "AWS credential validation",
                {"region": self.region, "profile": self.profile_name}
            )
            raise AuthenticationError(
                f"AWS credential validation failed: {str(e)}"
            ) from e

    def get_bedrock_client(self):
        """
        Get or create a Bedrock Runtime client using the configured session.
        """
        if self._bedrock_client is None:
            try:
                # आता थेट आपली _get_session() मेथड वापरा
                session = self._get_session()
                
                self._bedrock_client = session.client(
                    service_name='bedrock-runtime',
                    region_name=self.region
                )
                self.logger.debug(
                    f"Bedrock Runtime client created | Region: {self.region} | Profile: {self.profile_name or 'default'}"
                )
            except Exception as e:
                log_error_with_context(
                    self.logger,
                    e,
                    "Bedrock client creation",
                    {"region": self.region, "profile": self.profile_name}
                )
                raise BedrockError(f"Failed to create Bedrock client: {str(e)}") from e
        
        return self._bedrock_client
    
    def get_secret(self, secret_name: str) -> Dict[str, Any]:
        """
        Retrieve a secret from AWS Secrets Manager.
        
        Args:
            secret_name: Name of the secret to retrieve
        
        Returns:
            Dictionary containing the secret data
        
        Raises:
            SecretNotFoundError: If secret doesn't exist or access denied
            AuthenticationError: If credentials are invalid
            BedrockError: For other AWS errors
        
        Requirements:
            - 1.2: Retrieve credentials from AWS Secrets Manager
            - 1.3: Log errors for AWS operations
        """
        if self._secrets_client is None:
            try:
                # इथं पण session चा वापर करा
                session = self._get_session() 
                self._secrets_client = session.client(
                    service_name='secretsmanager',
                    region_name=self.region
                )
                self.logger.debug(
                    f"Secrets Manager client created | Region: {self.region}"
                )
            except (ClientError, BotoCoreError) as e:
                log_error_with_context(
                    self.logger,
                    e,
                    "Secrets Manager client creation",
                    {"region": self.region}
                )
                raise BedrockError(
                    f"Failed to create Secrets Manager client: {str(e)}"
                ) from e
        
        try:
            self.logger.debug(f"Retrieving secret: {secret_name}")
            response = self._secrets_client.get_secret_value(
                SecretId=secret_name
            )
            
            # Parse secret string as JSON
            secret_string = response.get('SecretString', '{}')
            secret_data = json.loads(secret_string)
            
            self.logger.info(f"Secret retrieved successfully: {secret_name}")
            return secret_data
            
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            
            log_error_with_context(
                self.logger,
                e,
                f"Secret retrieval: {secret_name}",
                {"error_code": error_code, "region": self.region}
            )
            
            if error_code == 'ResourceNotFoundException':
                raise SecretNotFoundError(
                    f"Secret '{secret_name}' not found in AWS Secrets Manager"
                ) from e
            elif error_code in ['AccessDeniedException', 'InvalidRequestException']:
                raise SecretNotFoundError(
                    f"Access denied to secret '{secret_name}': {str(e)}"
                ) from e
            elif error_code in ['InvalidClientTokenId', 'SignatureDoesNotMatch']:
                raise AuthenticationError(
                    f"Invalid AWS credentials: {str(e)}"
                ) from e
            else:
                raise BedrockError(
                    f"Failed to retrieve secret '{secret_name}': {str(e)}"
                ) from e
                
        except (json.JSONDecodeError, BotoCoreError) as e:
            log_error_with_context(
                self.logger,
                e,
                f"Secret parsing: {secret_name}",
                {"region": self.region}
            )
            raise BedrockError(
                f"Failed to parse secret '{secret_name}': {str(e)}"
            ) from e
    
    def _get_mock_response(self, prompt: str) -> str:
        """
        Generate mock response based on the prompt content.
        Returns realistic analysis in plain Marathi text without LaTeX formatting.
        Generates ticker-specific mock data for different stocks.
        """
        prompt_lower = prompt.lower()
        
        # Extract ticker from prompt if available
        import re
        ticker_match = re.search(r'\b([A-Z]{1,5})\b', prompt)
        ticker = ticker_match.group(1) if ticker_match else "AAPL"
        
        # Ticker-specific mock data
        ticker_data = {
            "AAPL": {
                "price": 185.50,
                "ma_50": 178.20,
                "ma_200": 172.40,
                "rsi": 62.5,
                "market_cap": "2.85 ट्रिलियन डॉलर",
                "pe_ratio": 28.5,
                "name": "Apple"
            },
            "MSFT": {
                "price": 412.30,
                "ma_50": 398.75,
                "ma_200": 385.60,
                "rsi": 58.3,
                "market_cap": "3.12 ट्रिलियन डॉलर",
                "pe_ratio": 32.8,
                "name": "Microsoft"
            },
            "GOOGL": {
                "price": 142.80,
                "ma_50": 138.45,
                "ma_200": 135.20,
                "rsi": 55.7,
                "market_cap": "1.78 ट्रिलियन डॉलर",
                "pe_ratio": 24.3,
                "name": "Google"
            },
            "TSLA": {
                "price": 248.90,
                "ma_50": 235.60,
                "ma_200": 228.15,
                "rsi": 68.2,
                "market_cap": "0.79 ट्रिलियन डॉलर",
                "pe_ratio": 65.4,
                "name": "Tesla"
            },
            "AMZN": {
                "price": 178.25,
                "ma_50": 172.80,
                "ma_200": 168.50,
                "rsi": 60.1,
                "market_cap": "1.85 ट्रिलियन डॉलर",
                "pe_ratio": 48.7,
                "name": "Amazon"
            },
            "RELIANCE": {
                "price": 2547.80,
                "ma_50": 2498.30,
                "ma_200": 2465.90,
                "rsi": 56.8,
                "market_cap": "17.25 लाख करोड रुपये",
                "pe_ratio": 26.4,
                "name": "Reliance Industries"
            }
        }
        
        # Get data for the ticker, default to AAPL if not found
        data = ticker_data.get(ticker, ticker_data["AAPL"])
        
        # Technical Analysis Mock Response
        if "technical" in prompt_lower or "moving average" in prompt_lower or "rsi" in prompt_lower:
            return f"""तांत्रिक विश्लेषण (Technical Analysis):

या शेअरचे तांत्रिक संकेतक सकारात्मक दिसत आहेत. ५०-दिवसांची मूव्हिंग अॅव्हरेज २००-दिवसांच्या मूव्हिंग अॅव्हरेजच्या वर गेली आहे, जे गोल्डन क्रॉस म्हणून ओळखले जाते. हे एक मजबूत तेजीचे संकेत आहे.

RSI (Relative Strength Index) {data['rsi']:.1f} वर आहे, जे दर्शवते की शेअर ना ओव्हरबॉट आहे ना ओव्हरसोल्ड. याचा अर्थ असा की किंमतीत वाढीची शक्यता आहे.

सध्याची किंमत {data['price']:.2f} डॉलर आहे, जी ५०-दिवसांच्या मूव्हिंग अॅव्हरेज ({data['ma_50']:.2f} डॉलर) आणि २००-दिवसांच्या मूव्हिंग अॅव्हरेज ({data['ma_200']:.2f} डॉलर) च्या वर आहे. हे तेजीची गती दर्शवते.

एकूणच, तांत्रिक दृष्टीकोनातून हा शेअर खरेदीसाठी योग्य दिसतो."""

        # Fundamental Analysis Mock Response
        elif "fundamental" in prompt_lower or "p/e ratio" in prompt_lower or "market cap" in prompt_lower:
            return f"""मूलभूत विश्लेषण (Fundamental Analysis):

कंपनीचे मार्केट कॅपिटलायझेशन {data['market_cap']} आहे, जे याला लार्ज-कॅप कंपनी बनवते. मोठ्या कंपन्या सामान्यतः अधिक स्थिर असतात.

P/E रेशो {data['pe_ratio']:.1f} आहे, जो सेक्टरच्या सरासरीच्या तुलनेत वाजवी आहे. याचा अर्थ असा की शेअर ना जास्त महाग आहे ना खूप स्वस्त.

तिमाही निकालांमध्ये सातत्याने वाढ दिसून येत आहे. कंपनीचे उत्पन्न आणि नफा दोन्ही वाढत आहेत, जे चांगले संकेत आहे.

कंपनीची आर्थिक स्थिती मजबूत आहे आणि भविष्यात वाढीची चांगली शक्यता दिसते. गुंतवणूकदारांसाठी हा एक चांगला पर्याय असू शकतो."""

        # Risk Manager / Synthesis Mock Response
        elif "synthesize" in prompt_lower or "recommendation" in prompt_lower or "buy" in prompt_lower or "sell" in prompt_lower:
            # Extract actual numbers from the prompt if available
            import re
            
            # Try to extract current price
            price_match = re.search(r'Current Price:\s*\$?([\d.]+)', prompt)
            current_price = float(price_match.group(1)) if price_match else data['price']
            
            # Try to extract MA-50
            ma50_match = re.search(r'50-day Moving Average:\s*\$?([\d.]+)', prompt)
            ma_50 = float(ma50_match.group(1)) if ma50_match else data['ma_50']
            
            # Try to extract MA-200
            ma200_match = re.search(r'200-day Moving Average:\s*\$?([\d.]+)', prompt)
            ma_200 = float(ma200_match.group(1)) if ma200_match else data['ma_200']
            
            # Try to extract RSI
            rsi_match = re.search(r'RSI.*?:\s*([\d.]+)', prompt)
            rsi = float(rsi_match.group(1)) if rsi_match else data['rsi']
            
            # Try to extract Market Cap
            mcap_match = re.search(r'Market Cap:\s*\$?([\d.]+)([BMT])', prompt)
            if mcap_match:
                mcap_value = float(mcap_match.group(1))
                mcap_unit = mcap_match.group(2)
                if mcap_unit == 'T':
                    market_cap_display = f"{mcap_value:.2f} ट्रिलियन डॉलर"
                elif mcap_unit == 'B':
                    market_cap_display = f"{mcap_value:.2f} बिलियन डॉलर"
                else:
                    market_cap_display = f"{mcap_value:.2f} मिलियन डॉलर"
            else:
                market_cap_display = data['market_cap']
            
            # Try to extract P/E Ratio
            pe_match = re.search(r'P/E Ratio:\s*([\d.]+)', prompt)
            pe_ratio = float(pe_match.group(1)) if pe_match else data['pe_ratio']
            
            return f"""RECOMMENDATION: BUY
CONFIDENCE: 78
REASONING:

गुंतवणूक शिफारस (Investment Recommendation):

तांत्रिक विश्लेषणानुसार, शेअरमध्ये मजबूत तेजीची गती दिसत आहे. गोल्डन क्रॉस झाला आहे आणि RSI योग्य पातळीवर आहे, जे पुढील वाढीची शक्यता दर्शवते.

सध्याची किंमत {current_price:.2f} डॉलर आहे. ५०-दिवसांची मूव्हिंग अॅव्हरेज {ma_50:.2f} डॉलर आहे आणि २००-दिवसांची मूव्हिंग अॅव्हरेज {ma_200:.2f} डॉलर आहे. RSI {rsi:.1f} वर आहे, जे आरोग्यदायी गती दर्शवते.

मूलभूत विश्लेषणानुसार, कंपनीची आर्थिक स्थिती उत्तम आहे. मार्केट कॅप {market_cap_display} आहे आणि P/E रेशो {pe_ratio:.1f} आहे, जो वाजवी आहे. तिमाही निकालांमध्ये सातत्याने वाढ दिसत आहे.

बाजारातील भावना तटस्थ आहे, परंतु तांत्रिक आणि मूलभूत दोन्ही घटक सकारात्मक आहेत.

निष्कर्ष: तांत्रिक संकेत आणि मजबूत मूलभूत तत्त्वांच्या संयोजनामुळे, हा शेअर खरेदीसाठी योग्य आहे. उच्च आत्मविश्वासासह (७८ टक्के) खरेदीची शिफारस केली जाते.

सावधगिरी: गुंतवणूक करण्यापूर्वी स्वतःचे संशोधन करा आणि आर्थिक सल्लागाराचा सल्ला घ्या. भूतकाळातील कामगिरी भविष्यातील परिणामांची हमी देत नाही."""

        # Sentiment scoring prompt (from analyze_headlines_with_claude) — must return valid JSON
        elif "sentiment score" in prompt_lower and "json" in prompt_lower:
            import re as _re
            # Count how many headlines are in the prompt (lines starting with a digit + dot)
            headline_lines = _re.findall(r'^\d+\.\s+.+', prompt, flags=re.MULTILINE)
            n = len(headline_lines) if headline_lines else 3
            scores = {str(i + 1): round(0.3 + (i % 3) * 0.15, 2) for i in range(n)}
            import json as _json
            return _json.dumps({
                "scores": scores,
                "insight": f"Overall news sentiment for {ticker} appears moderately bullish with mixed signals."
            })

        # Default response
        else:
            return """विश्लेषण (Analysis):

शेअर बाजारातील गुंतवणूक ही जोखमीची असते. कोणतीही गुंतवणूक करण्यापूर्वी सखोल संशोधन करणे आवश्यक आहे.

तांत्रिक आणि मूलभूत विश्लेषण दोन्ही महत्त्वाचे आहेत. बाजारातील भावना देखील लक्षात घेणे आवश्यक आहे.

आपल्या गुंतवणूकीच्या उद्दिष्टांनुसार आणि जोखीम सहनशीलतेनुसार निर्णय घ्या. दीर्घकालीन दृष्टीकोन ठेवणे फायदेशीर ठरू शकते.

आर्थिक सल्लागाराचा सल्ला घेणे नेहमीच चांगले असते."""
    
    def invoke_claude(
        self,
        prompt: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Invoke Claude 3 Haiku via Amazon Bedrock Converse API with retry logic.
        
        **PRODUCTION MODE**: Uses real AWS Bedrock when MOCK_MODE=false.
        **MOCK MODE**: Returns realistic mock responses in Marathi when MOCK_MODE=true.
        
        This method implements exponential backoff retry logic to handle
        rate limits and transient errors. It will retry up to 3 times
        with increasing delays (1s, 2s, 4s).
        
        For Marathi stock analysis, a default system prompt is added to ensure
        responses are in Marathi language with proper financial terminology.
        
        Args:
            prompt: User prompt/message to send to Claude Haiku
            max_tokens: Maximum tokens in response (default: 2000)
            temperature: Sampling temperature 0.0-1.0 (default: 0.7)
            system_prompt: Optional system prompt for context (auto-added for Marathi)
        
        Returns:
            Generated text response from Claude Haiku (or mock response)
        
        Raises:
            BedrockError: If API call fails after all retries
            AuthenticationError: If credentials are invalid
        
        Requirements:
            - 1.1: Use Claude 3 Haiku model
            - 1.3: Log errors for AWS operations
            - 9.2: Exponential backoff retry logic for rate limits
            - 9.3: Retry up to 3 times for network errors
        """
        # MOCK MODE: Return mock response immediately without calling AWS
        if self.mock_mode:
            self.logger.info("🎭 MOCK MODE: Generating mock response in Marathi")
            time.sleep(0.5)  # Simulate API delay
            return self._get_mock_response(prompt)
        
        # REAL MODE: Use actual AWS Bedrock with Converse API
        client = self.get_bedrock_client()
        
        # Add default system prompt for Marathi analysis if not provided
        if system_prompt is None:
            system_prompt = """You are a financial analyst providing stock market analysis in Marathi language (मराठी).

Your analysis should:
1. Use proper Marathi financial terminology
2. Be clear and professional
3. Include both technical and fundamental insights
4. Provide actionable recommendations
5. Use Devanagari script for Marathi text

Format your response with clear sections and maintain a professional tone suitable for Indian investors."""
        
        # Construct messages for Converse API
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
        
        # Construct system messages (Claude supports system prompts)
        system_messages = []
        if system_prompt:
            system_messages.append({
                "text": system_prompt
            })
        
        # Configure inference parameters optimized for Claude Haiku
        inference_config = {
            "maxTokens": max_tokens,
            "temperature": temperature,
            "topP": 0.9
        }
        
        # Retry loop with exponential backoff
        last_exception = None
        for attempt in range(self.MAX_RETRIES):
            try:
                self.logger.debug(
                    f"Invoking Claude Haiku model (attempt {attempt + 1}/{self.MAX_RETRIES}) | "
                    f"Model: {self.model_id} | Prompt length: {len(prompt)} chars"
                )
                
                # Call Converse API
                converse_params = {
                    "modelId": self.model_id,
                    "messages": messages,
                    "inferenceConfig": inference_config
                }
                
                # Add system messages if provided
                if system_messages:
                    converse_params["system"] = system_messages
                
                response = client.converse(**converse_params)
                
                # Extract text from Claude's response
                if 'output' in response and 'message' in response['output']:
                    message = response['output']['message']
                    if 'content' in message and len(message['content']) > 0:
                        # Claude returns content as a list of content blocks
                        generated_text = message['content'][0].get('text', '')
                        
                        self.logger.info(
                            f"Claude Haiku invocation successful | "
                            f"Response length: {len(generated_text)} chars | "
                            f"Attempts: {attempt + 1}"
                        )
                        
                        return generated_text
                    else:
                        raise BedrockError(
                            "Invalid response format from Claude: missing content"
                        )
                else:
                    raise BedrockError(
                        "Invalid response format from Claude: missing output/message"
                    )
                
            except ClientError as e:
                error_code = e.response.get('Error', {}).get('Code', '')
                error_message = e.response.get('Error', {}).get('Message', str(e))
                
                log_error_with_context(
                    self.logger,
                    e,
                    f"Claude Haiku invocation (attempt {attempt + 1}/{self.MAX_RETRIES})",
                    {
                        "error_code": error_code,
                        "model_id": self.model_id,
                        "attempt": attempt + 1
                    }
                )
                
                # Check if error is retryable
                retryable_errors = [
                    'ThrottlingException',
                    'TooManyRequestsException',
                    'ServiceUnavailableException',
                    'InternalServerException',
                    'ModelTimeoutException'
                ]
                
                if error_code in retryable_errors and attempt < self.MAX_RETRIES - 1:
                    # Calculate exponential backoff delay
                    delay = self.INITIAL_RETRY_DELAY * (2 ** attempt)
                    self.logger.warning(
                        f"Retryable error encountered: {error_code}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    last_exception = e
                    continue
                
                # Non-retryable error or max retries reached
                if error_code in ['InvalidClientTokenId', 'SignatureDoesNotMatch']:
                    raise AuthenticationError(
                        f"Invalid AWS credentials: {error_message}"
                    ) from e
                
                raise BedrockError(
                    f"Bedrock API error after {attempt + 1} attempts: "
                    f"{error_code} - {error_message}"
                ) from e
                
            except (json.JSONDecodeError, BotoCoreError, KeyError) as e:
                log_error_with_context(
                    self.logger,
                    e,
                    f"Claude Haiku invocation (attempt {attempt + 1}/{self.MAX_RETRIES})",
                    {"model_id": self.model_id, "attempt": attempt + 1}
                )
                
                # Retry on network/parsing errors
                if attempt < self.MAX_RETRIES - 1:
                    delay = self.INITIAL_RETRY_DELAY * (2 ** attempt)
                    self.logger.warning(
                        f"Network/parsing error encountered. Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                    last_exception = e
                    continue
                
                raise BedrockError(
                    f"Failed to invoke Claude Haiku after {attempt + 1} attempts: {str(e)}"
                ) from e
        
        # If we get here, all retries failed
        raise BedrockError(
            f"Failed to invoke Claude Haiku after {self.MAX_RETRIES} attempts"
        ) from last_exception
