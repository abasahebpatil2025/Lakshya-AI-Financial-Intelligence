🎯 Lakshya AI: Advanced Financial Intelligence System

Institutional-Grade Multi-Agent Research Dashboard for NSE/BSE & US Markets
Lakshya AI is a proprietary multi-agent framework designed to automate complex stock market research. Built with 5+ years of trading expertise and powered by Amazon Bedrock (Claude 3.5 Sonnet), it provides intelligent Buy/Sell/Hold recommendations with institutional-grade confidence scores.

🌟 Key Highlights

Bilingual Analysis: Specialized financial reporting in Marathi & English, promoting financial inclusion.

Global Market Reach: Full support for Indian (NSE/BSE) and American (NYSE/NASDAQ) markets.

Production-Ready UI: A clean, responsive Streamlit interface for real-time visual analysis.

Agentic Logic: Uses a network of specialized AI agents to cross-verify Technical, Fundamental, and Sentiment data.

🏗️ Architecture & Engine

The system employs a pipeline architecture where multiple specialized agents collaborate to deliver a comprehensive investment thesis.

LLM Backbone: Amazon Bedrock (Claude 3.5 Sonnet v2)

Orchestration: Multi-Agent Pipeline (Technical, Fundamental, Risk Management)

Data Source: Real-time integration via yfinance & custom indicators.

Security: Enterprise-grade credential management via AWS Secrets Manager.

    graph TD
    A[User Input/Ticker] --> B[Main Orchestrator]
    B --> C[Stock Data Service]
    C --> D[Technical Agent]
    C --> E[Fundamental Agent]
    C --> F[Sentiment Agent]
    D & E & F --> G[Risk Manager]
    G --> H[Final Marathi/English Report]

## 🤖 Specialized Research Modules

| Agent | Expertise | Deliverables |
| :--- | :--- | :--- |
| **Technical Agent** | Price Action & Indicators | Moving Averages (50/200 DMA), RSI, and Trend Analysis. |
| **Fundamental Agent** | Financial Health | P/E Ratios, Market Cap, Earnings Growth, and Debt evaluation. |
| **Sentiment Agent** | Market Pulse | Scans news and global trends (Phase 2 integration). |
| **Risk Manager** | Synthesis & Compliance | Confidence scoring and final recommendation logic. |

🚀 Getting Started

1. Installation
git clone https://github.com/abasahebpatil2025/Lakshya-AI-Financial-Intelligence.git
cd lakshya-ai
pip install -r requirements.txt

3. Configuration
Create a .env file from the template and configure your AWS credentials:
cp .env.example .env
# Add your AWS_ACCESS_KEY, AWS_SECRET_KEY, and REGION

3. Launch the Dashboard
streamlit run app.py

📋 Professional Roadmap (Research Analyst Vision)

Phase 1: Multi-agent core with Marathi/English reporting (Current).

Phase 2: Real-time sentiment integration via NewsAPI & Twitter.

Phase 3: Portfolio-level analysis and SEBI-compliant reporting formats.

Phase 4: Backtesting engine to calibrate agentic confidence scores.

📁 Project Structure

app.py: The Main Web Interface.

agents/: Core logic for Technical, Fundamental, and Risk agents.

services/: API wrappers for market data.

models/: Data models and schema definitions.

diagrams/: High-resolution architecture diagrams.

⚠️ Compliance & Disclaimer

Lakshya AI is an analytical research tool.

Not SEBI Registered: This platform provides data-driven analysis for educational purposes, NOT direct financial advice.

Market Risk: All investments are subject to market risks. Conduct independent research before trading.

Proprietary Software: All rights reserved by [Abasaheb Patil].




    
