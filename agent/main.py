#!/usr/bin/env python3
"""
Cortex AI Agent - Main Entry Point

XGBoost-based DeFi yield optimization agent for Solana.
"""
import structlog
from src.config import settings

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
)

logger = structlog.get_logger()


def main():
    """Run the Cortex AI API server."""
    import uvicorn
    from src.api.main import create_app

    logger.info(
        "Starting Cortex AI Agent",
        host=settings.api_host,
        port=settings.api_port,
    )

    app = create_app()
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
