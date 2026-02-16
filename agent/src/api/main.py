from __future__ import annotations
"""
FastAPI Application for Cortex DeFi Agent.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import structlog

from .routes import predictions, execution, health, solana, risk
from ..config import settings

logger = structlog.get_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info("Starting Cortex DeFi Agent API")

    # Initialize Redis persistence and restore saved A-LAMS-VaR models
    try:
        from cortex.persistence import init_persistence
        await init_persistence()
    except Exception:
        logger.warning("persistence_init_skipped", exc_info=True)

    try:
        from ..models.risk import ALAMSVaRModel
        n_restored = await ALAMSVaRModel.restore_all()
        if n_restored > 0:
            logger.info("alams_models_restored", count=n_restored)
    except Exception:
        logger.warning("alams_restore_skipped", exc_info=True)

    yield

    # Shutdown
    logger.info("Shutting down Cortex DeFi Agent API")
    try:
        from cortex.persistence import close_persistence
        await close_persistence()
    except Exception:
        pass


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    app = FastAPI(
        title="Cortex DeFi Agent",
        description="XGBoost-based DeFi trading agent with explainable predictions",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(health.router, tags=["Health"])
    app.include_router(predictions.router, prefix="/api/v1", tags=["Predictions"])
    app.include_router(execution.router, prefix="/api/v1", tags=["Execution"])
    app.include_router(solana.router, prefix="/api/v1", tags=["Solana"])
    app.include_router(risk.router, prefix="/api/v1", tags=["Risk"])

    return app


app = create_app()

