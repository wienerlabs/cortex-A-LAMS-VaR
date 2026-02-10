import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.routes import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("msm_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("MSM-VaR API starting up")
    yield
    logger.info("MSM-VaR API shutting down")


app = FastAPI(
    title="MSM-VaR API",
    description="Markov Switching Multifractal volatility model — regime detection, VaR, and forecasting for Solana DeFi and traditional markets.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1", tags=["msm"])


@app.get("/health")
def health():
    return {"status": "ok", "service": "msm-var-api"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)

