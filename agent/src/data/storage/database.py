from __future__ import annotations
"""
PostgreSQL/TimescaleDB storage for time-series data.
"""
import json
from datetime import datetime
from typing import Any
import asyncpg
import structlog
import pandas as pd

from ...config import settings

logger = structlog.get_logger()


class Database:
    """
    PostgreSQL/TimescaleDB client for storing DeFi data.
    
    Stores:
    - Historical price data
    - Lending rates
    - LP pool metrics
    - Trade history
    - Model predictions
    """
    
    def __init__(self, database_url: str | None = None):
        self.database_url = database_url or settings.database_url
        self.pool: asyncpg.Pool | None = None
        self.logger = logger.bind(component="database")
    
    async def connect(self) -> None:
        """Create connection pool."""
        if not self.database_url:
            self.logger.warning("No database URL configured")
            return
        
        self.pool = await asyncpg.create_pool(
            self.database_url,
            min_size=2,
            max_size=10
        )
        self.logger.info("Database connected")
    
    async def close(self) -> None:
        """Close connection pool."""
        if self.pool:
            await self.pool.close()
            self.logger.info("Database connection closed")
    
    async def init_tables(self) -> None:
        """Create tables if they don't exist."""
        if not self.pool:
            return
        
        async with self.pool.acquire() as conn:
            # DEX price data (hypertable for TimescaleDB)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS dex_prices (
                    timestamp TIMESTAMPTZ NOT NULL,
                    pool_id VARCHAR(66) NOT NULL,
                    pool_name VARCHAR(50),
                    dex VARCHAR(20),
                    price DECIMAL(30, 18),
                    volume_usd DECIMAL(30, 2),
                    liquidity_usd DECIMAL(30, 2),
                    PRIMARY KEY (timestamp, pool_id)
                );
            """)
            
            # Lending rates
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS lending_rates (
                    timestamp TIMESTAMPTZ NOT NULL,
                    protocol VARCHAR(20) NOT NULL,
                    asset VARCHAR(20) NOT NULL,
                    supply_apy DECIMAL(10, 4),
                    borrow_apy DECIMAL(10, 4),
                    utilization DECIMAL(10, 4),
                    PRIMARY KEY (timestamp, protocol, asset)
                );
            """)
            
            # LP pool data
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS lp_pools (
                    timestamp TIMESTAMPTZ NOT NULL,
                    pool_name VARCHAR(50) NOT NULL,
                    pool_address VARCHAR(66),
                    tvl_usd DECIMAL(30, 2),
                    base_apy DECIMAL(10, 4),
                    reward_apy DECIMAL(10, 4),
                    volume_24h DECIMAL(30, 2),
                    PRIMARY KEY (timestamp, pool_name)
                );
            """)
            
            # Gas prices
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS gas_prices (
                    timestamp TIMESTAMPTZ NOT NULL PRIMARY KEY,
                    safe_gas DECIMAL(10, 2),
                    standard_gas DECIMAL(10, 2),
                    fast_gas DECIMAL(10, 2),
                    base_fee DECIMAL(10, 4)
                );
            """)
            
            # Model predictions
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    strategy VARCHAR(20) NOT NULL,
                    prediction DECIMAL(10, 6),
                    confidence DECIMAL(10, 6),
                    features JSONB,
                    executed BOOLEAN DEFAULT FALSE,
                    profit_realized DECIMAL(20, 8)
                );
            """)
            
            self.logger.info("Database tables initialized")
    
    async def insert_dex_price(
        self,
        timestamp: datetime,
        pool_id: str,
        pool_name: str,
        dex: str,
        price: float,
        volume_usd: float,
        liquidity_usd: float
    ) -> None:
        """Insert DEX price data."""
        if not self.pool:
            return
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO dex_prices 
                (timestamp, pool_id, pool_name, dex, price, volume_usd, liquidity_usd)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (timestamp, pool_id) DO UPDATE SET
                    price = EXCLUDED.price,
                    volume_usd = EXCLUDED.volume_usd,
                    liquidity_usd = EXCLUDED.liquidity_usd
            """, timestamp, pool_id, pool_name, dex, price, volume_usd, liquidity_usd)
    
    async def insert_lending_rate(
        self,
        timestamp: datetime,
        protocol: str,
        asset: str,
        supply_apy: float,
        borrow_apy: float,
        utilization: float
    ) -> None:
        """Insert lending rate data."""
        if not self.pool:
            return
        
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO lending_rates 
                (timestamp, protocol, asset, supply_apy, borrow_apy, utilization)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (timestamp, protocol, asset) DO UPDATE SET
                    supply_apy = EXCLUDED.supply_apy,
                    borrow_apy = EXCLUDED.borrow_apy,
                    utilization = EXCLUDED.utilization
            """, timestamp, protocol, asset, supply_apy, borrow_apy, utilization)

    async def get_historical_prices(
        self,
        pool_id: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Get historical price data for a pool."""
        if not self.pool:
            return pd.DataFrame()

        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM dex_prices
                WHERE pool_id = $1
                AND timestamp >= $2
                AND timestamp <= $3
                ORDER BY timestamp ASC
            """, pool_id, start_time, end_time)

            return pd.DataFrame([dict(r) for r in rows])

    async def get_historical_rates(
        self,
        protocol: str,
        asset: str,
        start_time: datetime,
        end_time: datetime
    ) -> pd.DataFrame:
        """Get historical lending rates."""
        if not self.pool:
            return pd.DataFrame()

        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM lending_rates
                WHERE protocol = $1
                AND asset = $2
                AND timestamp >= $3
                AND timestamp <= $4
                ORDER BY timestamp ASC
            """, protocol, asset, start_time, end_time)

            return pd.DataFrame([dict(r) for r in rows])

    async def save_prediction(
        self,
        strategy: str,
        prediction: float,
        confidence: float,
        features: dict[str, Any]
    ) -> int:
        """Save a model prediction."""
        if not self.pool:
            return -1

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow("""
                INSERT INTO predictions
                (strategy, prediction, confidence, features)
                VALUES ($1, $2, $3, $4)
                RETURNING id
            """, strategy, prediction, confidence, json.dumps(features))

            return row["id"] if row else -1

    async def update_prediction_result(
        self,
        prediction_id: int,
        executed: bool,
        profit_realized: float
    ) -> None:
        """Update prediction with execution result."""
        if not self.pool:
            return

        async with self.pool.acquire() as conn:
            await conn.execute("""
                UPDATE predictions
                SET executed = $2, profit_realized = $3
                WHERE id = $1
            """, prediction_id, executed, profit_realized)
