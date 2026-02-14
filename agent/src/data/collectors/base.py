"""
Base collector interface for all data sources.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional
import httpx
import structlog
from pydantic import BaseModel

logger = structlog.get_logger()


class CollectorConfig(BaseModel):
    """Configuration for a data collector."""
    name: str
    base_url: str
    api_key: str | None = None
    timeout: float = 30.0
    max_retries: int = 3


class BaseCollector(ABC):
    """
    Abstract base class for all data collectors.
    
    All collectors must implement:
    - fetch_latest(): Get current/latest data
    - fetch_historical(): Get historical data for a time range
    - validate_response(): Validate API response
    """
    
    def __init__(self, config: CollectorConfig):
        self.config = config
        self.client = httpx.AsyncClient(timeout=config.timeout)
        self.logger = logger.bind(collector=config.name)
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    @abstractmethod
    async def fetch_latest(self) -> dict[str, Any]:
        """
        Fetch the latest/current data.
        
        Returns:
            dict containing the latest data from the source.
        """
        pass
    
    @abstractmethod
    async def fetch_historical(
        self,
        start_time: datetime,
        end_time: datetime,
        interval: str = "5m"
    ) -> list[dict[str, Any]]:
        """
        Fetch historical data for a time range.
        
        Args:
            start_time: Start of the time range
            end_time: End of the time range
            interval: Data interval (e.g., "5m", "1h", "1d")
            
        Returns:
            List of data points for the time range.
        """
        pass
    
    @abstractmethod
    def validate_response(self, response: dict[str, Any]) -> bool:
        """
        Validate the API response.
        
        Args:
            response: Raw API response
            
        Returns:
            True if response is valid, False otherwise.
        """
        pass
    
    async def _make_request(
        self,
        method: str,
        url: str,
        **kwargs
    ) -> dict[str, Any]:
        """
        Make an HTTP request with retry logic.
        
        Args:
            method: HTTP method (GET, POST)
            url: Request URL
            **kwargs: Additional request parameters
            
        Returns:
            Parsed JSON response
            
        Raises:
            httpx.HTTPError: If request fails after retries
        """
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                if method.upper() == "GET":
                    response = await self.client.get(url, **kwargs)
                elif method.upper() == "POST":
                    response = await self.client.post(url, **kwargs)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPError as e:
                last_error = e
                self.logger.warning(
                    "Request failed, retrying",
                    attempt=attempt + 1,
                    max_retries=self.config.max_retries,
                    error=str(e)
                )
        
        self.logger.error("All retries exhausted", error=str(last_error))
        raise last_error

