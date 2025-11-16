"""Main entry point for the Hive Mind system."""

import asyncio
import os
import signal
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI

from src.hive_mind.api.main import app, initialize_hive_mind
from src.hive_mind.utils.config_loader import load_config_from_file
from src.hive_mind.utils.logger import setup_logging, get_logger


def setup_signal_handlers():
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger = get_logger("main")
        logger.info("Received shutdown signal, shutting down gracefully...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    # Setup logging
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_file = os.getenv("LOG_FILE")
    setup_logging(level=log_level, log_file=log_file)
    
    logger = get_logger("main")
    logger.info("Starting Hive Mind system")
    
    # Load configuration
    config_path = os.getenv("CONFIG_PATH", "config/default_config.yaml")
    
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found, using default configuration")
        from src.hive_mind.utils.config_loader import create_default_config
        config = create_default_config()
    else:
        logger.info(f"Loading configuration from {config_path}")
        config = load_config_from_file(config_path)
    
    # Initialize Hive Mind
    try:
        initialize_hive_mind(config)
        logger.info(f"Hive Mind initialized with {len(config.get_enabled_models())} models")
        
        # Log enabled models
        for model in config.get_enabled_models():
            logger.info(f"Enabled model: {model.model_id} ({model.provider})")
            
    except Exception as e:
        logger.error(f"Failed to initialize Hive Mind: {e}")
        raise
    
    return app


def main():
    """Main function to run the Hive Mind API server."""
    setup_signal_handlers()
    
    # Create the app
    app = create_app()
    
    # Get server configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    workers = int(os.getenv("WORKERS", 1))
    
    logger = get_logger("main")
    logger.info(f"Starting server on {host}:{port}")
    
    # Run the server
    if workers > 1:
        uvicorn.run(
            "main:app",
            host=host,
            port=port,
            workers=workers,
            log_level="info",
            access_log=True
        )
    else:
        uvicorn.run(
            app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )


if __name__ == "__main__":
    main()