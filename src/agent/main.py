#!/usr/bin/env python3
"""Main entry point for Meta AI Agent."""

import asyncio
import os
import signal
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger

from src.agent.core import MetaAgent


def setup_logging() -> None:
    """Setup logging configuration."""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    # Remove default handler
    logger.remove()

    # Add console handler
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )

    # Add file handler
    logger.add(
        log_dir / "agent_{time:YYYY-MM-DD}.log",
        rotation="00:00",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )

    logger.info("Logging configured")


def load_config() -> dict:
    """Load configuration from environment."""
    config = {
        "api_base_url": os.getenv("AGENT_API_URL", "http://localhost:8088"),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),
        "telegram_token": os.getenv("TELEGRAM_BOT_TOKEN"),
        "telegram_chat_id": os.getenv("TELEGRAM_CHAT_ID"),
        "db_path": os.getenv("AGENT_DB_PATH", "data/agent_memory.db"),
        "check_interval": int(os.getenv("AGENT_CHECK_INTERVAL", "60")),
    }

    # Validate required settings
    if not config["anthropic_api_key"]:
        logger.warning("ANTHROPIC_API_KEY not set - Claude API calls will fail")

    if not config["telegram_token"] or not config["telegram_chat_id"]:
        logger.warning("Telegram not configured - notifications will be skipped")

    return config


async def main() -> None:
    """Main async entry point."""
    setup_logging()
    logger.info("Starting Meta AI Agent")

    config = load_config()

    # Create agent
    agent = MetaAgent(
        api_base_url=config["api_base_url"],
        anthropic_api_key=config["anthropic_api_key"],
        telegram_token=config["telegram_token"],
        telegram_chat_id=config["telegram_chat_id"],
        db_path=config["db_path"],
        check_interval=config["check_interval"],
    )

    # Setup signal handlers for graceful shutdown
    loop = asyncio.get_event_loop()

    def handle_signal(sig: int) -> None:
        logger.info(f"Received signal {sig}, shutting down...")
        asyncio.create_task(agent.stop())

    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))

    # Run agent
    try:
        await agent.run()
    except Exception as e:
        logger.exception(f"Agent crashed: {e}")
        raise
    finally:
        logger.info("Agent shutdown complete")


def run() -> None:
    """Synchronous entry point for console script."""
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    run()
