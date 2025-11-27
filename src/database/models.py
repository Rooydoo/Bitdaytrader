"""SQLite database models."""

from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

Base = declarative_base()


class Trade(Base):
    """Trade history table."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # LONG or SHORT
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=True)
    size = Column(Float, nullable=False)
    pnl = Column(Float, nullable=True)
    pnl_percent = Column(Float, nullable=True)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime, nullable=True)
    status = Column(String(20), nullable=False)  # OPEN, CLOSED, STOPPED
    stop_loss = Column(Float, nullable=True)
    take_profit_1 = Column(Float, nullable=True)
    take_profit_2 = Column(Float, nullable=True)
    take_profit_3 = Column(Float, nullable=True)
    confidence = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Signal(Base):
    """Signal history table."""

    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False)
    direction = Column(Integer, nullable=False)  # 1 for up, 0 for down
    confidence = Column(Float, nullable=False)
    price = Column(Float, nullable=False)
    features = Column(Text, nullable=True)  # JSON string of features
    executed = Column(Boolean, default=False)
    reason = Column(String(200), nullable=True)  # Why not executed
    created_at = Column(DateTime, default=datetime.utcnow)


class DailyPnL(Base):
    """Daily PnL summary table."""

    __tablename__ = "daily_pnl"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False, unique=True)  # YYYY-MM-DD
    trades = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    gross_pnl = Column(Float, default=0.0)
    fees = Column(Float, default=0.0)
    net_pnl = Column(Float, default=0.0)
    win_rate = Column(Float, default=0.0)
    capital_start = Column(Float, nullable=True)
    capital_end = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ModelPerformance(Base):
    """Model performance tracking table."""

    __tablename__ = "model_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String(10), nullable=False)
    symbol = Column(String(20), nullable=False)
    predictions = Column(Integer, default=0)
    correct = Column(Integer, default=0)
    accuracy = Column(Float, default=0.0)
    avg_confidence = Column(Float, default=0.0)
    model_version = Column(String(50), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# Database initialization
_engine = None
_SessionLocal = None


def init_db(db_path: str = "data/trading.db") -> Session:
    """
    Initialize database and return session.

    Args:
        db_path: Path to SQLite database file

    Returns:
        SQLAlchemy Session
    """
    global _engine, _SessionLocal

    # Create directory if needed
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Create engine
    _engine = create_engine(
        f"sqlite:///{db_path}",
        connect_args={"check_same_thread": False},
        echo=False,
    )

    # Create tables
    Base.metadata.create_all(_engine)

    # Create session factory
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=_engine)

    return _SessionLocal()


def get_session() -> Session:
    """Get database session."""
    global _SessionLocal

    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    return _SessionLocal()


class TradeRepository:
    """Repository for trade operations."""

    def __init__(self, session: Session) -> None:
        """Initialize repository."""
        self.session = session

    def create(self, trade_data: dict[str, Any]) -> Trade:
        """Create a new trade record."""
        trade = Trade(**trade_data)
        self.session.add(trade)
        self.session.commit()
        return trade

    def update(self, trade_id: int, update_data: dict[str, Any]) -> Trade | None:
        """Update a trade record."""
        trade = self.session.query(Trade).filter(Trade.id == trade_id).first()
        if trade:
            for key, value in update_data.items():
                setattr(trade, key, value)
            self.session.commit()
        return trade

    def get_by_id(self, trade_id: int) -> Trade | None:
        """Get trade by ID."""
        return self.session.query(Trade).filter(Trade.id == trade_id).first()

    def get_open_trades(self) -> list[Trade]:
        """Get all open trades."""
        return self.session.query(Trade).filter(Trade.status == "OPEN").all()

    def get_trades_by_date(self, date: str) -> list[Trade]:
        """Get trades for a specific date."""
        return (
            self.session.query(Trade)
            .filter(Trade.entry_time >= f"{date} 00:00:00")
            .filter(Trade.entry_time < f"{date} 23:59:59")
            .all()
        )

    def get_trades_by_period(self, start_date: str, end_date: str) -> list[Trade]:
        """Get trades for a date range."""
        return (
            self.session.query(Trade)
            .filter(Trade.entry_time >= f"{start_date} 00:00:00")
            .filter(Trade.entry_time <= f"{end_date} 23:59:59")
            .all()
        )


class DailyPnLRepository:
    """Repository for daily PnL operations."""

    def __init__(self, session: Session) -> None:
        """Initialize repository."""
        self.session = session

    def get_or_create(self, date: str) -> DailyPnL:
        """Get or create daily PnL record."""
        record = self.session.query(DailyPnL).filter(DailyPnL.date == date).first()
        if not record:
            record = DailyPnL(date=date)
            self.session.add(record)
            self.session.commit()
        return record

    def update(self, date: str, data: dict[str, Any]) -> DailyPnL:
        """Update daily PnL record."""
        record = self.get_or_create(date)
        for key, value in data.items():
            setattr(record, key, value)
        self.session.commit()
        return record

    def get_by_period(self, start_date: str, end_date: str) -> list[DailyPnL]:
        """Get daily PnL for a date range."""
        return (
            self.session.query(DailyPnL)
            .filter(DailyPnL.date >= start_date)
            .filter(DailyPnL.date <= end_date)
            .order_by(DailyPnL.date)
            .all()
        )

    def get_last_n_days(self, n: int) -> list[DailyPnL]:
        """Get last N days of PnL data."""
        return (
            self.session.query(DailyPnL)
            .order_by(DailyPnL.date.desc())
            .limit(n)
            .all()
        )
