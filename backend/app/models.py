from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlmodel import Field, SQLModel


class SessionRecord(SQLModel, table=True):
    id: str = Field(primary_key=True)
    style: str
    group_name: Optional[str] = Field(default=None)
    consented: bool = Field(default=False)
    accent: Optional[str] = Field(default=None)
    notes: Optional[str] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AnswerRecord(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key="sessionrecord.id")
    turn: int
    answer: str
    style: str
    group_name: Optional[str] = Field(default=None)
    speaking_rate: Optional[float] = Field(default=None)
    pause_ratio: Optional[float] = Field(default=None)
    gaze: Optional[float] = Field(default=None)
    fillers: Optional[int] = Field(default=None)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class TelemetryRecord(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key="sessionrecord.id")
    event_type: str
    latency_ms: Optional[float] = Field(default=None)
    group_name: Optional[str] = Field(default=None)
    payload: Optional[str] = Field(default=None)  # JSON string of extra context
    created_at: datetime = Field(default_factory=datetime.utcnow)


class CheckInRecord(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: str = Field(foreign_key="sessionrecord.id")
    group_name: Optional[str] = Field(default=None)
    confidence: int
    stress: int
    created_at: datetime = Field(default_factory=datetime.utcnow)
