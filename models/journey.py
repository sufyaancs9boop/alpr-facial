import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base


class Journey(Base):
    __tablename__ = "journeys"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    plateText: Mapped[str] = mapped_column(String, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, default="active")  # active|closed
    startedAt: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    lastSeenAt: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    sightings: Mapped[list["JourneySighting"]] = relationship(
        "JourneySighting", back_populates="journey", lazy="select"
    )


class JourneySighting(Base):
    __tablename__ = "journey_sightings"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    journeyId: Mapped[str] = mapped_column(String, ForeignKey("journeys.id"), nullable=False)
    detectionEventId: Mapped[str | None] = mapped_column(String, nullable=True)
    cameraId: Mapped[str | None] = mapped_column(String, nullable=True)
    cameraName: Mapped[str | None] = mapped_column(String, nullable=True)
    thumbnailBase64: Mapped[str | None] = mapped_column(String, nullable=True)
    confidence: Mapped[float] = mapped_column(String, default=0)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    journey: Mapped["Journey"] = relationship("Journey", back_populates="sightings")
