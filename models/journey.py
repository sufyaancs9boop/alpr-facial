import uuid
from datetime import datetime, timezone
from sqlalchemy import Float, String, DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship
from database import Base
from sqlalchemy.dialects.postgresql import UUID

class Journey(Base):
    __tablename__ = "journeys"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4())
    plateText: Mapped[str] = mapped_column(String, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, default="active")  # active|closed
    startedAt: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
    lastSeenAt: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)

    sightings: Mapped[list["JourneySighting"]] = relationship(
        "JourneySighting", back_populates="journey", lazy="select"
    )


class JourneySighting(Base):
    __tablename__ = "journey_sightings"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    journeyId: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("journeys.id"), nullable=False)
    detectionEventId: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    cameraId: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    cameraName: Mapped[str | None] = mapped_column(String, nullable=True)
    thumbnailBase64: Mapped[str | None] = mapped_column(String, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0)
    timestamp: Mapped[datetime] = mapped_column(
    DateTime(timezone=True),  # Store timezone info
    default=lambda: datetime.now(timezone.utc)  # UTC timezone
)

    journey: Mapped["Journey"] = relationship("Journey", back_populates="sightings")
