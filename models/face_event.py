import uuid
from datetime import datetime
from sqlalchemy import String, Float, Boolean, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from database import Base


class FaceEvent(Base):
    __tablename__ = "face_events"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    personId: Mapped[str | None] = mapped_column(String, nullable=True)
    personName: Mapped[str | None] = mapped_column(String, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0)
    quality: Mapped[float] = mapped_column(Float, default=0)
    spoofScore: Mapped[float | None] = mapped_column(Float, nullable=True)
    spoofDetected: Mapped[bool] = mapped_column(Boolean, default=False)
    occluded: Mapped[bool] = mapped_column(Boolean, default=False)
    thumbnailBase64: Mapped[str | None] = mapped_column(String, nullable=True)
    x: Mapped[float] = mapped_column(Float, default=0)
    y: Mapped[float] = mapped_column(Float, default=0)
    width: Mapped[float] = mapped_column(Float, default=0)
    height: Mapped[float] = mapped_column(Float, default=0)
    cameraId: Mapped[str | None] = mapped_column(String, nullable=True)
    cameraName: Mapped[str | None] = mapped_column(String, nullable=True)
    detectionEventId: Mapped[str | None] = mapped_column(String, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
