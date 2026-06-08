import uuid
from datetime import datetime
from sqlalchemy import String, Float, Boolean, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from database import Base


class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    plateText: Mapped[str] = mapped_column(String, nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    personId: Mapped[str | None] = mapped_column(String, nullable=True)
    personName: Mapped[str | None] = mapped_column(String, nullable=True)
    source: Mapped[str] = mapped_column(String, default="image")  # image|video|stream|camera
    thumbnailBase64: Mapped[str | None] = mapped_column(String, nullable=True)
    x: Mapped[float] = mapped_column(Float, default=0)
    y: Mapped[float] = mapped_column(Float, default=0)
    width: Mapped[float] = mapped_column(Float, default=0)
    height: Mapped[float] = mapped_column(Float, default=0)
    vehicleMake: Mapped[str | None] = mapped_column(String, nullable=True)
    vehicleModel: Mapped[str | None] = mapped_column(String, nullable=True)
    vehicleColor: Mapped[str | None] = mapped_column(String, nullable=True)
    vehicleThumbnail: Mapped[str | None] = mapped_column(String, nullable=True)
    direction: Mapped[str | None] = mapped_column(String, nullable=True)  # left|right|stationary
    cameraId: Mapped[str | None] = mapped_column(String, nullable=True)
    cameraName: Mapped[str | None] = mapped_column(String, nullable=True)
    gunDetected: Mapped[bool] = mapped_column(Boolean, default=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, index=True)
