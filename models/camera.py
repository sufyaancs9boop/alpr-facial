import uuid
from datetime import datetime
from sqlalchemy import String, Boolean, DateTime, Integer, Float, JSON
from sqlalchemy.orm import Mapped, mapped_column
from database import Base


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String, nullable=False)
    url: Mapped[str] = mapped_column(String, nullable=False)
    region: Mapped[str] = mapped_column(String, default="NORTH_AMERICAN")
    frameStep: Mapped[int] = mapped_column(Integer, default=5)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lng: Mapped[float | None] = mapped_column(Float, nullable=True)
    zone: Mapped[str | None] = mapped_column(String, nullable=True)
    notes: Mapped[str | None] = mapped_column(String, nullable=True)
    roiInclude: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    roiExclude: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    testVideoPath: Mapped[str | None] = mapped_column(String, nullable=True)
    createdAt: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
