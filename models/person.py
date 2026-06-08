import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column
from database import Base


class Person(Base):
    __tablename__ = "persons"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String, nullable=False)
    notes: Mapped[str | None] = mapped_column(String, nullable=True)
    plateNumbers: Mapped[list] = mapped_column(JSON, default=list)
    faceEmbeddings: Mapped[list] = mapped_column(JSON, default=list)
    faceThumbnail: Mapped[str | None] = mapped_column(String, nullable=True)  # base64 JPEG crop of enrolled face
    createdAt: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
