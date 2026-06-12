import uuid
from datetime import datetime
from sqlalchemy import String, DateTime, JSON
from sqlalchemy.orm import Mapped, mapped_column
from database import Base
from sqlalchemy.dialects.postgresql import UUID

class Person(Base):
    __tablename__ = "persons"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4())
    name: Mapped[str] = mapped_column(String, nullable=False)
    notes: Mapped[str | None] = mapped_column(String, nullable=True)
    plateNumbers: Mapped[list] = mapped_column(JSON, default=list)
    faceEmbeddings: Mapped[list] = mapped_column(JSON, default=list)
    faceThumbnail: Mapped[str | None] = mapped_column(String, nullable=True)  # base64 JPEG crop of enrolled face
    createdAt: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
