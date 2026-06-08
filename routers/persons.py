import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from dependencies import get_db_factory
from models.person import Person
from services.inference_service import get_face_analyzer

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/persons", tags=["Persons"])


@router.get("/")
async def list_persons():
    db_factory = get_db_factory()
    async with db_factory() as db:
        result = await db.execute(select(Person).order_by(Person.createdAt.desc()))
        return [_to_dict(p) for p in result.scalars()]


@router.get("/{person_id}")
async def get_person(person_id: str):
    db_factory = get_db_factory()
    async with db_factory() as db:
        person = await db.get(Person, person_id)
        if not person:
            raise HTTPException(404, "Person not found")
        return _to_dict(person)


@router.post("/")
async def create_person(body: dict):
    db_factory = get_db_factory()
    async with db_factory() as db:
        person = Person(
            name=body["name"],
            notes=body.get("notes"),
            plateNumbers=body.get("plateNumbers", []),
            faceEmbeddings=[],
        )
        db.add(person)
        await db.commit()
        await db.refresh(person)
        return _to_dict(person)


@router.put("/{person_id}")
async def update_person(person_id: str, body: dict):
    db_factory = get_db_factory()
    async with db_factory() as db:
        person = await db.get(Person, person_id)
        if not person:
            raise HTTPException(404, "Person not found")
        if "name" in body:
            person.name = body["name"]
        if "notes" in body:
            person.notes = body["notes"]
        if "plateNumbers" in body:
            person.plateNumbers = body["plateNumbers"]
        await db.commit()
        await db.refresh(person)
        # Refresh gallery
        _reload_gallery(db_factory)
        return _to_dict(person)


@router.delete("/{person_id}")
async def delete_person(person_id: str):
    db_factory = get_db_factory()
    async with db_factory() as db:
        person = await db.get(Person, person_id)
        if not person:
            raise HTTPException(404, "Person not found")
        await db.delete(person)
        await db.commit()
    get_face_analyzer().remove_person(person_id)
    return {"ok": True}


@router.post("/{person_id}/enroll-face")
async def enroll_face(person_id: str, image: UploadFile = File(...)):
    """Enroll a face image for a person — extracts ArcFace embedding and stores it."""
    db_factory = get_db_factory()
    async with db_factory() as db:
        person = await db.get(Person, person_id)
        if not person:
            raise HTTPException(404, "Person not found")

        image_bytes = await image.read()
        analyzer = get_face_analyzer()

        embedding, thumbnail = analyzer.extract_embedding_and_thumbnail(image_bytes)
        if not embedding:
            raise HTTPException(422, "No face detected in the enrollment image")

        person.faceEmbeddings = (person.faceEmbeddings or []) + [embedding]
        # Always replace thumbnail with the latest enrolled photo
        if thumbnail:
            person.faceThumbnail = thumbnail
        await db.commit()
        await db.refresh(person)

    # Update live gallery
    analyzer.add_person(person_id, person.name, person.faceEmbeddings)
    logger.info("Enrolled face for person %s (%s)", person_id, person.name)
    return {"ok": True, "totalEmbeddings": len(person.faceEmbeddings)}


def _reload_gallery(db_factory):
    """Rebuild the in-memory face gallery from DB (fire-and-forget)."""
    import asyncio
    asyncio.create_task(_async_reload_gallery(db_factory))


async def _async_reload_gallery(db_factory):
    analyzer = get_face_analyzer()
    async with db_factory() as db:
        result = await db.execute(select(Person))
        gallery = {
            p.id: {"name": p.name, "embeddings": p.faceEmbeddings or []}
            for p in result.scalars()
        }
    analyzer.load_gallery(gallery)


def _to_dict(p: Person) -> dict:
    return {
        "id": p.id, "name": p.name, "notes": p.notes,
        "plateNumbers": p.plateNumbers or [],
        "faceCount": len(p.faceEmbeddings or []),
        "faceThumbnail": p.faceThumbnail,
        "createdAt": p.createdAt.isoformat(),
    }
