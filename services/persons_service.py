from sqlalchemy import select
from models.person import Person


class PersonsService:
    def __init__(self, db_factory):
        self._db_factory = db_factory

    async def find_by_plate(self, plate_text: str) -> Person | None:
        """Find a person whose plateNumbers list includes this plate text."""
        async with self._db_factory() as db:
            result = await db.execute(select(Person))
            for person in result.scalars():
                if plate_text in (person.plateNumbers or []):
                    return person
        return None

    async def find_one(self, person_id: str) -> Person | None:
        async with self._db_factory() as db:
            return await db.get(Person, person_id)
