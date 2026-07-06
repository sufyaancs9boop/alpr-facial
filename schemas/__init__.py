from pydantic import BaseModel
from typing import List, Optional


class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float
    rotation: Optional[float] = None


class Plate(BaseModel):
    text: str
    confidence: float
    quality: float
    bounding_box: BoundingBox
    thumbnail: Optional[str] = None
    direction: Optional[str] = None
    vehicle_make: Optional[str] = None
    vehicle_model: Optional[str] = None
    vehicle_color: Optional[str] = None
    vehicle_thumbnail: Optional[str] = None
    person_id: Optional[str] = None
    person_name: Optional[str] = None


class Face(BaseModel):
    confidence: float
    quality: float
    bounding_box: BoundingBox
    thumbnail: Optional[str] = None
    person_id: Optional[str] = None
    person_name: Optional[str] = None
    similarity: Optional[float] = None
    spoof_score: Optional[float] = None
    spoof_detected: Optional[bool] = False
    occluded: Optional[bool] = False


class Vehicle(BaseModel):
    make: Optional[str]
    model: Optional[str]
    color: Optional[str]
    type: Optional[str]
    confidence: float
    bounding_box: BoundingBox
    thumbnail: Optional[str] = None


class DetectionResponse(BaseModel):
    success: bool
    count: int
    plates: List[Plate]
    faces: List[Face]
    vehicles: List[Vehicle]
    processingTimeMs: int
    gunDetected: Optional[bool] = False


class DetectURLRequest(BaseModel):
    imageUrl: str
    thumbnail: Optional[bool] = True
    cameraRegion: Optional[str] = None


class DetectStreamRequest(BaseModel):
    url: str
    frameStep: Optional[int] = 5
    cameraRegion: Optional[str] = None


class PersonCreate(BaseModel):
    name: str
    notes: Optional[str] = None
    plateNumbers: Optional[List[str]] = []


class PersonUpdate(BaseModel):
    name: Optional[str] = None
    notes: Optional[str] = None
    plateNumbers: Optional[List[str]] = None


class PersonOut(BaseModel):
    id: str
    name: str
    notes: Optional[str]
    plateNumbers: List[str]
    faceCount: int
    faceThumbnail: Optional[str]
    createdAt: str


class CameraCreate(BaseModel):
    name: str
    url: str
    region: Optional[str] = "PAKISTAN"
    frameStep: Optional[int] = 5
    active: Optional[bool] = True
    lat: Optional[float] = None
    lng: Optional[float] = None
    zone: Optional[str] = None
    notes: Optional[str] = None
    roiInclude: Optional[dict] = None
    roiExclude: Optional[dict] = None


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    region: Optional[str] = None
    frameStep: Optional[int] = None
    active: Optional[bool] = None
    lat: Optional[float] = None
    lng: Optional[float] = None
    zone: Optional[str] = None
    notes: Optional[str] = None
    roiInclude: Optional[dict] = None
    roiExclude: Optional[dict] = None


class CameraOut(BaseModel):
    id: str
    name: str
    url: str
    region: str
    frameStep: int
    active: bool
    streaming: bool
    lat: Optional[float]
    lng: Optional[float]
    zone: Optional[str]
    notes: Optional[str]
    roiInclude: Optional[dict]
    roiExclude: Optional[dict]
    testVideoPath: Optional[str]
    createdAt: str


class WatchlistCreate(BaseModel):
    plateText: str
    reason: Optional[str] = None
    active: Optional[bool] = True


class WatchlistUpdate(BaseModel):
    plateText: Optional[str] = None
    reason: Optional[str] = None
    active: Optional[bool] = None


class WatchlistOut(BaseModel):
    id: str
    plateText: str
    reason: Optional[str]
    active: bool
    createdAt: str
