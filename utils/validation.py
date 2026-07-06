from fastapi import HTTPException
import cv2
import numpy as np


def check_image_dimensions(image_bytes: bytes, max_width: int = 4096, max_height: int = 4096):
    # Decode image from bytes without writing to disk
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image data")
    h, w = img.shape[:2]
    if w > max_width or h > max_height:
        raise HTTPException(status_code=413, detail=f"Image resolution {w}x{h} exceeds limit {max_width}x{max_height}")


def check_video_size(video_bytes: bytes, max_size_bytes: int = 500 * 1024 * 1024):
    if len(video_bytes) > max_size_bytes:
        raise HTTPException(status_code=413, detail=f"Video exceeds maximum size of {max_size_bytes} bytes")
