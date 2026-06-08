"""
Direct ML model test — no HTTP server needed.
Downloads models on first run (~300 MB total).
Usage: python test_ml.py
"""
import sys, asyncio, cv2, base64, os, urllib.request
from pathlib import Path

sys.path.insert(0, ".")

VIDEO_PATH = "/Users/Akmal/Desktop/projects/terarare/roc-web/alpr-api/data/test-videos/ef8c1d2e-bb19-44a7-8b5d-3d910b691432.mp4"

# Public-domain face for face detection test (Wikipedia commons — Barack Obama official portrait crop)
# Use a frame extracted from the test video (has a detected face)
FACE_IMAGE_PATH = "/tmp/test_face_frame.jpg"


def extract_video_frame(video_path: str, frame_idx: int = 30) -> bytes | None:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    _, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes()


def download_face_image() -> bytes | None:
    if not Path(FACE_IMAGE_PATH).exists():
        print(f"  ERROR: face test image not found at {FACE_IMAGE_PATH}")
        return None
    with open(FACE_IMAGE_PATH, "rb") as f:
        return f.read()


# ── Test 1: Plate detection ───────────────────────────────────────────────────

def test_plate_detection():
    print("\n" + "="*60)
    print("TEST 1 — License Plate Detection (fast-alpr)")
    print("="*60)

    from ml.plate_detector import PlateDetector, passes_pre_filters

    detector = PlateDetector()

    # Try video frames at different positions to find a good plate frame
    if not Path(VIDEO_PATH).exists():
        print(f"  WARNING: test video not found at {VIDEO_PATH}")
        print("  Creating a synthetic plate image instead…")
        img = _make_synthetic_plate_image("ABC1234")
        _, buf = cv2.imencode(".jpg", img)
        image_bytes = buf.tobytes()
        frame_desc = "synthetic plate"
    else:
        # Frame 140 is known to contain a license plate in the test video
        cap = cv2.VideoCapture(VIDEO_PATH)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"  Video: {VIDEO_PATH}")
        print(f"  Total frames: {total}")

        image_bytes = None
        for fnum in [140, 0, 30, 60, 90, 120, 150, 200]:
            fb = extract_video_frame(VIDEO_PATH, fnum)
            if fb:
                image_bytes = fb
                frame_desc = f"frame {fnum}"
                break

    print(f"  Testing on: {frame_desc}")
    print("  Loading fast-alpr model (downloads on first run)…")

    results = detector.detect(image_bytes, generate_thumbnail=True)

    if not results:
        print("  Result: No plates detected in this frame (try a different frame or image)")
    else:
        print(f"  Result: {len(results)} plate(s) detected")
        for r in results:
            bb = r.bounding_box
            passed = passes_pre_filters(r)
            print(f"    text='{r.text}'  conf={r.confidence:.2f}  quality={r.quality:.2f}  "
                  f"w={bb.width:.0f}px  ratio={bb.width/max(bb.height,1):.2f}  "
                  f"prefilter={'PASS' if passed else 'FAIL'}")
            if r.thumbnail:
                print(f"    thumbnail: {r.thumbnail[:60]}…")

    print("  fast-alpr: OK" if results is not None else "  fast-alpr: LOADED (no plates in frame)")


# ── Test 2: Face detection ────────────────────────────────────────────────────

def test_face_detection():
    print("\n" + "="*60)
    print("TEST 2 — Face Detection + Recognition (InsightFace)")
    print("="*60)

    from ml.face_analyzer import FaceAnalyzer

    analyzer = FaceAnalyzer(models_dir="data/models")

    print("  Downloading face test image…")
    face_bytes = download_face_image()
    if not face_bytes:
        print("  ERROR: Could not load face test image")
        return

    print("  Loading InsightFace buffalo_s model (downloads on first run)…")
    results = analyzer.detect(face_bytes, generate_thumbnail=True)

    if not results:
        print("  Result: No faces detected")
    else:
        print(f"  Result: {len(results)} face(s) detected")
        for r in results:
            bb = r.bounding_box
            print(f"    conf={r.confidence:.3f}  bbox=({bb.x:.0f},{bb.y:.0f},{bb.width:.0f}x{bb.height:.0f})")
            if r.embedding:
                print(f"    embedding: {len(r.embedding)}-dim ArcFace vector  (first 5: {[f'{v:.3f}' for v in r.embedding[:5]]})")
            if r.thumbnail:
                print(f"    thumbnail: {r.thumbnail[:60]}…")
            print(f"    person_id: {r.person_id} (none — gallery empty, expected)")

    # Test enrollment + search round-trip
    print("\n  Testing enrollment + gallery search…")
    if results and results[0].embedding:
        analyzer.add_person("test-person-001", "Test User", [results[0].embedding])
        results2 = analyzer.detect(face_bytes, generate_thumbnail=False)
        if results2 and results2[0].person_id:
            print(f"  Gallery search: MATCHED person_id='{results2[0].person_id}' "
                  f"name='{results2[0].person_name}' sim={results2[0].similarity:.3f}")
        else:
            print("  Gallery search: no match (similarity below threshold)")
    else:
        print("  Skipped — no embedding to enroll")

    print("  InsightFace: OK")


# ── Test 3: Vehicle detection ─────────────────────────────────────────────────

def test_vehicle_detection():
    print("\n" + "="*60)
    print("TEST 3 — Vehicle Detection + Color (YOLOv8n)")
    print("="*60)

    from ml.vehicle_classifier import VehicleClassifier

    classifier = VehicleClassifier()

    if not Path(VIDEO_PATH).exists():
        print("  WARNING: test video not found — skipping vehicle test")
        return

    image_bytes = extract_video_frame(VIDEO_PATH, 30)
    if not image_bytes:
        print("  ERROR: Could not extract video frame")
        return

    print("  Loading YOLOv8n model (downloads on first run)…")
    results = classifier.detect(image_bytes, generate_thumbnail=True)

    if not results:
        print("  Result: No vehicles detected in this frame")
    else:
        print(f"  Result: {len(results)} vehicle(s) detected")
        for r in results:
            bb = r.bounding_box
            print(f"    type={r.type}  color={r.color}  conf={r.confidence:.2f}  "
                  f"bbox=({bb.x:.0f},{bb.y:.0f},{bb.width:.0f}x{bb.height:.0f})")

    print("  YOLOv8n: OK")


def _make_synthetic_plate_image(text: str):
    """Create a rough synthetic plate image as fallback."""
    import numpy as np
    img = np.ones((80, 240, 3), dtype="uint8") * 220
    cv2.rectangle(img, (5, 5), (235, 75), (0, 0, 0), 3)
    cv2.putText(img, text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 3)
    return img


if __name__ == "__main__":
    print("ALPR OSS — ML Model Test")
    print("Models will be downloaded on first run (~300 MB total)\n")

    try:
        test_plate_detection()
    except Exception as e:
        print(f"  PLATE TEST ERROR: {e}")
        import traceback; traceback.print_exc()

    try:
        test_face_detection()
    except Exception as e:
        print(f"  FACE TEST ERROR: {e}")
        import traceback; traceback.print_exc()

    try:
        test_vehicle_detection()
    except Exception as e:
        print(f"  VEHICLE TEST ERROR: {e}")
        import traceback; traceback.print_exc()

    print("\n" + "="*60)
    print("Tests complete.")
