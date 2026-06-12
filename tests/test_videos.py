import requests
import json
import sys

video_path = sys.argv[1] if len(sys.argv) > 1 else "video1.mp4"

print(f"Uploading {video_path}...")

with open(video_path, "rb") as f:
    files = {"video": f}
    params = {"frameStep": 5}
    
    response = requests.post(
        "http://localhost:3000/api/alpr/detect-video",
        files=files,
        params=params,
        stream=True
    )

print("\n--- DETECTIONS ---\n")

plate_count = 0
vehicle_count = 0
face_count = 0

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith("data: "):
            try:
                data = json.loads(line[6:])
                
                # Count detections
                if data.get("count", 0) > 0:
                    plates = data.get("plates", [])
                    vehicles = data.get("vehicles", [])
                    faces = data.get("faces", [])
                    
                    plate_count += len(plates)
                    vehicle_count += len(vehicles)
                    face_count += len(faces)
                    
                    # Print each detection
                    for plate in plates:
                        print(f"🚗 PLATE: {plate['text']} (confidence: {plate['confidence']:.2f}, quality: {plate['quality']:.2f})")
                    
                    for vehicle in vehicles:
                        color = vehicle.get('color', 'unknown')
                        print(f"   → Vehicle: {color} (confidence: {vehicle['confidence']:.2f})")
                    
                    for face in faces:
                        print(f"👤 FACE (confidence: {face['confidence']:.2f})")
            except json.JSONDecodeError:
                pass

print(f"\n--- SUMMARY ---")
print(f"Total plates detected: {plate_count}")
print(f"Total vehicles detected: {vehicle_count}")
print(f"Total faces detected: {face_count}")