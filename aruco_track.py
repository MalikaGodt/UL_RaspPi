#!/usr/bin/env python3
"""
Finds ArUco markers (dictionary 4x4_50) and prints the probe position inside the table square.

  IDs 0-3 = corner markers, placed counter-clockwise as the camera sees them:
      ID 3 ---- ID 2
       |         |
      ID 0 ---- ID 1          (ID 0 = table origin, x towards ID 1, y towards ID 3)
  ID 4    = marker on top of the probe

Usage:
  python3 aruco_track.py --size 320                 camera 0, distance between marker CENTRES = 320 mm
  python3 aruco_track.py --size 320 --cam 1         use /dev/video1
  python3 aruco_track.py --size 320 --image a.jpg   test on a photo instead of a camera
  python3 aruco_track.py --size 320 --udp 5005      also send the position as JSON to localhost:5005

While running with a camera, snapshot.jpg is saved about once per second (open it to check detection).
"""
import argparse
import json
import socket
import time

import cv2
import numpy as np

PROBE_ID = 4

ap = argparse.ArgumentParser()
ap.add_argument("--size", type=float, default=300.0, help="distance between neighbouring corner marker centres, mm")
ap.add_argument("--cam", type=int, default=0)
ap.add_argument("--image", type=str, default=None)
ap.add_argument("--udp", type=int, default=0, help="send position as JSON to this local UDP port")
args = ap.parse_args()
S = args.size

# Corner marker centres in table coordinates (mm)
TABLE = {0: (0, 0), 1: (S, 0), 2: (S, S), 3: (0, S)}

# --- works with both the old (4.6, Raspberry Pi OS bookworm) and new (4.7+) OpenCV ArUco API ---
aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
if hasattr(aruco, "ArucoDetector"):
    _detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
    def detect(img):
        return _detector.detectMarkers(img)
else:
    _params = aruco.DetectorParameters_create()
    def detect(img):
        return aruco.detectMarkers(img, dictionary, parameters=_params)

homography = None
last_print = 0.0
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if args.udp else None


def process(frame):
    """Detect markers, draw on the frame, print/send the probe position."""
    global homography, last_print
    corners, ids, _ = detect(frame)

    centers = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            cx, cy = c[0].mean(axis=0)
            centers[int(i)] = (float(cx), float(cy))
            colour = (0, 0, 255) if i == PROBE_ID else (0, 255, 0)
            cv2.circle(frame, (int(cx), int(cy)), 8, colour, 2)
            cv2.putText(frame, f"ID {i}", (int(cx) + 10, int(cy) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Update the pixel -> table mapping whenever all four corner markers are visible.
    if all(k in centers for k in TABLE):
        src = np.array([centers[k] for k in TABLE], dtype=np.float32)
        dst = np.array([TABLE[k] for k in TABLE], dtype=np.float32)
        homography, _ = cv2.findHomography(src, dst)

    if homography is None:
        missing = [k for k in TABLE if k not in centers]
        status = f"waiting for corner markers, missing ID: {', '.join(map(str, missing))}"
    elif PROBE_ID not in centers:
        status = "probe marker (ID 4) not visible"
    else:
        pt = np.array([[centers[PROBE_ID]]], dtype=np.float32)
        x, y = cv2.perspectiveTransform(pt, homography)[0][0]
        inside = 0 <= x <= S and 0 <= y <= S
        status = f"probe:  x = {x:7.1f} mm   y = {y:7.1f} mm" + ("" if inside else "   (outside the square)")
        cv2.putText(frame, f"x={x:.0f} y={y:.0f} mm", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        if sock:
            sock.sendto(json.dumps({"x": float(x), "y": float(y), "t": time.time()}).encode(),
                        ("127.0.0.1", args.udp))

    now = time.time()
    if now - last_print > 0.2:
        last_print = now
        print(status, flush=True)


# --- test on a still image -------------------------------------------------
if args.image:
    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Could not read image: {args.image}")
    last_print = 0.0
    process(img)
    cv2.imwrite("annotated.jpg", img)
    print("Saved annotated.jpg")
    raise SystemExit

# --- live camera -----------------------------------------------------------
cap = cv2.VideoCapture(args.cam, cv2.CAP_V4L2)
if not cap.isOpened():
    raise SystemExit(f"Could not open camera {args.cam}. Check 'ls /dev/video*' and that the webcam is plugged in.")
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # lets USB webcams send HD at full speed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Running - press Ctrl+C to stop.")
last_save = 0.0
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            time.sleep(0.01)
            continue
        process(frame)
        if time.time() - last_save > 1.0:
            cv2.imwrite("snapshot.jpg", frame)
            last_save = time.time()
except KeyboardInterrupt:
    pass
finally:
    cap.release()
