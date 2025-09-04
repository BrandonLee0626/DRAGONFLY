import cv2
import math
from src import box
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from src.speed_estimation.camera_calibration.take_distance import take_distance_aruco

cache = dict()

def convert_for_tracker(bbs):
    detections = list()
    helmet_bbs = list()  

    for bb in bbs:
        x1, y1, x2, y2 = bb.xyxy[0].tolist()
        conf = bb.conf[0].item()
        x, y, w, h = x1, y1, x2 - x1, y2 - y1
        cls = int(bb.cls[0])
        if cls == 1:
            detections.append([[x, y, w, h], conf, cls])
        else:
            helmet_bbs.append(bb)
    
    return detections, helmet_bbs

def get_speed(person, camera_height):
    init_distance = person['init_distance']
    final_distance = person['final_distance']
    duration = person['duration']

    displacement = math.sqrt(final_distance**2-camera_height**2)-math.sqrt(init_distance**2-camera_height**2)

    return (displacement/10) / duration

model_path = 'models/yolov8m'
yolo = YOLO(model_path)

tracker_embedder = "mobilenet"
tracker = DeepSort(max_age=30, embedder=tracker_embedder, embedder_gpu=True)

video_path = 'data/video/demo12.mp4'
cap = cv2.VideoCapture(video_path)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

camera_height = 2400 # in mm

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    result = yolo(frame, verbose=False)[0]

    bbs = result.boxes

    detections, helmet_bbs = convert_for_tracker(bbs)

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r, b = map(int, track.to_ltrb())

        if track_id not in cache:
            init_distance = take_distance_aruco(frame, 183)
            if init_distance:
                print("========================")
                flag = True
                cache[track_id] = {
                    'image': box.subimage(frame, (l, t, r, b)),
                    'init_distance': init_distance,
                    'curr_height': box.box_height((l, t, r, b)),
                    'init_frame':frame_count,
                    'final_distance': None,
                    'duration': None,
                    'flag': False
                }
            else:
                continue

        curr_height = box.box_height((l, t, r, b))
        prev_height = cache[track_id]['curr_height']

        if prev_height - curr_height > 1 and not cache[track_id]['flag']:
            cache[track_id]['flag'] = True
            cache[track_id]['final_distance'] = take_distance_aruco(frame, 183)
            cache[track_id]['image'] = box.subimage(frame, (l ,t, r, b))
            cache[track_id]['duration'] = (frame_count-cache[track_id]['init_frame']) * (1/fps)
        
        cache[track_id]['curr_height'] = curr_height

    cv2.imshow('frame', frame)
    frame_count += 1

print(cache)

for id in cache:
    person = cache[id]

    print(get_speed(person, camera_height), 'm/s')