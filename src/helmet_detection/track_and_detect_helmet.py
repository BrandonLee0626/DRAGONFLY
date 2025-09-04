from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
import cv2
import numpy as np
import sys
import math
import src.box as box

class DetectedPerson:
    def __init__(self, ratio, time):
        self.photo = None
        self.with_helmet = False
        self.box = (0, 0, 0, 0)
        self.photo_fix = False
        self.first_detected_time = time
        self.threshold_time = None
        self.first_detected_ratio = ratio
        self.threshold_ratio = None
        self.average_speed = None

    def show(self, frame):
        if self.with_helmet:
            cv2.rectangle(frame, self.box[:2], self.box[2:], (0, 255, 0), 2)

            cv2.putText(
                frame,
                f"ID {track_id} (with helmet)",
                (self.box[0], self.box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        else:
            cv2.rectangle(frame, self.box[:2], self.box[2:], (255, 0, 0), 2)

            cv2.putText(
                frame,
                f"ID {track_id} (without helmet)",
                (self.box[0], self.box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return frame
    
    def get_displacement(self, camera_height):
        threshold_distance = camera_height * math.tan(math.radians(43.65))
        print("thr dis", threshold_distance)
        threshold_depth = camera_height / math.cos(math.radians(43.65))
        print("thr dep", threshold_depth)

        first_detected_depth = threshold_depth * self.threshold_ratio / self.first_detected_ratio
        print("fir dep", first_detected_depth)
        first_detected_distance = first_detected_depth * math.sin(math.radians(43.65))
        print("fir dis", first_detected_distance)

        return first_detected_distance - threshold_distance
    
    def get_average_speed(self, camera_height):
        displacement = self.get_displacement(camera_height)

        self.average_speed = displacement / (self.threshold_time - self.first_detected_time)


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

def resize_to_height(img, target_height):
    h, w = img.shape[:2]
    ratio = target_height / h
    new_w = int(w * ratio)
    return cv2.resize(img, (new_w, target_height))

def concat_images_horizontally(images):
    min_height = min(img.shape[0] for img in images)
    valid_images = [img for img in images if img is not None and img.size > 0]
    resized_images = [resize_to_height(img, min_height) for img in valid_images]
    return np.hstack(resized_images)

def pad_to_same_width(img1, img2):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    target_width = max(w1, w2)

    def pad_right(img, target_w):
        h, w = img.shape[:2]
        pad_w = target_w - w
        if pad_w > 0:
            return cv2.copyMakeBorder(img, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        else:
            return img

    return pad_right(img1, target_width), pad_right(img2, target_width)

def put_label(img, text, font_scale=0.6, thickness=2):
    labeled_img = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = 10
    text_y = 30

    cv2.putText(labeled_img, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness + 1, lineType=cv2.LINE_AA)
    cv2.putText(labeled_img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, lineType=cv2.LINE_AA)
    return labeled_img

def send_result(cache):
    with_helmet = list()
    without_helmet = list()

    with_helmet_img = None
    without_helmet_img = None

    for id in cache:
        person = cache[id]

        if person.photo_fix:
            photo_with_text = person.photo.copy()

            speed_text = f"{person.average_speed:.2f} cm/s"

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_color = (0, 0, 0)
            thickness = 2
            
            text_size, _ = cv2.getTextSize(speed_text, font, font_scale, thickness)
            text_x = (photo_with_text.shape[1] - text_size[0]) // 2
            text_y = photo_with_text.shape[0] - 15

            cv2.putText(photo_with_text, speed_text, (text_x, text_y), font, font_scale, font_color, thickness)
            
            cv2.putText(photo_with_text, speed_text, (text_x, text_y), font, font_scale, (255, 255, 255), 1)

            if person.with_helmet:
                with_helmet.append(photo_with_text)
            
            else:
                without_helmet.append(photo_with_text)
    if with_helmet:
        with_helmet_img = concat_images_horizontally(with_helmet)
    if without_helmet:
        without_helmet_img = concat_images_horizontally(without_helmet)

    if with_helmet_img is None and without_helmet_img is None:
        print('No Detection!')
        sys.exit(1)
        
    elif without_helmet_img is None:
        without_helmet_img = np.full(with_helmet_img.shape, 255, dtype=np.uint8)
    
    elif with_helmet_img is None:
        with_helmet_img = np.full(without_helmet_img.shape, 255, dtype=np.uint8)

    padded_with_helmet, padded_without_helmet = pad_to_same_width(with_helmet_img, without_helmet_img)

    labeled_padded_with_helmet = put_label(padded_with_helmet, 'with helmet')
    labeled_padded_without_helmet = put_label(padded_without_helmet, 'without helmet')

    result = np.vstack([labeled_padded_with_helmet, labeled_padded_without_helmet])

    cv2.imwrite('result/result.png', result)


model_name = 'custom_yolov8m_helmet_person'
yolo = YOLO(f'models/{model_name}.pt')

tracker_embedder = "mobilenet"
tracker = DeepSort(max_age=30, embedder=tracker_embedder, embedder_gpu=True)

video_name = 'demo11'
cap = cv2.VideoCapture(f'video/{video_name}.mp4')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

output_file = 'result/'+model_name+'_'+tracker_embedder+'_'+video_name+'.mp4'
out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

prev_frame = None

camera_height = 308.5

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
            cache[track_id] = DetectedPerson(box.box_height((l, t, r, b))/frame.shape[0], frame_count / fps)

        curr_height = box.box_height((l, t, r, b))
        prev_height = box.box_height(cache[track_id].box)

        if prev_height - curr_height > 5 and not cache[track_id].photo_fix:
            cache[track_id].photo_fix = True
            cache[track_id].photo = box.subimage(prev_frame, cache[track_id].box)
            cache[track_id].threshold_time = frame_count / fps
            cache[track_id].threshold_ratio = prev_height / frame.shape[0]
            cache[track_id].get_average_speed(camera_height)

        cache[track_id].box = (l, t, r, b)

        for helmet_bb in helmet_bbs:
            person_box = (l, t, r, b)
            helmet_box = helmet_bb.xyxy[0].tolist()
            if box.intersection_area(person_box, helmet_box) / box.box_area(helmet_box) > 0.5:
                cache[track_id].with_helmet = True
                break

        prev_frame = frame
        frame = cache[track_id].show(frame)
    
    cv2.imshow("Tracking", frame)
    out.write(frame)

    frame_count += 1

for id in cache:
    if cache[id].with_helmet:
        cv2.imwrite(f'result/with_helmet_{id}.png', cache[id].photo)
    else:
        cv2.imwrite(f'result/without_helmet_{id}.png', cache[id].photo)

send_result(cache)