import cv2
import numpy as np

def intersection_xyxy(boxA, boxB):
    xA = max(boxA[0], boxB[0]) 
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])  
    yB = min(boxA[3], boxB[3]) 

    if xB <= xA or yB <= yA:
        return None
    
    return [xA, yA, xB, yB] 

def intersection_area(boxA, boxB):
    inter = intersection_xyxy(boxA, boxB)
    if inter is None:
        return 0.0

    xA, yA, xB, yB = inter
    return (xB - xA) * (yB - yA)

def box_area(box):
    xA, yA, xB, yB = box

    return (xB - xA) * (yB - yA)

def box_height(box):
    return box[3] - box[1]

def subimage(frame, box):
    l, t, r, b = map(int, box)
    return frame[t:b, l:r]

def in_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    
    is_x_in = (x1 <= px <= x2)
    is_y_in = (y1 <= py <= y2)
    
    return is_x_in and is_y_in

def center(box):
    return ((box[0]+box[2])//2, (box[1]+box[3])//2)

def IoU(boxA, boxB):
    """
    두 개의 바운딩 박스(boxA, boxB)의 IoU를 계산합니다.

    Args:
        boxA (list or tuple): 첫 번째 박스의 좌표 [x1, y1, x2, y2].
        boxB (list or tuple): 두 번째 박스의 좌표 [x1, y1, x2, y2].

    Returns:
        float: 계산된 IoU 값 (0.0 ~ 1.0).
    """
    # 1. 교차 영역(Intersection)의 좌표 계산
    # 두 박스 중에서 더 큰 x1, y1과 더 작은 x2, y2를 찾아 교차 영역을 구합니다.
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # 2. 교차 영역의 넓이 계산
    # 너비와 높이는 음수가 될 수 없으므로, 0과 비교하여 큰 값을 사용합니다.
    # 박스가 겹치지 않으면 너비나 높이가 음수가 되어 넓이는 0이 됩니다.
    inter_width = max(0, xB - xA)
    inter_height = max(0, yB - yA)
    inter_area = inter_width * inter_height

    # 3. 각 박스의 넓이 계산
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # 4. 합집합(Union)의 넓이 계산
    # Union(A, B) = Area(A) + Area(B) - Intersection(A, B)
    union_area = float(boxA_area + boxB_area - inter_area)

    # 5. IoU 계산
    # 분모가 0이 되는 경우(박스 넓이가 0일 때 등)를 방지합니다.
    if union_area == 0:
        return 0.0
        
    iou = inter_area / union_area
    
    return iou

def draw_box(image: np.ndarray, box: tuple, caption: str = '', color: str = 'green', thickness: int = 2) -> np.ndarray:
    """
    NumPy 배열 형태의 이미지에 사각형을 그립니다.

    Args:
        image (np.ndarray): 그림을 그릴 대상 이미지 (NumPy 배열).
        box (tuple): 사각형의 좌표 (x1, y1, x2, y2).
        caption (str): box 위에 들어갈 텍스트
        color (str): red, r 또는 green, g 또는 blue, b.
        thickness (int): 선의 두께. -1을 주면 채워진 사각형을 그립니다. 기본값은 2.

    Returns:
        np.ndarray: 사각형이 그려진 이미지.
    """
    match color:
        case 'red' | 'r':
            color = (0, 0, 255)
        case 'green' | 'g':
            color = (0, 255, 0)
        case  'blue' | 'b':
            color = (255, 0, 0)

    # 입력된 이미지의 복사본을 만들어 원본이 변경되지 않도록 합니다.
    output_image = image.copy()
    
    # box 좌표를 (x1, y1)과 (x2, y2)로 분리합니다.
    x1, y1, x2, y2 = box
    
    # 정수형으로 변환해야 cv2.rectangle 함수에서 오류가 발생하지 않습니다.
    pt1 = (int(x1), int(y1))
    pt2 = (int(x2), int(y2))
    
    # cv2.rectangle 함수를 사용하여 사각형을 그립니다.
    cv2.rectangle(output_image, pt1, pt2, color, thickness)

    if caption:
        cv2.putText(
            image,
            caption,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
    
    return output_image