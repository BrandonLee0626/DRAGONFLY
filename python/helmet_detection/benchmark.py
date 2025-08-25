from ultralytics import YOLO
import xml.etree.ElementTree as ET

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import python.helmet_detection.box as box

model_name = 'custom_yolov8m_helmet_person'
yolo = YOLO(f'models/{model_name}.pt')

class DetectedPerson:
    def __init__(self, bb_, with_helmet_):
        self.bb = bb_
        self.with_helmet = with_helmet_
        self.checked = False
    
    def check(self):
        self.checked = True

class TruthHead:
    def __init__(self, center_, bottom_, with_helmet_):
        self.center = center_
        self.bottom = bottom_
        self.with_helmet = with_helmet_
        self.checked = False

    def check(self):
        self.checked = True

def read_annotation(index):
    path = f'data/helmet_detection/annotations/BikesHelmets{index}.xml' 

    root = ET.parse(path).getroot()

    with_or_withouts = [element.text for element in root.findall(".//name")]
    xmins = [int(element.text) for element in root.findall(".//xmin")]
    ymins = [int(element.text) for element in root.findall(".//ymin")]
    xmaxs = [int(element.text) for element in root.findall(".//xmax")]
    ymaxs = [int(element.text) for element in root.findall(".//ymax")]

    return [
        {
        'with_or_without': with_or_withouts[i],
        'xmin': xmins[i],
        'ymin': ymins[i],
        'xmax': xmaxs[i],
        'ymax': ymaxs[i]
        }
        for i in range(len(xmaxs))]

def get_parsons(index):
    path = f'data/helmet_detection/images/BikesHelmets{index}.png'

    persons = list()

    person_bbs = list()
    helmet_bbs = list()

    bbs = yolo(path, verbose=False)[0].boxes

    for bb in bbs:
        if int(bb.cls[0]) == 1: # person
            person_bbs.append(tuple(bb.xyxy[0].tolist()))
        else:
            helmet_bbs.append(tuple(bb.xyxy[0].tolist()))
    
    for person_bb in person_bbs:
        with_helmet = False
        for helmet_bb in helmet_bbs:
            if box.intersection_area(person_bb, helmet_bb) >= box.box_area(helmet_bb) * 0.5:
                with_helmet = True
                break
        if with_helmet:
            persons.append(DetectedPerson(person_bb, with_helmet))

    return persons

def get_heads(index):
    truths = read_annotation(index)

    heads = list()

    for head in truths:
        with_or_without = head['with_or_without']
        l = head['xmin']
        t = head['ymin']
        r = head['xmax']
        b = head['ymax']

        heads.append(TruthHead(box.center((l, t, r, b)), ((l + r) // 2, b), 1 if with_or_without == "With Helmet" else 0))
    
    return heads

def benchmark(index):
    TP, TN, FP, FN = (0, 0, 0, 0)

    persons = get_parsons(index)
    heads = get_heads(index)

    for person in persons:
        for head in heads:
            if box.in_box(head.center, person.bb):
                head.check()
                person.check()

                if head.with_helmet:
                    if person.with_helmet:
                        TP += 1
                    else:
                        FN += 1
                
                else:
                    if person.with_helmet:
                        FP += 1
                    else:
                        TN += 1

                break
    
    for person in persons:
        if not person.checked and person.with_helmet:
            FP += 1
    
    for head in heads:
        if not head.checked and head.with_helmet:
            FN += 1

    return TP, TN, FP, FN    

def plot_confusion_matrix(tp, tn, fp, fn, title='Confusion Matrix', filename='benchmark.png'):
    """
    TP, TN, FP, FN 값을 사용하여 Confusion Matrix를 시각화하고 파일로 저장합니다.

    Args:
        tp (int): True Positive (참 양성)
        tn (int): True Negative (참 음성)
        fp (int): False Positive (거짓 양성) - Type I Error
        fn (int): False Negative (거짓 음성) - Type II Error
        title (str): 그래프의 제목
        filename (str): 저장할 파일 경로. None으로 설정 시 저장하지 않고 화면에 표시합니다.
    """
    # 혼동 행렬 데이터 구성
    # [[TP, FN],
    #  [FP, TN]]
    confusion_matrix = np.array([[tp, fn], [fp, tn]])

    # 각 셀에 표시될 텍스트 레이블 구성
    group_names = ['True Positive (TP)', 'False Negative (FN)', 'False Positive (FP)', 'True Negative (TN)']
    group_counts = [f"{value}" for value in confusion_matrix.flatten()]
    group_percentages = [f"{value:.2%}" for value in confusion_matrix.flatten() / np.sum(confusion_matrix)]
    
    # 최종 레이블 생성 (예: "True Positive\n50\n30.30%")
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    # Seaborn을 사용하여 히트맵 생성
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=labels, fmt='', cmap='Blues', cbar=False,
                xticklabels=['Positive', 'Negative'], yticklabels=['Positive', 'Negative'])

    # 축 레이블 및 제목 설정
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('Actual Label', fontsize=14)
    plt.title(title, fontsize=16)
    
    # 그래프를 파일로 저장하거나 화면에 출력
    if filename:
        # 해상도(dpi)와 여백(bbox_inches)을 조절하여 깔끔하게 저장
        plt.savefig(f'result/{filename}', dpi=300, bbox_inches='tight')
        print(f"Confusion matrix가 '{filename}' 파일로 저장되었습니다.")
    else:
        plt.show()
    
    # 메모리 누수를 방지하기 위해 figure를 닫아줍니다.
    plt.close()

if __name__ == '__main__':
    total_TP, total_TN, total_FP, total_FN = (0, 0, 0, 0)

    for index in range(766):

        if index in [50, 54, 103, 140, 205, 279, 326, 343, 441, 444, 530, 75, 80, 764, 616, 671, 706]:
            continue
        TP, TN, FP, FN = benchmark(index)

        total_TP += TP
        total_TN += TN
        total_FP += FP
        total_FN += FN

    plot_confusion_matrix(total_TP, total_TN, total_FP, total_FN)