from object_detection import HelmetDetection

import xml.etree.ElementTree as ET
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

import cv2
import box

def image_path(index):
    return f'data/helmet_detection/images/BikesHelmets{index}.png'

od = HelmetDetection('models/custom_yolov8m_helmet_person.pt')

def read_annotation(index):
    path = f'data/helmet_detection/annotations/BikesHelmets{index}.xml'

    root = ET.parse(path).getroot()

    gt_with_helmets = list()
    gt_without_helmets = list()

    with_or_withouts = [element.text for element in root.findall(".//name")]
    xmins = [int(element.text) for element in root.findall(".//xmin")]
    ymins = [int(element.text) for element in root.findall(".//ymin")]
    xmaxs = [int(element.text) for element in root.findall(".//xmax")]
    ymaxs = [int(element.text) for element in root.findall(".//ymax")]

    for i in range(len(xmaxs)):
        if with_or_withouts[i] == "With Helmet":
            gt_with_helmets.append((xmins[i], ymins[i], xmaxs[i], ymaxs[i]))
        else:
            gt_without_helmets.append((xmins[i], ymins[i], xmaxs[i], ymaxs[i]))

    return gt_with_helmets, gt_without_helmets

def benchmark(index, threshold_helmet, threshold_gt):
    TP, TN, FP, FN = (0, 0, 0, 0)

    image = cv2.imread(image_path(index))

    with_helmets, without_helmets = od.get_boxes(image, threshold_helmet)
    gt_with_helmets, gt_without_helmets = read_annotation(index)

    predicts = without_helmets + with_helmets
    predict_threshold = len(without_helmets)
    predict_cache = [False] * len(predicts)

    gts = gt_without_helmets + gt_with_helmets
    gt_threshold = len(gt_without_helmets)
    gt_cache = [False] * len(gts)

    for predict_idx, predict in enumerate(predicts):
        for gt_idx, gt in enumerate(gts):
            if box.intersection_area(gt, predict):
                if box.intersection_area(gt, predict) / box.box_area(gt) > threshold_gt:
                    predict_cache[predict_idx] = True
                    gt_cache[gt_idx] = True
                    if predict_idx < predict_threshold and gt_idx < gt_threshold:
                        TP += 1
                    elif predict_idx < predict_threshold and gt_idx > gt_threshold:
                        FP += 1
                    elif predict_idx > predict_threshold and gt_idx < gt_threshold:
                        FN += 1
                    else:
                        TN += 1

    # 수정된 코드
    for i in range(len(gt_cache)):
        if not gt_cache[i]: # gt가 어떤 predict와도 매칭되지 않았다면
            # 이 gt가 '헬멧 안 씀(Positive)' 클래스인지 확인
            if i < gt_threshold:
                FN += 1 # '헬멧 안 쓴' 사람을 놓쳤으므로 FN
    
    return TP, TN, FP, FN

def score(TP, TN, FP, FN):
    acc = (TP+TN) / (TP+TN+FP+FN)
    pre = (TP) / (TP+FP) if TP!=0 else 0
    rec = (TP) / (TP+FN) if TP!=0 else 0
    F1 = 2*pre*rec / (pre+rec) if not (pre==0 or rec==0) else 0

    return acc, pre, rec, F1



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

def total_benchmark(threshold_helmet, threshold_gt):
    total_TP, total_TN, total_FP, total_FN = (0, 0, 0, 0)

    for index in range(766):

        if index in [50, 54, 103, 140, 205, 279, 326, 343, 441, 444, 530, 75, 80, 764, 616, 671, 706]:
            continue
        TP, TN, FP, FN = benchmark(index, threshold_helmet, threshold_gt)

        total_TP += TP
        total_TN += TN
        total_FP += FP
        total_FN += FN

    return total_TP, total_TN, total_FP, total_FN

if __name__ == '__main__':
    acc_list = list()
    pre_list = list()
    rec_list = list()
    f1_list = list()

    for threshold in [0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]:
        total_TP, total_TN, total_FP, total_FN = total_benchmark(threshold, 0.5)

        acc, pre, rec, F1 = score(total_TP, total_TN, total_FP, total_FN)

        acc_list.append(acc)
        pre_list.append(pre)
        rec_list.append(rec)
        f1_list.append(F1)

    print(acc_list)
    print(pre_list)
    print(rec_list)
    print(f1_list)

    plt.plot([0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99], acc_list)
    plt.plot([0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99], pre_list)
    plt.plot([0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99], rec_list)
    plt.plot([0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99], f1_list)

    plt.ylim(0.8, 1.0)

    plt.legend(['accuracy', 'precision', 'recall', 'f1-score'])

    plt.show()