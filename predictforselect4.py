#Images with a confidence level higher than 0.7 are copied to the target folder
import os
import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import shutil
from ultralytics import YOLO

model = YOLO('best.pt')

def get_filelist(path):
    filelist = []
    for root, dirs, files in os.walk(path):
        for file_name in files:
            filelist.append(os.path.join(root, file_name))
    return filelist

def get_info(results):
    max_confidence = 0
    for result in results:
        conf_score = result['confidence']
        if conf_score > max_confidence:
            max_confidence = conf_score
    return max_confidence

if __name__ == "__main__":
    dir_path = './test_images'
    sa_path = './test_results_test_four'
    min_confidence = 0.7

    dirlist = os.listdir(dir_path)
    
    for dir in dirlist:
        filelist = get_filelist(os.path.join(dir_path, dir))
        file_result_dict = {}

        for file in tqdm(filelist):
            try:
                image = Image.open(file)
            except:
                print(f"Error opening image {file}. Skipping...")
                continue
            else:
                results = model(image, imgsz=512)
                if len(results[0].boxes) == 0:
                    pass
                else:
                    detection_results = []
                    for i in range(len(results[0].boxes)):
                        xyxy = results[0].boxes[i].xyxy.cpu().numpy()
                        if len(xyxy) == 1 and len(xyxy[0]) == 4:
                            x1, y1, x2, y2 = xyxy[0]
                        else:
                            print("Unexpected xyxy format:", xyxy)
                            continue
                        label = int(results[0].boxes[i].cls.item())
                        conf_score = results[0].boxes[i].conf.item()
                        detection_results.append({'class': label, 'confidence': conf_score})
                    result_info = get_info(detection_results)
                    file_result_dict[file] = result_info

        file_result_dict_items = sorted(file_result_dict.items(), key=lambda x: x[1], reverse=True)

        for file_result in file_result_dict_items:
            file_path = file_result[0]
            confidence = file_result[1]

            if confidence > min_confidence:
                id = os.path.split(os.path.split(file_path)[0])[1]
                path = os.path.join(sa_path, id)
                if not os.path.exists(path):
                    os.makedirs(path)
                shutil.copy(file_path, path + os.sep + os.path.split(file_path)[1])
                print(f"Copied: {file_path} with confidence {confidence}")
            else:
                print(f"Skipped: {file_path} with confidence {confidence}")

