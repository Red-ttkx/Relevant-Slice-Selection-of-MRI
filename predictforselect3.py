#Select the top three images with the highest confidence level and save them to the specified result folder.
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
    sa_path = './test_results_test_three'

    dirlist = os.listdir(dir_path)
    
    for dir in dirlist:
        filelist = get_filelist(os.path.join(dir_path, dir))
        file_result_dict = {}

        for file in tqdm(filelist):
            try:
                image = Image.open(file)
            except:
                print('Open Error! Try again!')
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

        for file_result in file_result_dict_items[:3]:
            id = os.path.split(os.path.split(file_result[0])[0])[1]
            path = os.path.join(sa_path, id)
            if not os.path.exists(path):
                os.makedirs(path)
            shutil.copy(file_result[0], path + os.sep + os.path.split(file_result[0])[1])

