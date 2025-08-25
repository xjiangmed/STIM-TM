import numpy as np
import os
import cv2
from tqdm import tqdm

# ROOT_DIR = "/home/yangshu/Surgformer/data/AutoLaparo"
ROOT_DIR = "/home/xjiangbh/VideoTokenpruning_work/dataset/AutoLaparo_Task1/"
VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, "videos"))
VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if 'mp4' in x])

FRAME_NUMBERS = 0

##从视频数据集中按每秒抽取一帧图像，保存为图片文件，并统计总帧数
for video_name in VIDEO_NAMES:
    print(video_name)
    vidcap = cv2.VideoCapture(os.path.join(ROOT_DIR, "videos", video_name))
    fps = vidcap.get(cv2.CAP_PROP_FPS) #视频的 原始帧率
    print("fps", fps) #25.0
    success=True
    count=0
    save_dir = './frames/' + video_name.replace('.mp4', '') +'/'
    save_dir = os.path.join(ROOT_DIR, save_dir)
    os.makedirs(save_dir, exist_ok=True)
    while success is True:
        success,image = vidcap.read()
        if success:
            if count % fps == 0: #按时间维度降采样，每秒保留一帧，平衡数据量与信息完整性
                cv2.imwrite(save_dir + str(int(count//fps)).zfill(5) + '.png', image)
            count+=1
    vidcap.release()
    cv2.destroyAllWindows()
    print(count)
    FRAME_NUMBERS += count

print('Total Frams', FRAME_NUMBERS)