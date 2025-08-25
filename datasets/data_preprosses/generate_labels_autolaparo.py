import numpy as np
import os
import cv2
import pickle
from tqdm import tqdm

#提供结构化输入
def main():
    # ROOT_DIR = "/home/yangshu/Surgformer/data/AutoLaparo"
    ROOT_DIR = "/home/xjiangbh/VideoTokenpruning_work/dataset/AutoLaparo_Task1/"
    VIDEO_NAMES = os.listdir(os.path.join(ROOT_DIR, 'frames'))
    VIDEO_NAMES = sorted([x for x in VIDEO_NAMES if "DS" not in x]) #过滤掉包含"DS"的文件

    TRAIN_NUMBERS = np.arange(1,11).tolist()
    VAL_NUMBERS = np.arange(11,15).tolist()
    TEST_NUMBERS = np.arange(15,22).tolist()

    TRAIN_FRAME_NUMBERS = 0
    VAL_FRAME_NUMBERS = 0
    TEST_FRAME_NUMBERS = 0
    
    #将三个字典train_pkl、val_pkl、test_pkl用来存储每个视频的帧信息
    train_pkl = dict() 
    val_pkl = dict()
    test_pkl = dict()

    unique_id = 0
    unique_id_train = 0
    unique_id_val = 0
    unique_id_test = 0

    id2phase = {0: "Preparation", 1: "Dividing Ligament and Peritoneum", 2: "Dividing Uterine Vessels and Ligament", 
                3: "Transecting the Vagina", 4: "Specimen Removal", 5: "Suturing", 6: "Washing"}

    for video_id in VIDEO_NAMES:
        vid_id = int(video_id)
        if vid_id in TRAIN_NUMBERS:
            unique_id = unique_id_train
        elif vid_id in VAL_NUMBERS:
            unique_id = unique_id_val
        elif vid_id in TEST_NUMBERS:
            unique_id = unique_id_test

        # 总帧数(frames)
        video_path = os.path.join(ROOT_DIR, "frames", video_id)
        frames_list = os.listdir(video_path)

        # 打开Label文件
        phase_path = os.path.join(ROOT_DIR, 'labels', "label_" + video_id + '.txt')
        phase_file = open(phase_path, 'r')
        phase_results = phase_file.readlines()[1:]

        frame_infos = list()
        frame_id_ = 0
        for frame_id in tqdm(range(0, len(frames_list))):
            info = dict()
            info['unique_id'] = unique_id #全局唯一ID,跨视频的帧唯一标识
            info['frame_id'] = frame_id_  #视频内帧序号, 当前视频内的相对序号（从0开始）
            info['original_frame_id'] = frame_id #原始序号, 标签文件中的原始帧号（从1开始）
            info['video_id'] = video_id
            info['tool_gt'] = None
            info['frames'] = len(frames_list)
            phase = phase_results[frame_id].strip().split()
            print('phase:', phase)
            assert int(phase[0]) == frame_id + 1
            phase_id = int(phase[1])
            # info['phase_gt'] = phase_id
            # info['phase_name'] = id2phase[int(phase[1])]
            info['phase_gt'] = phase_id-1
            info['phase_name'] = id2phase[int(phase[1])-1]
            info['fps'] = 1 # 采样率（每秒1帧）
            frame_infos.append(info)
            unique_id += 1
            frame_id_ += 1
        
        if vid_id in TRAIN_NUMBERS:
            train_pkl[video_id] = frame_infos
            TRAIN_FRAME_NUMBERS += len(frames_list)
            unique_id_train = unique_id
        elif vid_id in VAL_NUMBERS:
            val_pkl[video_id] = frame_infos
            VAL_FRAME_NUMBERS += len(frames_list)
            unique_id_val = unique_id
        elif vid_id in TEST_NUMBERS:
            test_pkl[video_id] = frame_infos
            TEST_FRAME_NUMBERS += len(frames_list)
            unique_id_test = unique_id
    
    train_save_dir = os.path.join(ROOT_DIR, 'labels_pkl', 'train')
    os.makedirs(train_save_dir, exist_ok=True)
    with open(os.path.join(train_save_dir, '1fpstrain.pickle'), 'wb') as file:
        pickle.dump(train_pkl, file)

    val_save_dir = os.path.join(ROOT_DIR, 'labels_pkl', 'val')
    os.makedirs(val_save_dir, exist_ok=True)
    with open(os.path.join(val_save_dir, '1fpsval.pickle'), 'wb') as file:
        pickle.dump(val_pkl, file)

    test_save_dir = os.path.join(ROOT_DIR, 'labels_pkl', 'test')
    os.makedirs(test_save_dir, exist_ok=True)
    with open(os.path.join(test_save_dir, '1fpstest.pickle'), 'wb') as file:
        pickle.dump(test_pkl, file)


    print('TRAIN Frams', TRAIN_FRAME_NUMBERS, unique_id_train)
    print('VAL Frams', VAL_FRAME_NUMBERS, unique_id_val)
    print('TEST Frams', TEST_FRAME_NUMBERS, unique_id_test) 

if __name__ == '__main__':
    main()