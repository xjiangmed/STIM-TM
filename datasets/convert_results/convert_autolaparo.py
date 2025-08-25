import numpy as np
import os
import glob

phases = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("文件夹已创建：", folder_path)
    else:
        print("文件夹已存在：", folder_path)



main_path = "/home/xjiangbh/VideoTokenpruning_work/STIM-TM/results/surgformer_HTA_AutoLaparo_0.0005_0.75_online_key_frame_frame16_Fixed_Stride_4_baseline/"
file_path_0 = os.path.join(main_path, "0.txt")
file_path_1 = os.path.join(main_path, "1.txt")
file_path_2 = os.path.join(main_path, "2.txt")
file_path_3 = os.path.join(main_path, "3.txt")
file_path_4 = os.path.join(main_path, "4.txt")
file_path_5 = os.path.join(main_path, "5.txt")

with open(file_path_0) as f:
    lines0 = f.readlines()
with open(file_path_1) as f:
    lines1 = f.readlines()
with open(file_path_2) as f:
    lines2 = f.readlines()
with open(file_path_3) as f:
    lines3 = f.readlines()
with open(file_path_4) as f:
    lines4 = f.readlines()
with open(file_path_5) as f:
    lines5 = f.readlines()
     
anns_path = main_path + "/phase_annotations"
pred_path = main_path + "/prediction"

create_folder_if_not_exists(anns_path)
create_folder_if_not_exists(pred_path)


# 生成真实注释文件
for i in range(15, 22):
    with open(
        anns_path + "/video-{}.txt".format(str(i)), "w"
    ) as f:
        f.write("Frame")
        f.write("\t")
        f.write("Phase")
        f.write("\n")
        assert len(lines0) == len(lines1)
        for j in range(1, len(lines0)):
            temp0 = lines0[j].split()
            temp1 = lines1[j].split()
            temp2 = lines2[j].split()
            temp3 = lines3[j].split()
            temp4 = lines4[j].split()
            temp5 = lines5[j].split()
            
            if temp0[1] == "{}".format(str(i)):
                f.write(str(temp0[2]))  # phase_annotations
                f.write("\t")  # phase_annotations
                f.write(str(temp0[-1]))  # phase_annotations
                f.write("\n")  # phase_annotations
            if temp1[1] == "{}".format(str(i)):
                f.write(str(temp1[2]))  # phase_annotations
                f.write("\t")  # phase_annotations
                f.write(str(temp1[-1]))  # phase_annotations
                f.write("\n")  # phase_annotations
            if temp2[1] == "{}".format(str(i)):
                f.write(str(temp2[2]))  # phase_annotations
                f.write("\t")  # phase_annotations
                f.write(str(temp2[-1]))  # phase_annotations
                f.write("\n")  # phase_annotations
            if temp3[1] == "{}".format(str(i)):
                f.write(str(temp3[2]))  # phase_annotations
                f.write("\t")  # phase_annotations
                f.write(str(temp3[-1]))  # phase_annotations
                f.write("\n")  # phase_annotations
            if temp4[1] == "{}".format(str(i)):
                f.write(str(temp4[2]))  # phase_annotations
                f.write("\t")  # phase_annotations
                f.write(str(temp4[-1]))  # phase_annotations
                f.write("\n")  # phase_annotations
            if temp5[1] == "{}".format(str(i)):
                f.write(str(temp5[2]))  # phase_annotations
                f.write("\t")  # phase_annotations
                f.write(str(temp5[-1]))  # phase_annotations
                f.write("\n")  # phase_annotations
            
            
with open(file_path_0) as f:
    lines0 = f.readlines()
with open(file_path_1) as f:
    lines1 = f.readlines()
with open(file_path_2) as f:
    lines2 = f.readlines()
with open(file_path_3) as f:
    lines3 = f.readlines()
with open(file_path_4) as f:
    lines4 = f.readlines()
with open(file_path_5) as f:
    lines5 = f.readlines()

# 生成预测结果文件
for i in range(15, 22):
    print(i)
    with open(
        pred_path + "/video-{}.txt".format(str(i)), "w"
    ) as f:  # phase_annotations
        f.write("Frame")
        f.write("\t")
        f.write("Phase")
        f.write("\n")
        assert len(lines0) == len(lines1)
        for j in range(1, len(lines0)):
            temp0 = lines0[j].strip() # prediction
            temp1 = lines1[j].strip() # prediction
            temp2 = lines2[j].strip()
            temp3 = lines3[j].strip()
            temp4 = lines4[j].strip()
            temp5 = lines5[j].strip()
            
            data0 = np.fromstring(
                temp0.split("[")[1].split("]")[0], dtype=np.float32, sep=","
            ) # prediction
            data1 = np.fromstring(
                temp1.split("[")[1].split("]")[0], dtype=np.float32, sep=","
            ) # prediction
            data2 = np.fromstring(
                temp2.split("[")[1].split("]")[0], dtype=np.float32, sep=","
            ) # prediction
            data3 = np.fromstring(
                temp3.split("[")[1].split("]")[0], dtype=np.float32, sep=","
            ) # prediction
            data4 = np.fromstring(
                temp4.split("[")[1].split("]")[0], dtype=np.float32, sep=","
            ) # prediction
            data5 = np.fromstring(
                temp5.split("[")[1].split("]")[0], dtype=np.float32, sep=","
            ) # prediction
            
            data0 = data0.argmax() # prediction
            data1 = data1.argmax() # prediction
            data2 = data2.argmax() # prediction
            data3 = data3.argmax() # prediction
            data4 = data4.argmax() # prediction
            data5 = data5.argmax() # prediction
            
            temp0 = lines0[j].split()
            temp1 = lines1[j].split()
            temp2 = lines2[j].split()
            temp3 = lines3[j].split()
            temp4 = lines4[j].split()
            temp5 = lines5[j].split()
            
            if temp0[1] == "{}".format(str(i)):
                f.write(str(temp0[2])) # prediction
                f.write('\t') # prediction
                f.write(str(data0)) # prediction
                f.write('\n') # prediction
            if temp1[1] == "{}".format(str(i)):
                f.write(str(temp1[2])) # prediction
                f.write('\t') # prediction
                f.write(str(data1)) # prediction
                f.write('\n') # prediction
            if temp2[1] == "{}".format(str(i)):
                f.write(str(temp2[2])) # prediction
                f.write('\t') # prediction
                f.write(str(data2)) # prediction
                f.write('\n') # prediction
            if temp3[1] == "{}".format(str(i)):
                f.write(str(temp3[2])) # prediction
                f.write('\t') # prediction
                f.write(str(data3)) # prediction
                f.write('\n') # prediction
            if temp4[1] == "{}".format(str(i)):
                f.write(str(temp4[2])) # prediction
                f.write('\t') # prediction
                f.write(str(data4)) # prediction
                f.write('\n') # prediction
            if temp5[1] == "{}".format(str(i)):
                f.write(str(temp5[2])) # prediction
                f.write('\t') # prediction
                f.write(str(data5)) # prediction
                f.write('\n') # prediction
            