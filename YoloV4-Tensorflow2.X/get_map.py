import os
import xml.etree.ElementTree as ET

import os
os.environ['CUDA_VISIBLE_DEVICES']="0"
import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
# 設定 Keras 使用的 Session
tf.compat.v1.keras.backend.set_session(sess)
from tensorflow.keras import backend as K
K.clear_session()
import tensorflow.keras.backend as K
from PIL import Image
from tqdm import tqdm

from utils.utils import get_classes
from utils.utils_map import get_coco_map, get_map
from yolo import YOLO

    
if __name__ == "__main__":
    '''
    Recall和Precision不像AP是一個面積的概念，在門限值不同時，網絡的Recall和Precision值是不同的。
    map計算結果中的Recall和Precision代表的是當預測時，門限置信度為0.5時，所對應的Recall和Precision值。

    此處獲得的./map_out/detection-results/裡面的txt的框的數量會比直接predict多一些，這是因為這裡的門限低，
    目的是為了計算不同門限條件下的Recall和Precision值，從而實現map的計算。
    '''
    #------------------------------------------------------------------------------------------------------------------#
    #   map_mode用於指定該文件運行時計算的內容
    #   map_mode為0代表整個map計算流程，包括獲得預測結果、獲得真實框、計算VOC_map。
    #   map_mode為1代表僅僅獲得預測結果。
    #   map_mode為2代表僅僅獲得真實框。
    #   map_mode為3代表僅僅計算VOC_map。
    #   map_mode為4代表利用COCO工具箱計算當前數據集的0.50:0.95map。需要獲得預測結果、獲得真實框後並安裝pycocotools才行
    #-------------------------------------------------------------------------------------------------------------------#
    map_mode        = 0
    #-------------------------------------------------------#
    #   此處的classes_path用於指定需要測量VOC_map的類別
    #   一般情況下與訓練和預測所用的classes_path一致即可
    #-------------------------------------------------------#
    classes_path    = 'model_data/voc_classes.txt'
    #-------------------------------------------------------#
    #   MINOVERLAP用於指定想要獲得的mAP0.x
    #   比如計算mAP0.75，可以設定MINOVERLAP = 0.75。
    #-------------------------------------------------------#
    MINOVERLAP      = 0.5
    #-------------------------------------------------------#
    #   map_vis用於指定是否開啟VOC_map計算的可視化
    #-------------------------------------------------------#
    map_vis         = False
    #-------------------------------------------------------#
    #   指向VOC數據集所在的文件夾
    #   默認指向根目錄下的VOC數據集
    #-------------------------------------------------------#
    VOCdevkit_path  = 'VOCdevkit'
    #-------------------------------------------------------#
    #   結果輸出的文件夾，默認為map_out
    #-------------------------------------------------------#
    map_out_path    = 'map_out'

    image_ids = open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Main/test.txt")).read().strip().split()

    if not os.path.exists(map_out_path):
        os.makedirs(map_out_path)
    if not os.path.exists(os.path.join(map_out_path, 'ground-truth')):
        os.makedirs(os.path.join(map_out_path, 'ground-truth'))
    if not os.path.exists(os.path.join(map_out_path, 'detection-results')):
        os.makedirs(os.path.join(map_out_path, 'detection-results'))
    if not os.path.exists(os.path.join(map_out_path, 'images-optional')):
        os.makedirs(os.path.join(map_out_path, 'images-optional'))

    class_names, _ = get_classes(classes_path)

    if map_mode == 0 or map_mode == 1:
        print("Load model.")
        yolo = YOLO(confidence = 0.001, nms_iou = 0.5)
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path  = os.path.join(VOCdevkit_path, "VOC2007/JPEGImages/"+image_id+".png")
            image       = Image.open(image_path)
            if map_vis:
                image.save(os.path.join(map_out_path, "images-optional/" + image_id + ".png"))
            yolo.get_map_txt(image_id, image, class_names, map_out_path)
        print("Get predict result done.")
        
    if map_mode == 0 or map_mode == 2:
        print("Get ground truth result.")
        for image_id in tqdm(image_ids):
            with open(os.path.join(map_out_path, "ground-truth/"+image_id+".txt"), "w") as new_f:
                root = ET.parse(os.path.join(VOCdevkit_path, "VOC2007/Annotations/"+image_id+".xml")).getroot()
                for obj in root.findall('object'):
                    difficult_flag = False
                    if obj.find('difficult')!=None:
                        difficult = obj.find('difficult').text
                        if int(difficult)==1:
                            difficult_flag = True
                    obj_name = obj.find('name').text
                    if obj_name not in class_names:
                        continue
                    bndbox  = obj.find('bndbox')
                    left    = bndbox.find('xmin').text
                    top     = bndbox.find('ymin').text
                    right   = bndbox.find('xmax').text
                    bottom  = bndbox.find('ymax').text

                    if difficult_flag:
                        new_f.write("%s %s %s %s %s difficult\n" % (obj_name, left, top, right, bottom))
                    else:
                        new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))
        print("Get ground truth result done.")

    if map_mode == 0 or map_mode == 3:
        print("Get map.")
        get_map(MINOVERLAP, True, path = map_out_path)
        print("Get map done.")

    if map_mode == 4:
        print("Get map.")
        get_coco_map(class_names = class_names, path = map_out_path)
        print("Get map done.")
