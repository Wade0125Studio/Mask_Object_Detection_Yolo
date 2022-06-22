import os
import random
import xml.etree.ElementTree as ET

from utils.utils import get_classes

#--------------------------------------------------------------------------------------------------------------------------------#
#   annotation_mode用於指定該文件運行時計算的內容
#   annotation_mode為0代表整個標籤處理過程，包括獲得VOCdevkit/VOC2007/ImageSets裡面的txt以及訓練用的2007_train.txt、2007_val.txt
#   annotation_mode為1代表獲得VOCdevkit/VOC2007/ImageSets裡面的txt
#   annotation_mode為2代表獲得訓練用的2007_train.txt、2007_val.txt
#--------------------------------------------------------------------------------------------------------------------------------#
annotation_mode     = 0
#-------------------------------------------------------------------#
#   必須要修改，用於生成2007_train.txt、2007_val.txt的目標信息
#   與訓練和預測所用的classes_path一致即可
#   如果生成的2007_train.txt裡面沒有目標信息
#   那麼就是因為classes沒有設定正確
#   僅在annotation_mode為0和2的時候有效
#-------------------------------------------------------------------#
classes_path        = 'model_data/voc_classes.txt'
#--------------------------------------------------------------------------------------------------------------------------------#
#   trainval_percent用於指定(訓練集+驗證集)與測試集的比例，默認情況下 (訓練集+驗證集):測試集 = 9:1
#   train_percent用於指定(訓練集+驗證集)中訓練集與驗證集的比例，默認情況下 訓練集:驗證集 = 9:1
#   僅在annotation_mode為0和1的時候有效
#--------------------------------------------------------------------------------------------------------------------------------#
trainval_percent    = 0.9
train_percent       = 0.9
#-------------------------------------------------------#
#   指向VOC數據集所在的文件夾
#   默認指向根目錄下的VOC數據集
#-------------------------------------------------------#
VOCdevkit_path  = 'VOCdevkit'

VOCdevkit_sets  = [('2007', 'train'), ('2007', 'val')]
classes, _      = get_classes(classes_path)

def convert_annotation(year, image_id, list_file):
    in_file = open(os.path.join(VOCdevkit_path, 'VOC%s/Annotations/%s.xml'%(year, image_id)), encoding='utf-8')
    tree=ET.parse(in_file)
    root = tree.getroot()

    for obj in root.iter('object'):
        difficult = 0 
        if obj.find('difficult')!=None:
            difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(float(xmlbox.find('xmin').text)), int(float(xmlbox.find('ymin').text)), int(float(xmlbox.find('xmax').text)), int(float(xmlbox.find('ymax').text)))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        
if __name__ == "__main__":
    random.seed(0)
    if annotation_mode == 0 or annotation_mode == 1:
        print("Generate txt in ImageSets.")
        xmlfilepath     = os.path.join(VOCdevkit_path, 'VOC2007/Annotations')
        saveBasePath    = os.path.join(VOCdevkit_path, 'VOC2007/ImageSets/Main')
        temp_xml        = os.listdir(xmlfilepath)
        total_xml       = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        num     = len(total_xml)  
        list    = range(num)  
        tv      = int(num*trainval_percent)  
        tr      = int(tv*train_percent)  
        trainval= random.sample(list,tv)  
        train   = random.sample(trainval,tr)  
        
        print("train and val size",tv)
        print("train size",tr)
        ftrainval   = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
        ftest       = open(os.path.join(saveBasePath,'test.txt'), 'w')  
        ftrain      = open(os.path.join(saveBasePath,'train.txt'), 'w')  
        fval        = open(os.path.join(saveBasePath,'val.txt'), 'w')  
        
        for i in list:  
            name=total_xml[i][:-4]+'\n'  
            if i in trainval:  
                ftrainval.write(name)  
                if i in train:  
                    ftrain.write(name)  
                else:  
                    fval.write(name)  
            else:  
                ftest.write(name)  
        
        ftrainval.close()  
        ftrain.close()  
        fval.close()  
        ftest.close()
        print("Generate txt in ImageSets done.")

    if annotation_mode == 0 or annotation_mode == 2:
        print("Generate 2007_train.txt and 2007_val.txt for train.")
        for year, image_set in VOCdevkit_sets:
            image_ids = open(os.path.join(VOCdevkit_path, 'VOC%s/ImageSets/Main/%s.txt'%(year, image_set)), encoding='utf-8').read().strip().split()
            list_file = open('%s_%s.txt'%(year, image_set), 'w', encoding='utf-8')
            for image_id in image_ids:
                list_file.write('%s/VOC%s/JPEGImages/%s.png'%(os.path.abspath(VOCdevkit_path), year, image_id))

                convert_annotation(year, image_id, list_file)
                list_file.write('\n')
            list_file.close()
        print("Generate 2007_train.txt and 2007_val.txt for train done.")
