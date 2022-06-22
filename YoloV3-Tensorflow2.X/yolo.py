import colorsys
import os
import time

import numpy as np
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']="0"#"0,1,2,3,4,5,6,7"
import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

from PIL import ImageDraw, ImageFont
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model

from nets.yolo import yolo_body
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image)
from utils.utils_bbox import DecodeBox


class YOLO(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己訓練好的模型進行預測一定要修改model_path和classes_path！
        #   model_path指向logs文件夾下的權值文件，classes_path指向model_data下的txt
        #
        #   訓練好後logs文件夾下存在多個權值文件，選擇驗證集損失較低的即可。
        #   驗證集損失較低不代表mAP較高，僅代表該權值在驗證集上泛化性能較好。
        #   如果出現shape不匹配，同時要注意訓練時的model_path和classes_path參數的修改
        #--------------------------------------------------------------------------#
        "model_path"        : 'model_data/YoloV3_mask_objection_weights.h5',
        "classes_path"      : 'model_data/voc_classes.txt',
        #---------------------------------------------------------------------#
        #   anchors_path代表先驗框對應的txt文件，一般不修改。
        #   anchors_mask用於幫助代碼找到對應的先驗框，一般不修改。
        #---------------------------------------------------------------------#
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        #---------------------------------------------------------------------#
        #   輸入圖片的大小，必須為32的倍數。
        #---------------------------------------------------------------------#
        "input_shape"       : [416, 416],
        #---------------------------------------------------------------------#
        #   只有得分大於置信度的預測框會被保留下來
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   非極大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        "max_boxes"         : 100,
        #---------------------------------------------------------------------#
        #   該變量用於控制是否使用letterbox_image對輸入圖像進行不失真的resize，
        #   在多次測試後，發現關閉letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化yolo
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        #---------------------------------------------------#
        #   獲得種類和先驗框的數量
        #---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors     = get_anchors(self.anchors_path)

        #---------------------------------------------------#
        #   畫框設置不同的顏色
        #---------------------------------------------------#
        hsv_tuples  = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()

    #---------------------------------------------------#
    #   載入模型
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        self.yolo_model = yolo_body([None, None, 3], self.anchors_mask, self.num_classes)
        self.yolo_model.load_weights(self.model_path)

        print('{} model, anchors, and classes loaded.'.format(model_path))
        #---------------------------------------------------------#
        #   在DecodeBox函數中，我們會對預測結果進行後處理
        #   後處理的內容包括，解碼、非極大抑制、門限篩選等
        #---------------------------------------------------------#
        self.input_image_shape = Input([2,],batch_size=1)
        inputs  = [*self.yolo_model.output, self.input_image_shape]
        outputs = Lambda(
            DecodeBox, 
            output_shape = (1,), 
            name = 'yolo_eval',
            arguments = {
                'anchors'           : self.anchors, 
                'num_classes'       : self.num_classes, 
                'input_shape'       : self.input_shape, 
                'anchor_mask'       : self.anchors_mask,
                'confidence'        : self.confidence, 
                'nms_iou'           : self.nms_iou, 
                'max_boxes'         : self.max_boxes, 
                'letterbox_image'   : self.letterbox_image
             }
        )(inputs)
        self.yolo_model = Model([self.yolo_model.input, self.input_image_shape], outputs)

    @tf.function
    def get_pred(self, image_data, input_image_shape):
        out_boxes, out_scores, out_classes = self.yolo_model([image_data, input_image_shape], training=False)
        return out_boxes, out_scores, out_classes
    #---------------------------------------------------#
    #   檢測圖片
    #---------------------------------------------------#
    def detect_image(self, image, crop=False):
        #---------------------------------------------------------#
        #   在這裡將圖像轉換成RGB圖像，防止灰度圖在預測時報錯。
        #   代碼僅僅支持RGB圖像的預測，所有其它類型的圖像都會轉化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   給圖像增加灰條，實現不失真的resize
        #   也可以直接resize進行識別
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size維度，並進行歸一化
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        #---------------------------------------------------------#
        #   將圖像輸入網絡當中進行預測！
        #---------------------------------------------------------#
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        #---------------------------------------------------------#
        #   設置字體與邊框厚度
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        #---------------------------------------------------------#
        #   是否進行目標的裁剪
        #---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(out_boxes)):
                top, left, bottom, right = out_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        #---------------------------------------------------------#
        #   圖像繪製
        #---------------------------------------------------------#
        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[int(c)]
            box             = out_boxes[i]
            score           = out_scores[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        #---------------------------------------------------------#
        #   在這裡將圖像轉換成RGB圖像，防止灰度圖在預測時報錯。
        #   代碼僅僅支持RGB圖像的預測，所有其它類型的圖像都會轉化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   給圖像增加灰條，實現不失真的resize
        #   也可以直接resize進行識別
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size維度，並進行歸一化
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
        
        #---------------------------------------------------------#
        #   將圖像輸入網絡當中進行預測！
        #---------------------------------------------------------#
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 

        t1 = time.time()
        for _ in range(test_interval):
            out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    #---------------------------------------------------#
    #   檢測圖片
    #---------------------------------------------------#
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        #---------------------------------------------------------#
        #   在這裡將圖像轉換成RGB圖像，防止灰度圖在預測時報錯。
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   給圖像增加灰條，實現不失真的resize
        #   也可以直接resize進行識別
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size維度，並進行歸一化
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        #---------------------------------------------------------#
        #   將圖像輸入網絡當中進行預測！
        #---------------------------------------------------------#
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 

        for i, c in enumerate(out_classes):
            predicted_class             = self.class_names[int(c)]
            try:
                score                   = str(out_scores[i].numpy())
            except:
                score                   = str(out_scores[i])
            top, left, bottom, right    = out_boxes[i]
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
        
