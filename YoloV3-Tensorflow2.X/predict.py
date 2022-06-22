#-----------------------------------------------------------------------#
#   predict.py將單張圖片預測、攝像頭檢測、FPS測試和目錄遍歷檢測等功能
#   整合到了一個py文件中，通過指定mode進行模式的修改。
#-----------------------------------------------------------------------#
import time

import cv2
import numpy as np
import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']="0"#"0,1,2,3,4,5,6,7"
import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

from PIL import Image

from yolo import YOLO


    
if __name__ == "__main__":
    yolo = YOLO()
    #----------------------------------------------------------------------------------------------------------#
    #   mode用於指定測試的模式：
    #   'predict'表示單張圖片預測，如果想對預測過程進行修改，如保存圖片，截取對像等，可以先看下方詳細的註釋
    #   'video'表示視頻檢測，可調用攝像頭或者視頻進行檢測，詳情查看下方註釋。
    #   'fps'表示測試fps，使用的圖片是img裡面的street.jpg，詳情查看下方註釋。
    #   'dir_predict'表示遍歷文件夾進行檢測並保存。默認遍歷img文件夾，保存img_out文件夾，詳情查看下方註釋。
    #----------------------------------------------------------------------------------------------------------#
    mode            = 'dir_predict'
    #-------------------------------------------------------------------------#
    #   crop指定了是否在單張圖片預測後對目標進行截取
    #   crop僅在mode='predict'時有效
    #-------------------------------------------------------------------------#
    crop            = False
    #----------------------------------------------------------------------------------------------------------#
    #   video_path用於指定視頻的路徑，當video_path=0時表示檢測攝像頭
    #   想要檢測視頻，則設置如video_path = "xxx.mp4"即可，代表讀取出根目錄下的xxx.mp4文件。
    #   video_save_path表示視頻保存的路徑，當video_save_path=""時表示不保存
    #   想要保存視頻，則設置如video_save_path = "yyy.mp4"即可，代表保存為根目錄下的yyy.mp4文件。
    #   video_fps用於保存的視頻的fps
    #   video_path、video_save_path和video_fps僅在mode='video'時有效
    #   保存視頻時需要ctrl+c退出或者運行到最後一幀才會完成完整的保存步驟。
    #----------------------------------------------------------------------------------------------------------#
    
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #-------------------------------------------------------------------------#
    #   test_interval用於指定測量fps的時候，圖片檢測的次數
    #   理論上test_interval越大，fps越準確。
    #-------------------------------------------------------------------------#
    test_interval   = 100
    #-------------------------------------------------------------------------#
    #   dir_origin_path指定了用於檢測的圖片的文件夾路徑
    #   dir_save_path指定了檢測完圖片的保存路徑
    #   dir_origin_path和dir_save_path僅在mode='dir_predict'時有效
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"

    if mode == "predict":
        '''
        1、如果想要進行檢測完的圖片的保存，利用r_image.save("img.jpg")即可保存，直接在predict.py裡進行修改即可。
        2、如果想要獲得預測框的坐標，可以進入yolo.detect_image函數，在繪圖部分讀取top，left，bottom，right這四個值。
        3、如果想要利用預測框截取下目標，可以進入yolo.detect_image函數，在繪圖部分利用獲取到的top，left，bottom，right這四個值
        在原圖上利用矩陣的方式進行截取。
        4、如果想要在預測圖上寫額外的字，比如檢測到的特定目標的數量，可以進入yolo.detect_image函數，在繪圖部分對predicted_class進行判斷，
        比如判斷if predicted_class == 'car': 即可判斷當前目標是否為車，然後記錄數量即可。利用draw.text即可寫字。
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop = crop)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("未能正確讀取攝像頭（視頻），請注意是否正確安裝攝像頭（是否正確填寫視頻路徑）。")

        fps = 0.0
        while(True):
            t1 = time.time()
            # 讀取某一幀
            ref, frame = capture.read()
            if not ref:
                break
            # 格式轉變，BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # 轉變成Image
            frame = Image.fromarray(np.uint8(frame))
            # 進行檢測
            frame = np.array(yolo.detect_image(frame))
            # RGBtoBGR滿足opencv顯示格式
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    elif mode == "fps":
        img = Image.open('img/street.jpg')
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os
        from tqdm import tqdm
        
        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)
                
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps' or 'dir_predict'.")
