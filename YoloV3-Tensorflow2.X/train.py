import datetime
import os
from functools import partial

import tensorflow as tf
import os
os.environ['CUDA_VISIBLE_DEVICES']="4"#"0"
import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import (EarlyStopping, LearningRateScheduler,
                                        TensorBoard)
from tensorflow.keras.optimizers import SGD, Adam

from nets.yolo import get_train_model, yolo_body
from nets.yolo_training import get_lr_scheduler
from utils.callbacks import LossHistory, ModelCheckpoint
from utils.dataloader import YoloDatasets
from utils.utils import get_anchors, get_classes
from utils.utils_fit import fit_one_epoch

    
'''
訓練自己的目標檢測模型一定需要注意以下幾點：
1、訓練前仔細檢查自己的格式是否滿足要求，該庫要求數據集格式為VOC格式，需要準備好的內容有輸入圖片和標籤
   輸入圖片為.jpg圖片，無需固定大小，傳入訓練前會自動進行resize。
   灰度圖會自動轉成RGB圖片進行訓練，無需自己修改。
   輸入圖片如果後綴非jpg，需要自己批量轉成jpg後再開始訓練。

   標籤為.xml格式，文件中會有需要檢測的目標信息，標籤文件和輸入圖片文件相對應。

2、訓練好的權值文件保存在logs文件夾中，每個epoch都會保存一次，如果只是訓練了幾個step是不會保存的，epoch和step的概念要捋清楚一下。
   在訓練過程中，該代碼並沒有設定只保存最低損失的，因此按默認參數訓練完會有100個權值，如果空間不夠可以自行刪除。
   這個並不是保存越少越好也不是保存越多越好，有人想要都保存、有人想只保存一點，為了滿足大多數的需求，還是都保存可選擇性高。

3、損失值的大小用於判斷是否收斂，比較重要的是有收斂的趨勢，即驗證集損失不斷下降，如果驗證集損失基本上不改變的話，模型基本上就收斂了。
   損失值的具體大小並沒有什麼意義，大和小只在於損失的計算方式，並不是接近於0才好。如果想要讓損失好看點，可以直接到對應的損失函數里面除上10000。
   訓練過程中的損失值會保存在logs文件夾下的loss_%Y_%m_%d_%H_%M_%S文件夾中

4、調參是一門蠻重要的學問，沒有什麼參數是一定好的，現有的參數是我測試過可以正常訓練的參數，因此我會建議用現有的參數。
   但是參數本身並不是絕對的，比如隨著batch的增大學習率也可以增大，效果也會好一些；過深的網絡不要用太大的學習率等等。
   這些都是經驗上，只能靠各位同學多查詢資料和自己試試了。
''' 
if __name__ == "__main__":
    #----------------------------------------------------#
    #   是否使用eager模式訓練
    #----------------------------------------------------#
    eager           = False
    #---------------------------------------------------------------------#
    #   classes_path    指向model_data下的txt，與自己訓練的數據集相關 
    #                   訓練前一定要修改classes_path，使其對應自己的數據集
    #---------------------------------------------------------------------#
    classes_path    = 'model_data/voc_classes.txt'
    #---------------------------------------------------------------------#
    #   anchors_path    代表先驗框對應的txt文件，一般不修改。
    #   anchors_mask    用於幫助代碼找到對應的先驗框，一般不修改。
    #---------------------------------------------------------------------#
    anchors_path    = 'model_data/yolo_anchors.txt'
    anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    #----------------------------------------------------------------------------------------------------------------------------#
    #   權值文件的下載請看README，可以通過網盤下載。模型的 預訓練權重 對不同數據集是通用的，因為特徵是通用的。
    #   模型的 預訓練權重 比較重要的部分是 主幹特徵提取網絡的權值部分，用於進行特徵提取。
    #   預訓練權重對於99%的情況都必須要用，不用的話主幹部分的權值太過隨機，特徵提取效果不明顯，網絡訓練的結果也不會好
    #
    #   如果訓練過程中存在中斷訓練的操作，可以將model_path設置成logs文件夾下的權值文件，將已經訓練了一部分的權值再次載入。
    #   同時修改下方的 凍結階段 或者 解凍階段 的參數，來保證模型epoch的連續性。
    #   
    #   當model_path = ''的時候不加載整個模型的權值。
    #
    #   此處使用的是整個模型的權重，因此是在train.py進行加載的。
    #   如果想要讓模型從0開始訓練，則設置model_path = ''，下面的Freeze_Train = Fasle，此時從0開始訓練，且沒有凍結主幹的過程。
    #   
    #   一般來講，網絡從0開始的訓練效果會很差，因為權值太過隨機，特徵提取效果不明顯，因此非常、非常、非常不建議大家從0開始訓練！
    #   如果一定要從0開始，可以了解imagenet數據集，首先訓練分類模型，獲得網絡的主幹部分權值，分類模型的 主幹部分 和該模型通用，基於此進行訓練。
    #----------------------------------------------------------------------------------------------------------------------------#
    model_path      = 'model_data/Yolov3_weights.h5'
    #------------------------------------------------------#
    #   input_shape     輸入的shape大小，一定要是32的倍數
    #------------------------------------------------------#
    input_shape     = [416, 416]
    
    #----------------------------------------------------------------------------------------------------------------------------#
    #   訓練分為兩個階段，分別是凍結階段和解凍階段。設置凍結階段是為了滿足機器性能不足的同學的訓練需求。
    #   凍結訓練需要的顯存較小，顯卡非常差的情況下，可設置Freeze_Epoch等於UnFreeze_Epoch，此時僅僅進行凍結訓練。
    #      
    #   在此提供若干參數設置建議，各位訓練者根據自己的需求進行靈活調整：
    #   （一）從整個模型的預訓練權重開始訓練： 
    #       Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 100，Freeze_Train = True（默認參數）
    #       Init_Epoch = 0，UnFreeze_Epoch = 100，Freeze_Train = False（不凍結訓練）
    #       其中：UnFreeze_Epoch可以在100-300之間調整。 optimizer_type = 'sgd'，Init_lr = 1e-2。
    #   （二）從主幹網絡的預訓練權重開始訓練：
    #       Init_Epoch = 0，Freeze_Epoch = 50，UnFreeze_Epoch = 300，Freeze_Train = True（凍結訓練）
    #       Init_Epoch = 0，UnFreeze_Epoch = 300，Freeze_Train = False（不凍結訓練）
    #       其中：由於從主幹網絡的預訓練權重開始訓練，主幹的權值不一定適合目標檢測，需要更多的訓練跳出局部最優解。
    #             UnFreeze_Epoch可以在200-300之間調整，YOLOV5和YOLOX均推薦使用300。 optimizer_type = 'sgd'，Init_lr = 1e-2。
    #   （三）batch_size的設置：
    #       在顯卡能夠接受的範圍內，以大為好。顯存不足與數據集大小無關，提示顯存不足（OOM或者CUDA out of memory）請調小batch_size。
    #       受到BatchNorm層影響，batch_size最小為2，不能為1。
    #       正常情況下Freeze_batch_size建議為Unfreeze_batch_size的1-2倍。不建議設置的差距過大，因為關係到學習率的自動調整。
    #----------------------------------------------------------------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   凍結階段訓練參數
    #   此時模型的主幹被凍結了，特徵提取網絡不發生改變
    #   佔用的顯存較小，僅對網絡進行微調
    #   Init_Epoch          模型當前開始的訓練世代，其值可以大於Freeze_Epoch，如設置：
    #                       Init_Epoch = 60、Freeze_Epoch = 50、UnFreeze_Epoch = 100
    #                       會跳過凍結階段，直接從60代開始，並調整對應的學習率。
    #                       （斷點續練時使用）
    #   Freeze_Epoch        模型凍結訓練的Freeze_Epoch
    #                       (當Freeze_Train=False時失效)
    #   Freeze_batch_size   模型凍結訓練的batch_size
    #                       (當Freeze_Train=False時失效)
    #------------------------------------------------------------------#
    Init_Epoch          = 0
    Freeze_Epoch        = 60
    Freeze_batch_size   = 8
    #------------------------------------------------------------------#
    #   解凍階段訓練參數
    #   此時模型的主幹不被凍結了，特徵提取網絡會發生改變
    #   佔用的顯存較大，網絡所有的參數都會發生改變
    #   UnFreeze_Epoch          模型總共訓練的epoch
    #   Unfreeze_batch_size     模型在解凍後的batch_size
    #------------------------------------------------------------------#
    UnFreeze_Epoch      = 120
    Unfreeze_batch_size = 4
    #------------------------------------------------------------------#
    #   Freeze_Train    是否進行凍結訓練
    #                   默認先凍結主幹訓練後解凍訓練。
    #                   如果設置Freeze_Train=False，建議使用優化器為sgd
    #------------------------------------------------------------------#
    Freeze_Train        = False
    
    #------------------------------------------------------------------#
    #   其它訓練參數：學習率、優化器、學習率下降有關
    #------------------------------------------------------------------#
    #------------------------------------------------------------------#
    #   Init_lr         模型的最大學習率
    #                   當使用Adam優化器時建議設置  Init_lr=1e-3
    #                   當使用SGD優化器時建議設置   Init_lr=1e-2
    #   Min_lr          模型的最小學習率，默認為最大學習率的0.01
    #------------------------------------------------------------------#
    Init_lr             = 1e-3
    Min_lr              = Init_lr * 0.01
    #------------------------------------------------------------------#
    #   optimizer_type  使用到的優化器種類，可選的有adam、sgd
    #                   當使用Adam優化器時建議設置  Init_lr=1e-3
    #                   當使用SGD優化器時建議設置   Init_lr=1e-2
    #   momentum        優化器內部使用到的momentum參數
    #   weight_decay    權值衰減，可防止過擬合
    #------------------------------------------------------------------#
    optimizer_type      = "adam"
    momentum            = 0.937
    weight_decay        = 5e-4
    #------------------------------------------------------------------#
    #   lr_decay_type   使用到的學習率下降方式，可選的有'step'、'cos'
    #------------------------------------------------------------------#
    lr_decay_type       = 'cos'
    #------------------------------------------------------------------#
    #   save_period     多少個epoch保存一次權值，默認每個世代都保存
    #------------------------------------------------------------------#
    save_period         = 1
    #------------------------------------------------------------------#
    #   num_workers     用於設置是否使用多線程讀取數據，1代表關閉多線程
    #                   開啟後會加快數據讀取速度，但是會佔用更多內存
    #                   keras裡開啟多線程有些時候速度反而慢了許多
    #                   在IO為瓶頸的時候再開啟多線程，即GPU運算速度遠大於讀取圖片的速度。
    #------------------------------------------------------------------#
    num_workers         = 1

    #------------------------------------------------------#
    #   train_annotation_path   訓練圖片路徑和標籤
    #   val_annotation_path     驗證圖片路徑和標籤
    #------------------------------------------------------#
    train_annotation_path   = '2007_train.txt'
    val_annotation_path     = '2007_val.txt'

    #----------------------------------------------------#
    #   獲取classes和anchor
    #----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors     = get_anchors(anchors_path)

    #------------------------------------------------------#
    #   創建yolo模型
    #------------------------------------------------------#
    model_body  = yolo_body((None, None, 3), anchors_mask, num_classes, weight_decay)
    if model_path != '':
        #------------------------------------------------------#
        #   載入預訓練權重
        #------------------------------------------------------#
        print('Load weights {}.'.format(model_path))
        model_body.load_weights(model_path, by_name=True, skip_mismatch=True)

    if not eager:
        model = get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask)

    #---------------------------#
    #   讀取數據集對應的txt
    #---------------------------#
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    with open(val_annotation_path, encoding='utf-8') as f:
        val_lines   = f.readlines()
    num_train   = len(train_lines)
    num_val     = len(val_lines)
    
    #------------------------------------------------------#
    #   主幹特徵提取網絡特徵通用，凍結訓練可以加快訓練速度
    #   也可以在訓練初期防止權值被破壞。
    #   Init_Epoch為起始世代
    #   Freeze_Epoch為凍結訓練的世代
    #   UnFreeze_Epoch總訓練世代
    #   提示OOM或者顯存不足請調小Batch_size
    #------------------------------------------------------#
    if True:
        if Freeze_Train:
            freeze_layers = 184
            for i in range(freeze_layers): model_body.layers[i].trainable = False
            print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model_body.layers)))
            
        #-------------------------------------------------------------------#
        #   如果不凍結訓練的話，直接設置batch_size為Unfreeze_batch_size
        #-------------------------------------------------------------------#
        batch_size  = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size

        #-------------------------------------------------------------------#
        #   判斷當前batch_size與64的差別，自適應調整學習率
        #-------------------------------------------------------------------#
        nbs     = 64
        Init_lr_fit = max(batch_size / nbs * Init_lr, 1e-4)
        Min_lr_fit  = max(batch_size / nbs * Min_lr, 1e-6)

        #---------------------------------------#
        #   獲得學習率下降的公式
        #---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('數據集過小，無法進行訓練，請擴充數據集。')

        train_dataloader    = YoloDatasets(train_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, train = True)
        val_dataloader      = YoloDatasets(val_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, train = False)

        optimizer = {
            'adam'  : Adam(lr = Init_lr, beta_1 = momentum),
            'sgd'   : SGD(lr = Init_lr, momentum = momentum, nesterov=True)
        }[optimizer_type]
        
        if eager:
            start_epoch     = Init_Epoch
            end_epoch       = UnFreeze_Epoch
            UnFreeze_flag   = False

            gen     = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32, tf.float32, tf.float32))
            gen_val = tf.data.Dataset.from_generator(partial(val_dataloader.generate), (tf.float32, tf.float32, tf.float32, tf.float32))

            gen     = gen.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
            gen_val = gen_val.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)

            time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
            log_dir         = os.path.join('logs', "loss_" + str(time_str))
            loss_history    = LossHistory(log_dir)
            #---------------------------------------#
            #   開始模型訓練
            #---------------------------------------#
            for epoch in range(start_epoch, end_epoch):
                #---------------------------------------#
                #   如果模型有凍結學習部分
                #   則解凍，並設置參數
                #---------------------------------------#
                if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                    batch_size      = Unfreeze_batch_size

                    #-------------------------------------------------------------------#
                    #   判斷當前batch_size與64的差別，自適應調整學習率
                    #-------------------------------------------------------------------#
                    nbs     = 64
                    Init_lr_fit = max(batch_size / nbs * Init_lr, 1e-4)
                    Min_lr_fit  = max(batch_size / nbs * Min_lr, 1e-6)
                    #---------------------------------------#
                    #   獲得學習率下降的公式
                    #---------------------------------------#
                    lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)

                    for i in range(len(model_body.layers)): 
                        model_body.layers[i].trainable = True

                    epoch_step      = num_train // batch_size
                    epoch_step_val  = num_val // batch_size

                    if epoch_step == 0 or epoch_step_val == 0:
                        raise ValueError("數據集過小，無法繼續進行訓練，請擴充數據集。")

                    train_dataloader.batch_size    = batch_size
                    val_dataloader.batch_size      = batch_size

                    gen     = tf.data.Dataset.from_generator(partial(train_dataloader.generate), (tf.float32, tf.float32, tf.float32, tf.float32))
                    gen_val = tf.data.Dataset.from_generator(partial(val_dataloader.generate), (tf.float32, tf.float32, tf.float32, tf.float32))

                    gen     = gen.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
                    gen_val = gen_val.shuffle(buffer_size = batch_size).prefetch(buffer_size = batch_size)
                    
                    UnFreeze_flag = True

                lr = lr_scheduler_func(epoch)
                K.set_value(optimizer.lr, lr)
                K.set_value(optimizer.lr, lr)

                fit_one_epoch(model_body, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, 
                            end_epoch, input_shape, anchors, anchors_mask, num_classes, save_period)

                train_dataloader.on_epoch_end()
                val_dataloader.on_epoch_end()
        else:
            start_epoch = Init_Epoch
            end_epoch   = Freeze_Epoch if Freeze_Train else UnFreeze_Epoch
            
            model.compile(optimizer = optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})
            #-------------------------------------------------------------------------------#
            #   訓練參數的設置
            #   logging         用於設置tensorboard的保存地址
            #   checkpoint      用於設置權值保存的細節，period用於修改多少epoch保存一次
            #   lr_scheduler       用於設置學習率下降的方式
            #   early_stopping  用於設定早停，val_loss多次不下降自動結束訓練，表示模型基本收斂
            #-------------------------------------------------------------------------------#
            time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
            log_dir         = os.path.join('logs', "loss_" + str(time_str))
            logging         = TensorBoard(log_dir)
            loss_history    = LossHistory(log_dir)
            checkpoint      = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor = 'val_loss', save_weights_only = True, save_best_only = False, period = save_period)
            early_stopping  = EarlyStopping(monitor='val_loss', min_delta = 0, patience = 10, verbose = 1)
            lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
            callbacks       = [logging, loss_history, checkpoint, lr_scheduler, early_stopping]

            if start_epoch < end_epoch:
                print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
                model.fit_generator(
                    generator           = train_dataloader,
                    steps_per_epoch     = epoch_step,
                    validation_data     = val_dataloader,
                    validation_steps    = epoch_step_val,
                    epochs              = end_epoch,
                    initial_epoch       = start_epoch,
                    use_multiprocessing = True if num_workers > 1 else False,
                    workers             = num_workers,
                    callbacks           = callbacks
                )
            #---------------------------------------#
            #   如果模型有凍結學習部分
            #   則解凍，並設置參數
            #---------------------------------------#
            if Freeze_Train:
                batch_size  = Unfreeze_batch_size
                start_epoch = Freeze_Epoch if start_epoch < Freeze_Epoch else start_epoch
                end_epoch   = UnFreeze_Epoch
                    
                #-------------------------------------------------------------------#
                #   判斷當前batch_size與64的差別，自適應調整學習率
                #-------------------------------------------------------------------#
                nbs     = 64
                Init_lr_fit = max(batch_size / nbs * Init_lr, 1e-4)
                Min_lr_fit  = max(batch_size / nbs * Min_lr, 1e-6)
                #---------------------------------------#
                #   獲得學習率下降的公式
                #---------------------------------------#
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                lr_scheduler    = LearningRateScheduler(lr_scheduler_func, verbose = 1)
                callbacks       = [logging, loss_history, checkpoint, lr_scheduler, early_stopping]
                    
                for i in range(len(model_body.layers)): 
                    model_body.layers[i].trainable = True
                model.compile(optimizer = optimizer, loss={'yolo_loss': lambda y_true, y_pred: y_pred})

                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("數據集過小，無法繼續進行訓練，請擴充數據集。")

                train_dataloader.batch_size    = Unfreeze_batch_size
                val_dataloader.batch_size      = Unfreeze_batch_size

                print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
                model.fit_generator(
                    generator           = train_dataloader,
                    steps_per_epoch     = epoch_step,
                    validation_data     = val_dataloader,
                    validation_steps    = epoch_step_val,
                    epochs              = end_epoch,
                    initial_epoch       = start_epoch,
                    use_multiprocessing = True if num_workers > 1 else False,
                    workers             = num_workers,
                    callbacks           = callbacks
                )
