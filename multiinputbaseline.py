import os
import sys
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # This line is commented out to use gpu, uncommented is to use cpu
import tensorflow as tf
from tensorflow.keras.preprocessing import image
gpus = tf.config.list_physical_devices("GPU")
print(gpus)
if gpus:
    gpu0 = gpus[0]  # If there are multiple GPUs, only the 0th GPU is used
    tf.config.experimental.set_memory_growth(gpu0, True)  # GPU memory usage on demand
    tf.config.set_visible_devices([gpu0], "GPU")
from matplotlib.pyplot import imshow
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from mmd import MMD
from tensorflow.keras.layers import Flatten
import matplotlib.pyplot as pyplot
import matplotlib
import matplotlib.pyplot as plt
import keras_metrics as km
# import tensorflow
from tensorflow.keras import regularizers
from tensorflow.keras.layers import GlobalMaxPooling1D,GlobalMaxPooling2D,Permute,Lambda,Concatenate,multiply,GlobalAveragePooling2D,Reshape,Add
# from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import concatenate
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv1D, MaxPooling1D
from tensorflow.keras.layers import BatchNormalization
# from sklearn.preprocessing import OneHotEncoder  # LabelBinarizer
from multiinputmodel import AdaBoostClassifiermulti as Ada_CNN
import cv2

trainend=1000
trainend1=65000
testend=87300
testshape=testend-trainend1
#所有数据
X_data1=np.load('X_dataall1.npy')
X_train1=X_data1[0:trainend,:,:].reshape((trainend, 29, 78, 1))
X_test1=X_data1[trainend1:testend,:,:].reshape((testshape, 29, 78, 1))
X_data2=np.load('X_dataall2.npy')
X_train2=X_data2[0:trainend,:,:].reshape((trainend, 29, 78, 1))
X_test2=X_data2[trainend1:testend,:,:].reshape((testshape, 29, 78, 1))
X_data3=np.load('X_dataall3.npy')
X_train3=X_data3[0:trainend,:].reshape((trainend, 64,1))
X_test3=X_data3[trainend1:testend,:].reshape((testshape, 64,1))
X_data4=np.load('X_dataall4.npy')
X_train4=X_data4[0:trainend,:].reshape((trainend,64,1))
X_test4=X_data4[trainend1:testend,:].reshape((testshape,64,1))
Y_data=np.load('Y_labelall.npy')
Y_data=Y_data[0]
y_train=Y_data[0:trainend]
y_test=Y_data[trainend1:testend]

def cbam_block(cbam_feature, ratio=8):#attention module
    cbam_feature = channel_attention(cbam_feature, ratio)#channel_attention
    cbam_feature = spatial_attention(cbam_feature)#spatial_attention
    return cbam_feature
def channel_attention(input_feature, ratio=8):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]
    shared_layer_one = Dense(channel // ratio,
                             activation='relu',
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')
    shared_layer_two = Dense(channel,
                             kernel_initializer='he_normal',
                             use_bias=True,
                             bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1, 1, channel))(avg_pool)
    # assert avg_pool.shape[1:] == (1, 1, channel)
    avg_pool = shared_layer_one(avg_pool)
    # assert avg_pool.shape[1:] == (1, 1, channel // ratio)
    avg_pool = shared_layer_two(avg_pool)
    # assert avg_pool.shape[1:] == (1, 1, channel)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1, 1, channel))(max_pool)
    # assert max_pool.shape[1:] == (1, 1, channel)
    max_pool = shared_layer_one(max_pool)
    # assert max_pool.shape[1:] == (1, 1, channel // ratio)
    max_pool = shared_layer_two(max_pool)
    # assert max_pool.shape[1:] == (1, 1, channel)
    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])
def spatial_attention(input_feature):
    kernel_size = 3
    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        cbam_feature = Permute((2, 3, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        cbam_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(cbam_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(cbam_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    cbam_feature = Conv2D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert cbam_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, cbam_feature])
def lossall(model):
    def binary_loss(y_true, y_pred):
        loss1 = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        return loss1
    return binary_loss
    # loss1 = tf.keras.losses.BinaryCrossentropy
    fea1=model.get_layer('conv2d').output
    fea2 = model.get_layer('conv2d_4').output
    fea3 = model.get_layer('conv2d_1').output
    fea4 = model.get_layer('conv2d_5').output
    fea5 = model.get_layer('conv2d_2').output
    fea6 = model.get_layer('conv2d_6').output
    lossconv1 = tf.square(MMD(tf.reshape(fea1[:,:,:,1], [15, 39]), tf.reshape(fea2[:,:,:,1], [15, 39]), beta=1))#maximum mean discrepancy
    lossconv2 = tf.square(MMD(tf.reshape(fea3[:, :, :, 1], [7, 19]), tf.reshape(fea4[:, :, :, 1], [7, 19]), beta=1))
    lossconv3 = tf.square(MMD(tf.reshape(fea5[:, :, :, 1], [3, 9]), tf.reshape(fea6[:, :, :, 1], [3, 9]), beta=1))
    loss=1*lossconv1/1+1*lossconv2/1+1*lossconv3/1+loss1
    print(lossconv1)
    return loss
def baseline_model():
    visible1 = Input(shape=(29, 78, 1))
    # sam11 = spatial_attention(visible1)
    conv11 = Conv2D(2, kernel_size=16,padding= "same",strides=2)(visible1)
    BatchNormalization11 = BatchNormalization()(conv11)
    act11 = Activation('relu')(BatchNormalization11)
    pool11 = MaxPooling2D(pool_size=2,strides=2)(act11)
    conv12 = Conv2D(4, kernel_size=8,padding= "same")(pool11)
    BatchNormalization12 = BatchNormalization()(conv12)
    act12 = Activation('relu')(BatchNormalization12)
    pool12 = MaxPooling2D(pool_size=2,strides=2)(act12)
    conv13 = Conv2D(8, kernel_size=4, padding="same")(pool12)
    BatchNormalization13 = BatchNormalization()(conv13)
    act13 = Activation('relu')(BatchNormalization13)
    pool13 = MaxPooling2D(pool_size=3,strides=2)(act13)
    sam1 = cbam_block(pool13)
    # sam1=pool13
    # pool13 = MaxPooling2D(pool_size=4,strides=2)(act13)
    flat11 = GlobalMaxPooling2D()(sam1)
    # flat12 = GlobalMaxPooling2D()(sam11)
    # flat13 = GlobalMaxPooling2D()(sam12)
    # flat1 = Flatten()(pool13)
    visible2 = Input(shape=(29,78,1))
    conv21 = Conv2D(2, kernel_size=16, padding="same", strides=2)(visible2)
    BatchNormalization21 = BatchNormalization()(conv21)
    act21 = Activation('relu')(BatchNormalization21)
    pool21 = MaxPooling2D(pool_size=2, strides=2)(act21)
    conv22 = Conv2D(4, kernel_size=8, padding="same")(pool21)
    BatchNormalization22 = BatchNormalization()(conv22)
    act22 = Activation('relu')(BatchNormalization22)
    pool22 = MaxPooling2D(pool_size=2, strides=2)(act22)
    conv23 = Conv2D(8, kernel_size=4, padding="same")(pool22)
    BatchNormalization23 = BatchNormalization()(conv23)
    act23 = Activation('relu')(BatchNormalization23)
    pool23 = MaxPooling2D(pool_size=3, strides=2)(act23)
    sam2 = cbam_block(pool23)
    # sam2=pool23
    flat21 = GlobalMaxPooling2D()(sam2)
    visible3 = Input(shape=(64, 1))
    conv31 = Conv1D(2, kernel_size=32,padding= "same",strides=2)(visible3)
    BatchNormalization31 = BatchNormalization()(conv31)
    act31 = Activation('relu')(BatchNormalization31)
    pool31 = MaxPooling1D(pool_size=2,strides=2)(act31)
    conv32 = Conv1D(4, kernel_size=16,padding= "same",strides=2)(pool31)
    BatchNormalization32 = BatchNormalization()(conv32)
    act32 = Activation('relu')(BatchNormalization32)
    pool32 = MaxPooling1D(pool_size=2,strides=2)(act32)
    conv33 = Conv1D(8, kernel_size=8, padding="same")(pool32)
    BatchNormalization33 = BatchNormalization()(conv33)
    act33 = Activation('relu')(BatchNormalization33)
    pool33 = MaxPooling1D(pool_size=3, strides=2)(act33)
    flat3 = GlobalMaxPooling1D()(pool33)
    # flat3 = Flatten()(pool33)
    # second input model
    visible4 = Input(shape=(64, 1))
    conv41 = Conv1D(2, kernel_size=32, padding="same", strides=2)(visible4)
    BatchNormalization41 = BatchNormalization()(conv41)
    act41 = Activation('relu')(BatchNormalization41)
    pool41 = MaxPooling1D(pool_size=2, strides=2)(act41)
    conv42 = Conv1D(4, kernel_size=16, padding="same", strides=2)(pool41)
    BatchNormalization42 = BatchNormalization()(conv42)
    act42 = Activation('relu')(BatchNormalization42)
    pool42 = MaxPooling1D(pool_size=2, strides=2)(act42)
    conv43 = Conv1D(8, kernel_size=8, padding="same")(pool42)
    BatchNormalization43 = BatchNormalization()(conv43)
    act43 = Activation('relu')(BatchNormalization43)
    pool43 = MaxPooling1D(pool_size=4,strides=2)(act43)
    flat4 = GlobalMaxPooling1D()(pool43)
    # flat4 = Flatten()(pool43)
    # merge input models
    # merge = concatenate([flat1, flat2,flat3,flat4])
    merge = concatenate([flat11,flat21,flat3,flat4])
    # # interpretation model
    hidden1 = Dense(16, activation='relu')(merge)
    # # # 在分类器之前使用
    dropout1=Dropout(0.5)(hidden1)
    hidden2 = Dense(8, activation='relu',kernel_regularizer=regularizers.l2(0.01))(dropout1)
    output = Dense(2, activation='softmax')(hidden2)
    model = Model(inputs=[visible1, visible2,visible3,visible4], outputs=output)
    model.summary()
    adam = optimizers.Adam(learning_rate=0.003)
    model.compile(loss=[lossall(model)], optimizer='adam', metrics=[km.f1_score(), km.precision(), km.recall()])
    # optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=util.learning_rate).minimize(loss)
    # adam=optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # model.compile(loss='BinaryCrossentropy', optimizer='adam', metrics=[km.f1_score(), km.precision(), km.recall()])  # adagrad metrics=['accuracy'])
    return model
list1=[]
batch_size = [32]#8,16,32,64,128,256 ...
n_estimators =[15]#5,10,15,20,25,30,40,50...
epochs = [25]#15,20,25,30,40,50...
a=np.linspace(1,2,5)
aa=a.tolist()
b=np.linspace(3,10,7)
bb=a.tolist()
learning_rate=[0.1]#sample weight 0.1,0.5,1,1.5,2,3...
imbalance=[3]#initial 0.1,0.5,1,3,5,10...
probacoefficient=[1]
algrithom=['SAMME.R']

for i in range(len(n_estimators)):
    for z in range(len(epochs)):
        for a in range(len(learning_rate)):
            for k in range(len(batch_size)):
                for f in range(len(imbalance)):
                    for j in range(len(probacoefficient)):
                        for l in range(len(algrithom)):
                                bdt_real_test_CNN = Ada_CNN(
                                base_estimator=baseline_model(),
                                n_estimators=n_estimators[i],
                                learning_rate=learning_rate[a],
                                imbalance=imbalance[f],
                                probacoefficient=probacoefficient[j],
                                algorithm=algrithom[l],
                                epochs=epochs[z])
                                time_start = time.time()
                                bdt_real_test_CNN.fit([X_train1,X_train2,X_train3,X_train4],y_train,batch_size[k])
                                time_end = time.time()
                                print(time_end-time_start)
                                # joblib.dump(bdt_real_test_CNN, "train_model.m")#save
                                # bdt_real_test_CNN = joblib.load("train_model.m")
                                # test_real_errors_CNN=bdt_real_test_CNN.estimator_errors_[:]
                                y_pred_CNN = bdt_real_test_CNN.predict([X_test1,X_test2,X_test3,X_test4])
                                print('\n Training accuracy of bdt_real_test_CNN (AdaBoost+CNN): {}'.format(
                                accuracy_score(y_pred_CNN, y_test)))
                                # y_pred_CNN = bdt_real_test_CNN.predict([X_test1, X_test2, X_test3, X_test4])
                                p_class, r_class, f_class, support_micro = precision_recall_fscore_support(y_pred_CNN, y_test, average=None)
                                print(p_class,r_class,f_class)
                                line = [p_class[1],r_class[1],f_class[1]]
                                list1.append(line)
                                print(list1)
                                indexerror=np.arange(0,len(y_pred_CNN))
                                classerror=indexerror[y_pred_CNN != y_test]
                                IndexError = [y_test[index] for index in classerror]
                                ERROR=y_test[classerror]
                                # for i in range(0,len(classerror)):
                                #         plt.subplot(2, 2, 1)
                                #         plt.imshow(X_test1[classerror[i], :, :, :])
                                #         plt.subplot(2, 2, 2)
                                #         plt.imshow(X_test2[classerror[i], :, :, :])
                                #         plt.subplot(2, 2, 3)
                                #         plt.plot(X_test3[classerror[i], :,:])
                                #         plt.subplot(2, 2, 4)
                                #         plt.plot(X_test4[classerror[i], :,:])
                                #         plt.show()
                                # aaaaaa = bdt_real_test_CNN.estimators_[0]
                                # # time_start1 = time.time()
                                # y_pred1111 = aaaaaa.predict([X_test1, X_test2, X_test3, X_test4])
                                # # time_end1 = time.time()
                                # # time1=time_end1-time_start1
                                # # print(time1)
                                # y_pred111 = np.argmax(y_pred1111, axis=1)
                                # p_class1, r_class1, f_class1, support_micro1 = precision_recall_fscore_support(y_pred111,
                                #                                                                         y_test,
                                #                                                                         average=None)

time111=time_end-time_start

