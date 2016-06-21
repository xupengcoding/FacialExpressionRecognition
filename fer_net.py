'''this is a Inception architecture for facial expression recongnition'''

from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization, Flatten, Dense, Dropout
from keras.layers import Input, merge
from keras.models import Model
from keras import regularizers

#global constants
NB_CLASS = 7 #number of facial expression
DIM_ORERING = 'th' # 'th' (channels, width, height) 'tf' (width, height, channels)
WEIGHT_DECAY = 1. #L2 regularization factor
USE_BN = True #batch normalization


#conv2D with bn

def conv2D_bn(x, nb_filter, nb_row, nb_col,
              border_mode='same', subsample=(1, 1),
              activation='relu', batch_norm=USE_BN,
              weight_decay=WEIGHT_DECAY, dim_ordering=DIM_ORERING):

    if weight_decay:
        W_regularizer = regularizers.l2(weight_decay)
        b_regularizer = regularizers.l2(weight_decay)
    else:
        W_regularizer = None
        b_regularizer = None

    #conv2D
    x = Convolution2D(nb_filter, nb_row, nb_col,
                      subsample=subsample,
                      activation=activation,
                      border_mode=border_mode,
                      W_regularizer=W_regularizer,
                      b_regularizer=b_regularizer,
                      dim_ordering=dim_ordering)(x)

    #batch normalization
    if batch_norm:
        x = BatchNormalization()(x)
    return x

def fer_net_model():
    if DIM_ORERING == 'th':
        img_input = Input(shape=(3, 48, 48)) #all facial expression img is resize to (48,48),
                                             #fer is gray pic, others are three dim
        CONCAT_AXIS = 1
    elif DIM_ORERING == 'tf':
        img_input = Input(shape=(48, 48, 3))
        CONCAT_AXIS = 3
    else:
        raise Exception('Invalid dim ordering: ' + str(DIM_ORERING))

    #construct the model
    #Convolution-1 7x7/2 24
    x = conv2D_bn(img_input, 24, 7, 7, subsample=(2, 2), border_mode='valid')
    #Max pool-1 3x3/2 64
    x = MaxPooling2D(x, 64, 3, 3, strides=(2, 2), border_mode='valid', dim_ordering=DIM_ORERING)
    #Convolution-2 3x3/1 192
    x = conv2D_bn(x, 192, 3, 3, border_mode='valid')
    #Max pool-2 3x3/2 192
    x = MaxPooling2D(x, 192, 3, 3, strides=(2, 2), dim_ordering=DIM_ORERING)

    #Inception-3a
    #branch1x1 1x1 64
    branch1x1 = conv2D_bn(x, 64, 1, 1)
    #branch 3x3
    #reduce 3x3 96
    #conv 3x3 128
    branch3x3 = conv2D_bn(x, 96, 3, 3)
    branch3x3 = conv2D_bn(branch3x3, 128, 3, 3)
    #branch 5x5
    #reduce 5x5 16
    #conv 5x5 32
    branch5x5 = conv2D_bn(x, 16, 5, 5)
    branch5x5 = conv2D_bn(branch5x5, 32, 5, 5)
    #branch proj Pooling
    #max pool 3x3
    #conv 1x1 32
    branch_pool = MaxPooling2D((3,3), strides=(1, 1), border_mode='same', dim_ordering=DIM_ORERING)(x)
    branch_pool = conv2D_bn(branch_pool, 32, 1, 1)
    #concat
    x = merge([branch1x1, branch3x3, branch5x5, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

    #Inception 3b
    #branch1x1 1x1 128
    branch1x1 = conv2D_bn(x, 128, 1, 1)
    #branch 3x3
    #reduce 3x3 128
    #conv 3x3 192
    branch3x3 = conv2D_bn(x, 128, 3, 3)
    branch3x3 = conv2D_bn(branch3x3, 192, 3, 3)
    #branch 5x5
    #reduce 5x5 32
    #conv 5x5 96
    branch5x5 = conv2D_bn(x, 32, 5, 5)
    branch5x5 = conv2D_bn(branch5x5, 96, 5, 5)
    #branch proj Pooling
    #max pool 3x3
    #conv 1x1 64
    branch_pool = MaxPooling2D((3,3), strides=(1, 1), border_mode='same', dim_ordering=DIM_ORERING)(x)
    branch_pool = conv2D_bn(branch_pool, 64, 1, 1)
    #concat
    x = merge([branch1x1, branch3x3, branch5x5, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

    #max pool - 4 3x3 2
    x = MaxPooling2D((3,3), strides=(2, 2), border_mode='valid', dim_ordering=DIM_ORERING)(x)

    #Inception 4a
    #branch1x1 1x1 192
    branch1x1 = conv2D_bn(x, 64, 1, 1)
    #branch 3x3
    #reduce 3x3 96
    #conv 3x3 208
    branch3x3 = conv2D_bn(x, 96, 3, 3)
    branch3x3 = conv2D_bn(branch3x3, 208, 3, 3)
    #branch 5x5
    #reduce 5x5 16
    #conv 5x5 48
    branch5x5 = conv2D_bn(x, 16, 5, 5)
    branch5x5 = conv2D_bn(branch5x5, 48, 5, 5)
    #branch proj Pooling
    #max pool 3x3
    #conv 1x1 64
    branch_pool = MaxPooling2D((3,3), strides=(1, 1), border_mode='same', dim_ordering=DIM_ORERING)(x)
    branch_pool = conv2D_bn(branch_pool, 64, 1, 1)
    #concat
    x = merge([branch1x1, branch3x3, branch5x5, branch_pool], mode='concat', concat_axis=CONCAT_AXIS)

    #avg pool 6 1x1 1024
    x = AveragePooling2D((3, 3), strides=(1, 1), border_mode='valid', dim_ordering=DIM_ORERING)(x)

    #flatten
    x = Flatten(x)
    #full connected 7 4096
    x = Dense(4096, activation='relu')(x)
    #full connected 8 1024
    x = Dense(1024, activation='relu')(x)

    #classification
    preds = Dense(NB_CLASS, activation='softmax')(x)

    model = Model(input=img_input, output=preds)
    model.compile('rmsporp', 'categorical_crossentropy')
    return model

















