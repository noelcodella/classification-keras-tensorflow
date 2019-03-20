# Noel C. F. Codella
# Example Semantic Classification Code for Keras / TensorFlow

# GLOBAL DEFINES
T_G_WIDTH = 224
T_G_HEIGHT = 224 
T_G_NUMCHANNELS = 3
T_G_SEED = 1337

T_G_CHUNKSIZE = 11000

USAGE_LEARN = 'Usage: \n\t -learn <Train Images (TXT)> <Train Vectors (TXT)> <Val Images (TXT)> <Val Vectors (TXT)> <batch size> <num epochs> <output model prefix> <option: load class weights from...> <option: load weights from...> \n\t -extract <Model Prefix> <Input Image List (TXT)> <Output Path> <Optional: GT file for metrics> \n\t\tBuilds and scores a model'

# Misc. Necessities
import sys
import os
import ssl # these two lines solved issues loading pretrained model
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np

import sklearn
import sklearn.metrics
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import cv2
from scipy.misc import imresize
np.random.seed(T_G_SEED)
import scipy

# TensorFlow Includes
import tensorflow as tf
#from tensorflow.contrib.losses import metric_learning
tf.set_random_seed(T_G_SEED)

# Keras Imports & Defines 
import keras
import keras.applications
import keras.optimizers
import keras.losses
from keras import backend as K
from keras.models import Model
from keras import optimizers
import keras.layers as kl
from keras import regularizers

from keras.preprocessing.image import ImageDataGenerator

# Uncomment to use the TensorFlow Debugger
#from tensorflow.python import debug as tf_debug
#sess = K.get_session()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#K.set_session(sess)

# Generator object for data augmentation.
# Can change values here to affect augmentation style.
datagen = ImageDataGenerator(   #rotation_range=10,
                                width_shift_range=0.05,
                                height_shift_range=0.05,
                                zoom_range=0.10,
                                #brightness_range=[0.95,1.05],
                                horizontal_flip=True,
                                vertical_flip=True,
                                )



# generator function for data augmentation
def createDataGen(X, Y, b):

    local_seed = T_G_SEED
    genX = datagen.flow(X,Y, batch_size=b, seed=local_seed, shuffle=False)
    while True:
            Xi = genX.next()

            yield Xi[0], Xi[1]



def createModel(numk=1):

    # Initialize a Model
   
    net_input = kl.Input(shape=(T_G_WIDTH,T_G_HEIGHT,T_G_NUMCHANNELS))
    #net_model = keras.applications.densenet.DenseNet121(weights='imagenet', include_top = False, input_tensor=net_input)
    #net_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top = True, input_tensor=net_input)
    net_model = keras.applications.resnet50.ResNet50(weights='imagenet', include_top = False, input_tensor=net_input)

    # Uncomment to freeze model weights
    #for layer in net_model.layers:
    #   layer.trainable = False

    # New Layers 
    net = net_model.output
    gap = kl.GlobalAveragePooling2D()(net)
    outnet = kl.Dense(numk, activation='sigmoid')(gap)

    # model creation
    base_model = Model(net_model.input, outnet, name="base_model")

    print base_model.summary()

    for layer in base_model.layers:
        print(layer, layer.trainable)

    base_model.compile(optimizer=keras.optimizers.Adadelta(lr=0.1, decay=0.00001), loss=keras.losses.mean_squared_error, metrics=[keras.metrics.categorical_accuracy, keras.losses.categorical_crossentropy])

    return base_model



# loads an image and preprocesses
def t_read_image(loc, prep=1):
    t_image = cv2.imread(loc)
    t_image = cv2.resize(t_image, (T_G_WIDTH,T_G_HEIGHT))
    t_image = t_image.astype("float32")

    if (prep == 1):
        t_image = keras.applications.resnet50.preprocess_input(t_image, data_format='channels_last')

    return t_image

def t_norm_image(img):
    new_img = (img - np.amin(img)) / (np.amax(img) - np.amin(img))

    return new_img

# loads a set of images from a text index file   
def t_read_image_list(flist, start, length, color=1, norm=0, prep=1):

    with open(flist) as f:
        content = f.readlines() 
    content = [x.strip().split()[0] for x in content] 

    datalen = length
    if (datalen < 0):
        datalen = len(content)

    if (start + datalen > len(content)):
        datalen = len(content) - start

    if (color == 1):
        imgset = np.zeros((datalen, T_G_HEIGHT, T_G_WIDTH, T_G_NUMCHANNELS))
    else:
        imgset = np.zeros((datalen, T_G_HEIGHT, T_G_WIDTH, 1))

    for i in range(start, start+datalen):
        if ((i-start) < len(content)):
            val = t_read_image(content[i], prep)
            if (color == 0):
                val = val[:,:,0]
                val = np.expand_dims(val,2)
            imgset[i-start] = val
            if (norm == 1):
                imgset[i-start] = (t_norm_image(imgset[i-start]) * 1.0 + 0.0) 

    return imgset


def file_numlines(fn):
    with open(fn) as f:
        return sum(1 for _ in f)


def main(argv):

    if len(argv) < 2:
        print USAGE_LEARN    
        return

    if 'learn' in argv[0]:
        learn(argv[1:])
    elif 'extract' in argv[0]:
        extract(argv[1:])    

    return




def extract(argv):

    if len(argv) < 3:
        print 'Usage: \n\t <Model Prefix> <Input Image List (TXT)> <Output Path> <Optional: GT file for metrics>\n\t\tExtracts model'
        return

    modelpref = argv[0]
    imglist = argv[1]
    outfile = argv[2]

    with open(modelpref + '.json', "r") as json_file:
        model_json = json_file.read()

    loaded_model = keras.models.model_from_json(model_json)
    loaded_model.load_weights(modelpref + '.h5')

    base_model = loaded_model 

    preds = scoreModel(imglist,base_model,outfile,0)

    
    # An optional example evaluation framework for ISIC 2018 Challenge
    gt = []
    if len(argv) > 3:
        labels = [ 'MEL','NV','BCC','AKIEC','BKL','DF','VASC' ]
        style = ['-', '-', '--' ]
        lw = [3.0, 1.0, 1.0]
        colors = [ 'b', 'g', 'r', 'c', 'm', 'y', 'k' ]
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        
        print 'Loading gt from: ' + argv[3] + ' ... '
        gt = np.loadtxt(argv[3])

        numk = gt.shape[1]

        acc = sklearn.metrics.accuracy_score(np.argmax(gt,axis=1), np.argmax(preds,axis=1))
        bacc = sklearn.metrics.balanced_accuracy_score(np.argmax(gt,axis=1), np.argmax(preds,axis=1))
        
        print 'ACC: ' + str(acc) + '\nBACC: ' + str(bacc) + '\n'

        for i in range(0, numk):
            myroc = sklearn.metrics.roc_curve(gt[:,i], preds[:,i])
            auc = sklearn.metrics.roc_auc_score(gt[:,i], preds[:,i])
            print labels[i] + ' AUC: ' + str(auc) + '\n'
            ax.plot(myroc[0], myroc[1], label=labels[i], linewidth=lw[0], linestyle=style[0], color=colors[i])

        plt.legend()
        plt.xlabel('FPR')
        plt.ylabel('TPR')

        fig.savefig(outfile + '/' + 'ROC.png')   # save the figure to file
        plt.close(fig)    # close the figure


    return

def scoreModel(imglist, base_model, outfile, aug=0):

    chunksize = T_G_CHUNKSIZE
    total_img = file_numlines(imglist)
    total_img_ch = int(np.ceil(total_img / float(chunksize)))

    for i in range(0, total_img_ch):
        imgs = t_read_image_list(imglist, i*chunksize, chunksize)
        valsa = base_model.predict(imgs)
        
        # test time data augmentation
        if (aug > 0):
            valsb = base_model.predict(scipy.ndimage.rotate(imgs, 90, axes=(2,1), reshape=False))
            valsc = base_model.predict(scipy.ndimage.rotate(imgs, 180, axes=(2,1), reshape=False))
            valsd = base_model.predict(scipy.ndimage.rotate(imgs, 270, axes=(2,1), reshape=False))

            vals = (valsa + valsb + valsc + valsd) / 4.0

        else:
            vals = valsa
        

        # class-wise predictions
        np.savetxt(outfile + '/array.batch' + str(i) + '.txt', vals)

    return vals



def learn(argv):
    
    if len(argv) < 6:
        print USAGE_LEARN
        return

    in_t_i = argv[0]
    in_t_m = argv[1]

    in_v_i = argv[2]
    in_v_m = argv[3]

    batch = int(argv[4])
    numepochs = int(argv[5])
    outpath = argv[6] 

    # load the class assignment map
    cmap_t = np.loadtxt(in_t_m)
    cmap_v = np.loadtxt(in_v_m)
    numk = cmap_t.shape[1]

    # chunksize is the number of images we load from disk at a time
    chunksize = T_G_CHUNKSIZE
    total_t = file_numlines(in_t_i)
    total_v = file_numlines(in_v_i)
    total_t_ch = int(np.ceil(total_t / float(chunksize)))
    total_v_ch = int(np.ceil(total_v / float(chunksize)))

    print 'Dataset has ' + str(total_t) + ' training, and ' + str(total_v) + ' validation.'

    print 'Creating a model ...'
    model = createModel(numk)

    cw = []
    if len(argv) > 7:
        print 'Loading class weights from: ' + argv[7] + ' ... '
        cw = np.loadtxt(argv[7])
    else:
        cw = np.ones((numk)) 

    if len(argv) > 8:
        print 'Loading weights from: ' + argv[8] + ' ... '
        model.load_weights(argv[8])

    print 'Training loop ...'
   
    images_t = []
    maps_t = []
    images_v = []
    maps_v = []

    t_imloaded = 0
    v_imloaded = 0
 
    # manual loop over epochs to support very large sets 
    for e in range(0, numepochs):

        for t in range(0, total_t_ch):

            print 'Epoch ' + str(e) + ': train chunk ' + str(t+1) + '/ ' + str(total_t_ch) + ' ...'

            if ( t_imloaded == 0 or total_t_ch > 1 ): 
                print 'Reading image lists ...'
                images_t = t_read_image_list(in_t_i, t*chunksize, chunksize)
                maps_t = cmap_t[t*chunksize:(t+1)*chunksize,:] 
                t_imloaded = 1

            print 'Starting to fit ...'

            # This method uses data augmentation
            model.fit_generator(generator=createDataGen(images_t,maps_t,batch), steps_per_epoch=len(images_t) / batch, epochs=1, shuffle=False, use_multiprocessing=True)
        
        # In case the validation images don't fit in memory, we load chunks from disk again. 
        val_res = [0.0, 0.0]
        total_w = 0.0
        for v in range(0, total_v_ch):

            print 'Epoch ' + str(e) + ': val chunk ' + str(v+1) + '/ ' + str(total_v_ch) + ' ...'

            if ( v_imloaded == 0 or total_v_ch > 1 ):
                print 'Loading validation image lists ...'
                images_v = t_read_image_list(in_v_i, v*chunksize, chunksize)
                maps_v = cmap_v[v*chunksize:(v+1)*chunksize,:]
                v_imloaded = 1

            # Weight of current validation measurement. 
            # if loaded expected number of items, this will be 1.0, otherwise < 1.0, and > 0.0.
            w = float(images_v.shape[0]) / float(chunksize)
            total_w = total_w + w

            curval = model.evaluate(images_v, maps_v, batch_size=batch)
            val_res[0] = val_res[0] + w*curval[0]
            val_res[1] = val_res[1] + w*curval[1]

        val_res = [x / total_w for x in val_res]

        print 'Validation Metrics: ' + str(model.metrics_names)
        print 'Validation Results: ' + str(val_res)

    print 'Saving model ...'

    # Save the model and weights
    model.save(outpath + '.h5')

    # Due to some remaining Keras bugs around loading custom optimizers
    # and objectives, we save the model architecture as well
    model_json = model.to_json()
    with open(outpath + '.json', "w") as json_file:
        json_file.write(model_json)


    return


# Main Driver
if __name__ == "__main__":
    main(sys.argv[1:])
