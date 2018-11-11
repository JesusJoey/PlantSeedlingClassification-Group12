import tensorflow as tf
from keras.applications import xception
from keras.preprocessing import image 
import keras.preprocessing.image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import sklearn.ensemble
import numpy as np
import pandas as pd
import datetime as dt
from tqdm import tqdm
from glob import glob
import cv2
import os
import seaborn as sns
import mpl_toolkits.axes_grid1
import matplotlib
import matplotlib.pyplot as plt
import datetime

#%matplotlib inline
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 16

# start timer
global_start = datetime.datetime.now()

# validation set size
valid_set_size_percentage = 10 # default = 10%

# Initial operation
load_bf_train=True
load_bf_test=False

take_train_samples= True
take_test_samples= True
num_test_samples= 200

show_plot=True

## read train and test data

# directories
cw_dir = os.getcwd()
data_dir = 'C:\\Galaxy\\Tools\\Scripts\\PlantSeedling'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

# different species in the data set
species = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat',
           'Fat Hen', 'Loose Silky-bent', 'Maize', 'Scentless Mayweed',
           'Shepherds Purse', 'Small-flowered Cranesbill', 'Sugar beet']
num_species = len(species)

# print number of images of each species in the training data
for sp in species:
    print('{} images of {}'.format(len(os.listdir(os.path.join(train_dir, sp))),sp))
    
# read all train data
train = []
for species_id, sp in enumerate(species):
    for file in os.listdir(os.path.join(train_dir, sp)):
        train.append(['train/{}/{}'.format(sp, file), file, species_id, sp])
train_df = pd.DataFrame(train, columns=['filepath', 'file', 'species_id', 'species'])
print('')
print('train_df.shape = ', train_df.shape)

# read all test data
test = []
for file in os.listdir(test_dir):
    test.append(['test/{}'.format(file), file])
test_df = pd.DataFrame(test, columns=['filepath', 'file'])
print('test_df.shape = ', test_df.shape)

def read_image(filepath, target_size=None):
    img = cv2.imread(os.path.join(data_dir, filepath), cv2.IMREAD_COLOR)
    img = cv2.resize(img.copy(), target_size, interpolation = cv2.INTER_AREA)
    #img = image.load_img(os.path.join(data_dir, filepath),target_size=target_size)
    #img = image.img_to_array(img)
    return img

# print train data
print(train_df.describe())
train_df.head()
if show_plot:
    fig = plt.figure(1, figsize=(num_species, num_species))
    grid = mpl_toolkits.axes_grid1.ImageGrid(fig, 111, nrows_ncols=(num_species, num_species), 
                                             axes_pad=0.05)
    i = 0
    for species_id, sp in enumerate(species):
        for filepath in train_df[train_df['species'] == sp]['filepath'].values[:num_species]:
            ax = grid[i]
            img = read_image(filepath, (224, 224))
            ax.imshow(img.astype(np.uint8))
            ax.axis('off')
            if i % num_species == num_species - 1:
                ax.text(250, 112, sp, verticalalignment='center')
            i += 1
    plt.show()
if take_train_samples:
    train_df = pd.concat([train_df[train_df['species'] == sp][:200] for sp in species])
    train_df.index = np.arange(len(train_df))


if take_test_samples:
    test_df = test_df[:num_test_samples]

## detect and segment plants in the image 

def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def segment_plant(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask = mask)
    return output
def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp
def read_segmented_image(filepath, img_size):
    img = cv2.imread(os.path.join(data_dir, filepath), cv2.IMREAD_COLOR)
    img = cv2.resize(img.copy(), img_size, interpolation = cv2.INTER_AREA)

    image_mask = create_mask_for_plant(img)
    image_segmented = segment_plant(img)
    image_sharpen = sharpen_image(image_segmented)
    return img, image_mask, image_segmented, image_sharpen

if show_plot:
    for i in range(5):
        img, image_mask, image_segmented, image_sharpen = read_segmented_image(
            train_df.loc[i,'filepath'],(224,224))
        
        fig, axs = plt.subplots(1, 2, figsize=(20, 20))
        axs[0].imshow(img.astype(np.uint8))
        axs[1].imshow(image_sharpen.astype(np.uint8))

## read and preprocess all training/validation/test images and labels

def preprocess_image(img):
    img /= 255.
    img -= 0.5
    img *= 2
    return img

target_image_size = 299

# read, preprocess training and validation images  
if not load_bf_train:
    x_train_valid = np.zeros((len(train_df), target_image_size, target_image_size, 3),
                             dtype='float32')
    y_train_valid = train_df.loc[:, 'species_id'].values
    for i, filepath in tqdm(enumerate(train_df['filepath'])):
        # read original images
        # img = read_image(filepath, (target_image_size, target_image_size))

        # read segmented image
        _, _, _, img = read_segmented_image(filepath, (299, 299))

        # all pixel values are now between -1 and 1
        x_train_valid[i] = preprocess_image(np.expand_dims(img.copy().astype(np.float), axis=0))

# read, preprocess test images  
x_test = np.zeros((len(test_df), target_image_size, target_image_size, 3), dtype='float32')
for i, filepath in tqdm(enumerate(test_df['filepath'])):
    
    # read original image
    #img = read_image(filepath, (target_image_size, target_image_size))
    
    # read segmented image
    _,_,_,img = read_segmented_image(filepath, (299, 299))
    
    # all pixel values are now between -1 and 1
    x_test[i] = preprocess_image(np.expand_dims(img.copy().astype(np.float), axis=0)) 
    
# print('x_train_valid.shape = ', x_train_valid.shape)
# print('x_test.shape = ', x_test.shape)
'''
cache_dir = os.path.expanduser(os.path.join('~', '.keras'))
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)
models_dir = os.path.join(cache_dir, 'models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)
'''

if not load_bf_train:

    print('x_train_valid.shape = ', x_train_valid.shape)
    print('y_train_valid.shape = ', y_train_valid.shape)
    print('')


    print('compute bottleneck features from Xception network')

    local_start = datetime.datetime.now()
    
    # load xception base model and predict the last layer comprising 2048 neurons per image
    base_model = xception.Xception(weights='imagenet', include_top=False, pooling='avg')
    x_train_valid_bf = base_model.predict(x_train_valid, batch_size=32, verbose=1)

    print('running time: ', datetime.datetime.now()-local_start)    
    print('')
    print('x_train_valid_bf.shape = ', x_train_valid_bf.shape)
    print('')
    print('save bottleneck features and labels for later ')
    np.save(os.path.join(os.getcwd(),'x_train_valid_bf.npy'), x_train_valid_bf)
    np.save(os.path.join(os.getcwd(),'y_train_valid.npy'), y_train_valid)

    # compute bottleneck features from xception model
else:
    print('load bottleneck features and labels')
    
    x_train_valid_bf = np.load(os.path.join(os.getcwd(),'x_train_valid_bf.npy'))
    y_train_valid = np.load(os.path.join(os.getcwd(),'y_train_valid.npy'))

    print('x_train_valid_bf.shape = ', x_train_valid_bf.shape)
    print('y_train_valid.shape = ', y_train_valid.shape)
    
local_start = datetime.datetime.now()
    
# load xception base model and predict the last layer comprising 2048 neurons per image
base_model = xception.Xception(weights='imagenet', include_top=False, pooling='avg')
x_test_bf = base_model.predict(x_test, batch_size=32, verbose=1)
    
print('running time: ', datetime.datetime.now()-local_start)    
print('')
print('x_test_bf = ',x_test_bf.shape)

print('save bottleneck features ')
np.save(os.path.join(os.getcwd(),'x_test_bf.npy'), x_test_bf)

def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

def one_hot_to_dense(labels_one_hot):
    num_labels = labels_one_hot.shape[0]
    num_classes = labels_one_hot.shape[1]
    labels_dense = np.where(labels_one_hot == 1)[1]      
    return labels_dense

# function to shuffle randomly train and validation data
def shuffle_train_valid_data():
    
    print('shuffle train and validation data')
    
    # shuffle train and validation data of original data
    perm_array = np.arange(len(x_train_valid_bf)) 
    np.random.shuffle(perm_array)
    
    # split train and validation sets based on original data
    x_train_bf = x_train_valid_bf[perm_array[:train_set_size]]
    y_train = dense_to_one_hot(y_train_valid[perm_array[:train_set_size]], num_species)
    x_valid_bf = x_train_valid_bf[perm_array[-valid_set_size:]]
    y_valid = dense_to_one_hot(y_train_valid[perm_array[-valid_set_size:]], num_species)

    return x_train_bf, y_train, x_valid_bf, y_valid 

if valid_set_size_percentage > 0:
    # split into train and validation sets
    valid_set_size = int(len(x_train_valid_bf) * valid_set_size_percentage/100);
    train_set_size = len(x_train_valid_bf) - valid_set_size;
else:
    # train on all available data
    valid_set_size = int(len(x_train_valid_bf) * 0.1);
    train_set_size = len(x_train_valid_bf)

# split into train and validation sets including shuffling
x_train_bf, y_train, x_valid_bf, y_valid = shuffle_train_valid_data() 

print('x_train_bf.shape = ', x_train_bf.shape)
print('y_train.shape = ', y_train.shape)
print('x_valid_bf.shape = ', x_valid_bf.shape)
print('y_valid.shape = ', y_valid.shape)



## neural network with tensorflow

# permutation array for shuffling train data
perm_array_train = np.arange(len(x_train_bf)) 
index_in_epoch = 0

# function: to get the next mini batch
def get_next_batch(batch_size):
    
    global index_in_epoch, perm_array_train
  
    start = index_in_epoch
    index_in_epoch += batch_size
    
    if index_in_epoch > train_set_size:
        np.random.shuffle(perm_array_train) # shuffle data
        start = 0 # start next epoch
        index_in_epoch = batch_size
              
    end = index_in_epoch
    
    return x_train_bf[perm_array_train[start:end]], y_train[perm_array_train[start:end]]

x_size = x_train_bf.shape[1] # number of features
y_size = num_species # binary variable
n_n_fc1 = 1024 # number of neurons of first layer
n_n_fc2 = num_species # number of neurons of second layer

# variables for input and output 
x_data = tf.placeholder('float', shape=[None, x_size])
y_data = tf.placeholder('float', shape=[None, y_size])

# 1.layer: fully connected
W_fc1 = tf.Variable(tf.truncated_normal(shape = [x_size, n_n_fc1], stddev = 0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape = [n_n_fc1]))  
h_fc1 = tf.nn.relu(tf.matmul(x_data, W_fc1) + b_fc1)

# add dropout
tf_keep_prob = tf.placeholder('float')
h_fc1_drop = tf.nn.dropout(h_fc1, tf_keep_prob)

# 3.layer: fully connected
W_fc2 = tf.Variable(tf.truncated_normal(shape = [n_n_fc1, n_n_fc2], stddev = 0.1)) 
b_fc2 = tf.Variable(tf.constant(0.1, shape = [n_n_fc2]))  
z_pred = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

# cost function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_data, 
                                                                       logits=z_pred));

# optimisation function
tf_learn_rate = tf.placeholder(dtype='float', name="tf_learn_rate")
train_step = tf.train.AdamOptimizer(tf_learn_rate).minimize(cross_entropy)

# evaluation
y_pred = tf.cast(tf.nn.softmax(z_pred), dtype = tf.float32);
y_pred_class = tf.cast(tf.argmax(y_pred,1), tf.int32)
y_data_class = tf.cast(tf.argmax(y_data,1), tf.int32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(y_pred_class, y_data_class), tf.float32))

# parameters
cv_num = 1 # number of cross validations
n_epoch = 15 # number of epochs
batch_size = 50 
keep_prob = 0.33 # dropout regularization with keeping probability
learn_rate_range = [0.01,0.005,0.0025,0.001,0.001,0.001,0.00075,0.0005,0.00025,0.0001,
                   0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001,0.0001];
learn_rate_step = 3 # in terms of epochs

acc_train_DNN = 0
acc_valid_DNN = 0
loss_train_DNN = 0
loss_valid_DNN = 0
y_test_pred_proba_DNN = 0
y_valid_pred_proba = 0

# use cross validation
for j in range(cv_num):
    
    # start TensorFlow session and initialize global variables
    sess = tf.InteractiveSession() 
    sess.run(tf.global_variables_initializer())  

    # shuffle train/validation splits
    shuffle_train_valid_data() 
    n_step = -1;

    # training model
    for i in range(int(n_epoch*train_set_size/batch_size)):

        if i%int(learn_rate_step*train_set_size/batch_size) == 0:
            n_step += 1;
            learn_rate = learn_rate_range[n_step];
            print('learnrate = ', learn_rate)
        
        x_batch, y_batch = get_next_batch(batch_size)
        
        sess.run(train_step, feed_dict={x_data: x_batch, y_data: y_batch, 
                                        tf_keep_prob: keep_prob, 
                                        tf_learn_rate: learn_rate})

        if i%int(0.25*train_set_size/batch_size) == 0:
            
            train_loss = sess.run(cross_entropy,
                                  feed_dict={x_data: x_train_bf[:valid_set_size], 
                                             y_data: y_train[:valid_set_size], 
                                             tf_keep_prob: 1.0})

            
            train_acc = accuracy.eval(feed_dict={x_data: x_train_bf[:valid_set_size], 
                                                 y_data: y_train[:valid_set_size], 
                                                 tf_keep_prob: 1.0})    

            valid_loss = sess.run(cross_entropy,feed_dict={x_data: x_valid_bf, 
                                                           y_data: y_valid, 
                                                           tf_keep_prob: 1.0})

           
            valid_acc = accuracy.eval(feed_dict={x_data: x_valid_bf, 
                                                 y_data: y_valid, 
                                                 tf_keep_prob: 1.0})      

            print('%.2f epoch: train/val loss = %.4f/%.4f, train/val acc = %.4f/%.4f'%(
                (i+1)*batch_size/train_set_size, train_loss, valid_loss, train_acc, valid_acc))

    
    acc_train_DNN += train_acc
    acc_valid_DNN += valid_acc
    loss_train_DNN += train_loss
    loss_valid_DNN += valid_loss
    
    y_valid_pred_proba += y_pred.eval(feed_dict={x_data: x_valid_bf, tf_keep_prob: 1.0}) 
    y_test_pred_proba_DNN += y_pred.eval(feed_dict={x_data: x_test_bf, tf_keep_prob: 1.0}) 

    sess.close()
        
acc_train_DNN /= float(cv_num)
acc_valid_DNN /= float(cv_num)
loss_train_DNN /= float(cv_num)
loss_valid_DNN /= float(cv_num)

# final validation prediction
y_valid_pred_proba /= float(cv_num)
y_valid_pred_class = np.argmax(y_valid_pred_proba, axis = 1)

# final test prediction
y_test_pred_proba_DNN /= float(cv_num)
y_test_pred_class_DNN = np.argmax(y_test_pred_proba_DNN, axis = 1)

# final loss and accuracy
print('')
print('final: train/val loss = %.4f/%.4f, train/val acc = %.4f/%.4f'%(loss_train_DNN, 
                                                                      loss_valid_DNN, 
                                                                      acc_train_DNN, 
                                                                      acc_valid_DNN))

## show confusion matrix
    
cnf_matrix = confusion_matrix(one_hot_to_dense(y_valid), y_valid_pred_class)

abbreviation = ['BG', 'Ch', 'Cl', 'CC', 'CW', 'FH', 'LSB', 'M', 'SM', 'SP', 'SFC', 'SB']
pd.DataFrame({'class': species, 'abbreviation': abbreviation})

fig, ax = plt.subplots(1)
ax = sns.heatmap(cnf_matrix, ax=ax, cmap=plt.cm.Greens, annot=True)
ax.set_xticklabels(abbreviation)
ax.set_yticklabels(abbreviation)
plt.title('Confusion matrix of validation set')
plt.ylabel('True species')
plt.xlabel('Predicted species')
plt.show()

y_test_pred_class = y_test_pred_class_DNN
test_df['species_id'] = y_test_pred_class
test_df['species'] = [species[sp] for sp in y_test_pred_class]
test_df[['file', 'species']].to_csv('submission.csv', index=False)

print('total running time: ', datetime.datetime.now()-global_start)