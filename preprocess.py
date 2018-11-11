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

## read train and test data

# directories
cw_dir = os.getcwd()
data_dir = '/Users/joe/Desktop/BU_2018_fall/EC601/plant_seedling_classification/'
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

train_df = pd.concat([train_df[train_df['species'] == sp][:200] for sp in species])
train_df.index = np.arange(len(train_df))

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
x_train_valid = np.zeros((len(train_df), target_image_size, target_image_size, 3),
                         dtype='float32')
y_train_valid = train_df.loc[:, 'species_id'].values 
for i, filepath in tqdm(enumerate(train_df['filepath'])):
    
    # read original images
    #img = read_image(filepath, (target_image_size, target_image_size))
    
    # read segmented image
    _,_,_,img = read_segmented_image(filepath, (299, 299))
    
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
    

print('x_train_valid.shape = ', x_train_valid.shape)
print('x_test.shape = ', x_test.shape)