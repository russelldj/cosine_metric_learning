# vim: expandtab:ts=4:sw=4
import functools
import os
import numpy as np
import scipy.io as sio
import cosine_inference

import cv2
import glob
import random
import scipy
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import argparse

def plot_confusion_matrix(cm, classes, images=None, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(20, 34), dpi=150)
    plt.subplot(1,2,1,autoscale_on=True)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Cosine distances between descriptors')
    else:
        print('Cosine distances between descriptors')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")


    plt.ylabel('IDs')
    plt.xlabel('IDs')

    if images is not None:
        plt.subplot(1, 2, 2, autoscale_on=True) 
        # opencv deals with BGR and this appears to be RGB
        plt.imshow(np.flip(images, axis=2), cmap)
        plt.title("You can get the index by i * row length + j")
    plt.tight_layout()


parser = argparse.ArgumentParser()
parser.add_argument('--image_ids', nargs="+", type=int)
parser.add_argument('--num_per_id', type=int, default=3)
args = parser.parse_args()

IMAGE_IDS = args.image_ids
BASE_DIR = "/home/drussel1/data/ADL/ADL_Market_format/all_bounding_boxes"

args.num_per_id = args.num_per_id

cosine_extractor = cosine_inference.CosineInference()

features = []
image_collections = []
for image_id in IMAGE_IDS:
    files = glob.glob('{}/{}*'.format(BASE_DIR, image_id))
    random.shuffle(files)
    images = []
    for fn in files[:args.num_per_id]:
        img = cv2.imread(fn)
        images.append(img)
    while len(images) < args.num_per_id:
        images.append(np.zeros(images[0].shape, dtype=np.int8))
    features.append(cosine_extractor.get_features(files[:args.num_per_id]))
    image_collections.append(np.concatenate(images, axis=1))

features = np.concatenate(features, axis=0)
print(features.shape)

image_collections = np.concatenate(image_collections, axis=0)

output = np.zeros((features.shape[0], features.shape[0]))
for i in range(features.shape[0]):
    for j in range(features.shape[0]):
        output[i, j]=scipy.spatial.distance.cosine(features[i,:], features[j,:])

print(output)

plot_confusion_matrix(output, classes=[str(i) for i in range(output.shape[0])],
                              title='Cosine distance between descriptors', images=image_collections)
print("/home/drussel1/temp/cosine_descriptor_plots/{}.png".format(str(args.image_ids).replace(', ', '_').replace('[', '').replace(']', '')))
plt.show()
#plt.savefig("/home/drussel1/temp/cosine_descriptor_plots/{}.png".format(str(args.image_ids).replace(', ', '_').replace('[', '').replace(']', '')))

