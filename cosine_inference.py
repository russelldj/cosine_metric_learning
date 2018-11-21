import functools
import os
import numpy as np
import scipy.io as sio
from .datasets import ADL #MOD was ADL
from .datasets import util
import tensorflow as tf
import time
import numpy as np

import deep_sort.cosine_metric_learning.nets.deep_sort.network_definition as net
from . import train_app

class CosineInference(object):
    def __init__(self):
        arg_parser = train_app.create_default_argument_parser("Market1501")
        arg_parser.add_argument(
            "--dataset_dir", help="Path to MARS dataset directory.",
            default="resources/MARS-evaluation-master")
        self.network_factory = net.create_network_factory(
            is_training=False, num_classes=ADL.MAX_LABEL + 1,
            add_logits=True, reuse=tf.AUTO_REUSE)
        self.BATCH_SIZE = 1
        self.weights_path = "/home/drussel1/data/readonly/models/cosine_metric_extractor_ckpt/model.ckpt-208964"
        self.session = tf.Session()
        self.encoder = train_app.return_encoder(
            net.preprocess, self.network_factory, self.weights_path,
            np.random.randint(0, 255, (1,128, 128,3)), image_shape=ADL.IMAGE_SHAPE, batch_size=self.BATCH_SIZE, session=self.session)

    def get_features(self, filenames):# this will eventually need to be the actual images
        print("new style self.encoder")
        return self.encoder(filenames)

def test_image_reading(filenames):
    import tensorflow as tf
    cosine_inferer=CosineInference()
    with tf.Session() as sess:
        print(cosine_inferer.get_features(filenames, session=sess))


if __name__=="__main__":
    print('testing cosine inference')
    cosine_inference=CosineInference()
    print("generating a random \"image\"")
    images=np.random.random_integers(0, 255, (32,128,128,3))
    #images = ["/home/drussel1/data/ADL/ADL_Market_format/ADL_P_01/0235_c1s1_28752_00.jpg", "/home/drussel1/data/ADL/ADL_Market_format/ADL_P_01/0235_c1s1_28752_00.jpg"]
    start_time = time.time()
    for i in range(100):
        feats = cosine_inference.get_features(images)
        print("the features shape is {} and their norm is {}".format(feats.shape, np.linalg.norm(feats)))
        print(feats)
    print(start_time - time.time())
