# ################################################################################ #
# Please note that for copyright reasons, we are not allowed to publish our code!  #
# This is the only file that does not fall under this copyright regulation.        #
# ################################################################################ #


# Please modify the following variables before running:
input_folder = '/userdata/Argla/project1/' # The input location of the training-??-??, validation-??-?? and test-??-?? files
output_folder = '/userdata/cglanzer/with_skeletonRGB/' # The output location of the TFRecord data including the skeleton RGB

import tensorflow as tf
import numpy as np
from Skeleton import Skeleton
import dataset as ds
from dataset_numpy_to_tfrecord import *
tf.enable_eager_execution()

print("Generating files...")

patterns = ["training","validation","test"]
shards = [50,15,15]
for K in range(3):
    pattern = patterns[K]
    tfrecord_pattern = input_folder+pattern+"-??-of-??"
    dataset = ds.TFRecordDataset(data_path=tfrecord_pattern,
                            batch_size=1,
                            shuffle=False)

    output_dir = output_folder+pattern  # Output directory.
    n_shards = shards[K]
    tfrecord_writers = create_tfrecord_writers(output_dir, n_shards)
    data_iterator = dataset.get_iterator()
    counter = 0
    names = ["rgb","depth","skeleton","segmentation","length","label","id"]
    for batch in data_iterator:
        for name in names:
            batch[name] = batch[name].numpy()
        batch["skeletonRGB"] = np.empty_like(batch["rgb"])
        for i in range(batch["skeleton"].shape[1]):
            skeleton = Skeleton(batch["skeleton"][0][i])
            skeleton.resizePixelCoordinates()
            skeleton_img = skeleton.toImage(80,80)
            batch["skeletonRGB"][0][i] = skeleton_img
    
        tfexample = to_tfexample(batch)
        write_tfexample(tfrecord_writers, tfexample)
        counter += 1
        if counter%10 == 0:
            print("Current pattern: " + pattern + " / Current ID: " + str(counter))
    close_tfrecord_writers(tfrecord_writers)
        
