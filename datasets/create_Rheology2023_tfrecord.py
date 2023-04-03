# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Beam pipeline that generates Vimeo-90K (train or test) triplet TFRecords.
Vimeo-90K dataset is built upon 5,846 videos downloaded from vimeo.com. The list
of the original video links are available here:
https://github.com/anchen1011/toflow/blob/master/data/original_vimeo_links.txt.
Each video is further cropped into a fixed spatial size of (448 x 256) to create
89,000 video clips.
The Vimeo-90K dataset is designed for four video processing tasks. This script
creates the TFRecords of frame triplets for frame interpolation task.
Temporal frame interpolation triplet dataset:
  - 73,171 triplets of size (448x256) extracted from 15K subsets of Vimeo-90K.
  - The triplets are pre-split into (train,test) = (51313,3782)
  - Download links:
    Test-set: http://data.csail.mit.edu/tofu/testset/vimeo_interp_test.zip
    Train+test-set: http://data.csail.mit.edu/tofu/dataset/vimeo_triplet.zip
For more information, see the arXiv paper, project page or the GitHub link.
@article{xue17toflow,
  author = {Xue, Tianfan and
            Chen, Baian and
            Wu, Jiajun and
            Wei, Donglai and
            Freeman, William T},
  title = {Video Enhancement with Task-Oriented Flow},
  journal = {arXiv},
  year = {2017}
}
Project: http://toflow.csail.mit.edu/
GitHub: https://github.com/anchen1011/toflow
Inputs to the script are (1) the directory to the downloaded and unzipped folder
(2) the filepath of the text-file that lists the subfolders of the triplets.
Output TFRecord is a tf.train.Example proto of each image triplet.
The feature_map takes the form:
  feature_map {
      'frame_0/encoded':
          tf.io.FixedLenFeature((), tf.string, default_value=''),
      'frame_0/format':
          tf.io.FixedLenFeature((), tf.string, default_value='jpg'),
      'frame_0/height':
          tf.io.FixedLenFeature((), tf.int64, default_value=0),
      'frame_0/width':
          tf.io.FixedLenFeature((), tf.int64, default_value=0),
      'frame_1/encoded':
          tf.io.FixedLenFeature((), tf.string, default_value=''),
      'frame_1/format':
          tf.io.FixedLenFeature((), tf.string, default_value='jpg'),
      'frame_1/height':
          tf.io.FixedLenFeature((), tf.int64, default_value=0),
      'frame_1/width':
          tf.io.FixedLenFeature((), tf.int64, default_value=0),
      'frame_2/encoded':
          tf.io.FixedLenFeature((), tf.string, default_value=''),
      'frame_2/format':
          tf.io.FixedLenFeature((), tf.string, default_value='jpg'),
      'frame_2/height':
          tf.io.FixedLenFeature((), tf.int64, default_value=0),
      'frame_2/width':
          tf.io.FixedLenFeature((), tf.int64, default_value=0)
      'path':
          tf.io.FixedLenFeature((), tf.string, default_value='')
  }
Usage example:
  python3 -m frame_interpolation.datasets.create_vimeo90K_tfrecord \
    --input_dir=<root folder of vimeo90K dataset> \
    --input_triplet_list_filepath=<filepath of tri_{test|train}list.txt> \
    --output_tfrecord_filepath=<output tfrecord filepath>
"""
import os
from util import _resample_image, generate_image_triplet_example, ExampleGenerator
from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import numpy as np
import tensorflow as tf
import pandas as pd
from multiprocessing import Pool

# set number of CPUs to run on
ncore = "8"
# set env variables
# have to set these before importing numpy
os.environ["OMP_NUM_THREADS"] = ncore
os.environ["OPENBLAS_NUM_THREADS"] = ncore
os.environ["MKL_NUM_THREADS"] = ncore
os.environ["VECLIB_MAXIMUM_THREADS"] = ncore
os.environ["NUMEXPR_NUM_THREADS"] = ncore



_INTPUT_TRIPLET_LIST_FILEPATH = flags.DEFINE_string(
    'input_triplet_list_filepath',
    default='/media/tohn/SSD/rheology_data/Frame_Inter_rheology2023/_3FrameFilter/GLYrheology2023-pathimg-train.csv',
    help='/path/to/csv/dataset/each columns containing path to Images')

_OUTPUT_TFRECORD_FILEPATH = flags.DEFINE_string(
    'output_tfrecord_filepath',
    default='/media/tohn/SSD/rheology_data/Frame_Inter_rheology2023/_3FrameFilter/TFRecord-FILM/train/train',
    help='Filepath to the output TFRecord file.')

_NUM_SHARDS = flags.DEFINE_integer('num_shards',
    default=200, # set to 3 for vimeo_test, and 200 for vimeo_train.
    help='Number of shards used for the output.')


## Set tf ENV. 
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
# os.environ["CUDA_VISIBLE_DEVICES"]="0"


_INTERPOLATOR_IMAGES_MAP = ['frame_0', 'frame_1', 'frame_2']


def main(unused_argv):
    """Creates and runs a Beam pipeline to write frame triplets as a TFRecord."""

    triplets_list = pd.read_csv(_INTPUT_TRIPLET_LIST_FILEPATH.value) 
    #triplets_list = triplets_list.iloc[:1000]
    print(f"Data Set : {triplets_list.shape[0]} batch")

    triplet_dicts = []
    for index in range(len(triplets_list)):
        image_basename = []
        im1 = triplets_list['Path1'][index]
        im2 = triplets_list['Path2'][index]
        im3 = triplets_list['Path3'][index]
        image_basename = [im1, im2, im3]
        triplet_dict = {
            image_key: image_basename
            for image_key, image_basename in zip(_INTERPOLATOR_IMAGES_MAP, image_basename)
        }
        triplet_dicts.append(triplet_dict)
    p = beam.Pipeline('DirectRunner')
    (p | 'ReadInputTripletDicts' >> beam.Create(triplet_dicts)  # pylint: disable=expression-not-assigned
    | 'GenerateSingleExample' >> beam.ParDo(
       ExampleGenerator(_INTERPOLATOR_IMAGES_MAP))
    | 'WriteToTFRecord' >> beam.io.tfrecordio.WriteToTFRecord(
       file_path_prefix=_OUTPUT_TFRECORD_FILEPATH.value,
       num_shards=_NUM_SHARDS.value,
       coder=beam.coders.BytesCoder()))
    result = p.run()
    result.wait_until_finish()

    logging.info('Succeeded in creating the output TFRecord file: \'%s@%s\'.',
    _OUTPUT_TFRECORD_FILEPATH.value, str(_NUM_SHARDS.value))

    
if __name__ == '__main__':
    app.run(main)
