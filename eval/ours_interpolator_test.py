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
r"""A test script for mid frame interpolation from two input frames.

Usage example:
 python3 -m frame_interpolation.eval.interpolator_test \
   --frame1 <filepath of the first frame> \
   --frame2 <filepath of the second frame> \
   --model_path <The filepath of the TF2 saved model to use>

The output is saved to <the directory of the input frames>/output_frame.png. If
`--output_frame` filepath is provided, it will be used instead.
"""
import os
from typing import Sequence
import glob
import pandas as pd
from . import interpolator as interpolator_lib
from . import util
from absl import app
from absl import flags
import numpy as np

_DATA_ROOT = flags.DEFINE_string(
    name='data_root',
    default=None,
    help='The filepath of the text input.',
    required=True)
# _FRAME1 = flags.DEFINE_string(
#     name='frame1',
#     default=None,
#     help='The filepath of the first input frame.',
#     required=True)
# _FRAME2 = flags.DEFINE_string(
#     name='frame2',
#     default=None,
#     help='The filepath of the second input frame.',
#     required=True)
_MODEL_PATH = flags.DEFINE_string(
    name='model_path',
    default=None,
    help='The path of the TF2 saved model to use.')
# _OUTPUT_FRAME = flags.DEFINE_string(
#     name='output_frame',
#     default=None,
#     help='The output filepath of the interpolated mid-frame.')
_ALIGN = flags.DEFINE_integer(
    name='align',
    default=64,
    help='If >1, pad the input size so it is evenly divisible by this value.')
_BLOCK_HEIGHT = flags.DEFINE_integer(
    name='block_height',
    default=1,
    help='An int >= 1, number of patches along height, '
    'patch_height = height//block_height, should be evenly divisible.')
_BLOCK_WIDTH = flags.DEFINE_integer(
    name='block_width',
    default=1,
    help='An int >= 1, number of patches along width, '
    'patch_width = width//block_width, should be evenly divisible.')
_Gen_Broken = flags.DEFINE_integer(
    name='genbroken',
    default=0,
    help='0: No Imahes Broken, 1:images Broken.')

## Set tf ENV. 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="1"

        
def _run_interpolator() -> None:
  """Writes interpolated mid frame from a given two input frame filepaths."""

  interpolator = interpolator_lib.Interpolator(
      model_path=_MODEL_PATH.value,
      align=_ALIGN.value,
      block_shape=[_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value])
    
#   mid_frame_filepath = _OUTPUT_FRAME.value
#   if not os.path.exists(mid_frame_filepath):
#       os.makedirs(mid_frame_filepath)
  GenBroken = _Gen_Broken.value
  pth = _DATA_ROOT.value
  if GenBroken == 1:
     with open(pth, 'r') as txt:
           sequence_list = [line.strip() for line in txt]
     for seq in sequence_list:
         pth_image_1, pth_image_2 = seq.split(' ')
         ### Create name img path
         folder_name = pth_image_1.split('/')[:-1]
         folder_name_ = '/'.join(folder_name)
         mid_frame_filepath = folder_name_.replace("rheology2023", "Frame_Inter_rheology2023/broken-images/Frame_Inter/FILM_Model")
         import imageio
         os.makedirs(mid_frame_filepath, exist_ok=True) 
        
          # First batched image.
         image_1 = util.read_image(pth_image_1)
         image_batch_1 = np.expand_dims(image_1, axis=0)

          # Second batched image.
         image_2 = util.read_image(pth_image_2)
         image_batch_2 = np.expand_dims(image_2, axis=0)

         # Batched time.
         batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)

         # Invoke the model for one mid-frame interpolation.
         mid_frame = interpolator(image_batch_1, image_batch_2, batch_dt)[0]

           # Write interpolated mid-frame.
         mid_frame_save = os.path.join(mid_frame_filepath, os.path.basename(pth_image_1).split('.')[0]+'_inter'+'.png')
         util.write_image(mid_frame_save, mid_frame)
         print('result saved!')
     SHOW_pathimg = mid_frame_filepath.split('/')[:7]
     SHOW_pathimg_ = '/'.join(SHOW_pathimg)
     print('Frame Interpolation SaVe at -->>', SHOW_pathimg_)
     print('*'*130)
  else:
      pathframe = pd.read_csv(pth)
      data_path = pathframe['FolderPathDemo'].tolist()      
      for test_demo in data_path:
            print(f'On Process Folder  -->> [ {test_demo} ]')
            ##** Set SAve path  ##-- prepare path to save images---**  
            folder_name_ = test_demo.replace("demo", "inter")
            folder_name_ = folder_name_.split('.')[0]
            mid_frame_filepath = folder_name_.replace("pred_text", "Frame_Inter/FILM_Model") 
            import imageio
            os.makedirs(mid_frame_filepath, exist_ok=True)

      # with open(os.path.join(args.data_root, 'test-demo.txt'), 'r') as txt:
            with open(test_demo, 'r') as txt:
                  sequence_list = [line.strip() for line in txt]
            for seq in sequence_list:
                pth_image_1, pth_image_2 = seq.split(' ')
              # First batched image.
              #image_1 = util.read_image(_FRAME1.value)
                image_1 = util.read_image(pth_image_1)
                image_batch_1 = np.expand_dims(image_1, axis=0)

              # Second batched image.
              #image_2 = util.read_image(_FRAME2.value)
                image_2 = util.read_image(pth_image_2)
                image_batch_2 = np.expand_dims(image_2, axis=0)

              # Batched time.
                batch_dt = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)

              # Invoke the model for one mid-frame interpolation.
                mid_frame = interpolator(image_batch_1, image_batch_2, batch_dt)[0]

              # Write interpolated mid-frame.
                mid_frame_save = os.path.join(mid_frame_filepath, os.path.basename(pth_image_1).split('.')[0]+'_inter'+'.png')
        #       if not mid_frame_filepath:
        #         mid_frame_filepath = f'{os.path.dirname(_FRAME1.value)}/output_frame.png'
              #util.write_image(mid_frame_filepath, mid_frame)
                util.write_image(mid_frame_save, mid_frame)
                print('result saved!')
            print('Frame Interpolation SaVe at -->>', mid_frame_filepath)
            print('*'*140)

def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  _run_interpolator()


if __name__ == '__main__':
  app.run(main)
