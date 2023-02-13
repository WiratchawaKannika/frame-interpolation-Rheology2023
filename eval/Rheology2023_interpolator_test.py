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
_MODEL_PATH = flags.DEFINE_string(
    name='model_path',
    default='/media/SSD/DATA_FILM/pretrained_models/pretrained_models/film_net/Style/saved_model',
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
_GenNum = flags.DEFINE_integer(
    name='genNum',
    default=1,
    help='Number of gen frame using FLIM Model.')

## Set tf ENV. 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

        
def _run_interpolator() -> None:
  """Writes interpolated mid frame from a given two input frame filepaths."""

  interpolator = interpolator_lib.Interpolator(
      model_path=_MODEL_PATH.value,
      align=_ALIGN.value,
      block_shape=[_BLOCK_HEIGHT.value, _BLOCK_WIDTH.value])
  genNum = _GenNum.value
  _genNum = f'gen{genNum}'
  genNum_old = genNum-1
  _genNumold = f'gen{genNum_old}'
  data_root = _DATA_ROOT.value
  test_demo = glob.glob(f"{data_root}/*-2linedemo.txt")
  test_demo.sort()
  ## Read Text Files. 
  for file in test_demo: 
      ##** Set SAve path 
      print(f'On Process Folder  -->> [ {file} ]')
      list_imgframe = list()
      if genNum == 1:
            folder_name_ = file.replace("-2linedemo", f"_{_genNum}-inter")
            folder_name_ = folder_name_.split('.')[0]
            save_pathimg = folder_name_.replace("ues2Frame/pred_text/Saliva2/origin", f"FILM_Model/Frame_Inter/Saliva2/{_genNum}")
      else:
            folder_name_ = file.replace(f"{_genNumold}-2linedemo", f"{_genNum}-inter")
            folder_name_ = folder_name_.split('.')[0]
            save_pathimg = folder_name_.replace(_genNumold, _genNum)
     
      ##**Mkdir Directory 
      import imageio
      os.makedirs(save_pathimg, exist_ok=True)
        
      ### Create name img path
      name_img = save_pathimg.split("/")[-1]
      name_img_ = name_img.split("_")[:-1]
      __name_img = '_'.join(name_img_)

      ## Create path to save CSV.
      save_csv = save_pathimg.split("/")[:-1]
      save_csv_ = '/'.join(save_csv)
      pathName_csv = save_csv_+'/'+__name_img+'_'+_genNum+'.csv'
      _pathName_csv = pathName_csv.replace(_genNumold, f"FILM_Model/Saliva2/{_genNum}")
    
      ##read text files dataset
      with open(file, 'r') as txt:
           sequence_list = [line.strip() for line in txt]
      for i in range(len(sequence_list)): 
          pth_image_1, pth_image_2 = sequence_list[i].split(' ')

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
          mid_frame_save = os.path.join(save_pathimg, __name_img+'_inter'+str(i+1)+'_'+_genNum+'.jpg')
          util.write_image(mid_frame_save, mid_frame)
          print('result saved!')
          ## append images to list. 
          if i == len(sequence_list)-1 :
             list_imgframe.append(pth_image_1)
             list_imgframe.append(mid_frame_save)
             list_imgframe.append(pth_image_2)
          else:
               list_imgframe.append(pth_image_1)
               list_imgframe.append(mid_frame_save)

      ##save to CSV.       
      df = pd.DataFrame(list_imgframe, columns =['seq_inter'])
      df.to_csv(_pathName_csv)
      print('Frame Interpolation saVe at -->>', save_pathimg)
      print(f"Save Sequence Dataframe at -->> {_pathName_csv} With Shape: {df.shape}")
      print('*'*130)


        
def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  _run_interpolator()


if __name__ == '__main__':
  app.run(main)
