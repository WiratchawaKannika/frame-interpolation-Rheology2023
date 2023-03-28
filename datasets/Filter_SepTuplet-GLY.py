import os
import time
import numpy as np
from PIL import Image
import random
import glob
import json 
import pandas as pd
from pathlib import Path
import PIL
import cv2 
import tqdm
from multiprocessing import Pool
import argparse

# set number of CPUs to run on
ncore = "8"
# set env variables
# have to set these before importing numpy
os.environ["OMP_NUM_THREADS"] = ncore
os.environ["OPENBLAS_NUM_THREADS"] = ncore
os.environ["MKL_NUM_THREADS"] = ncore
os.environ["VECLIB_MAXIMUM_THREADS"] = ncore
os.environ["NUMEXPR_NUM_THREADS"] = ncore


my_parser = argparse.ArgumentParser()
my_parser.add_argument('--subset', type=str, help='train or test')
my_parser.add_argument('--data_root', type=str, default='/media/tohn/SSD/rheology_data/Frame_Inter_rheology2023/_3FrameFilter/', help='path to text files')
my_parser.add_argument('--root2save', type=str, default='/media/tohn/SSD/rheology_data/Frame_Inter_rheology2023/_3FrameFilter/', help='path to save new file text')
args = my_parser.parse_args()

subset = args.subset
data_root = args.data_root
print(f'Data Set : [ {data_root}]')
print(f'-'*100)
root2save = args.root2save
if not os.path.exists(root2save):
    os.makedirs(root2save)

    
##function check broken images path.
def img_verify(sub_testlist):
    _except = []
    for file in sub_testlist:
        try:
            img = Image.open(file)  # open the image file
            img.verify()  # verify that it is, in fact an image
        except (IOError, SyntaxError) as e:
            _except.append(file)
    return _except 


def Filter_dataset(trainlist0, subset, root2save): ## subset == train or test -->> type must be string
    pth_training1, pth_training2, pth_training3 = [],[],[]

    for index in range(len(trainlist0)):
        print(f"Load images : {index+1}")
        imgpaths = trainlist0[index]
        from PIL import Image, ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        # Load images
        _except = img_verify(imgpaths)
        if len(_except) == 0:
            pth_training1.append(imgpaths[0])
            pth_training2.append(imgpaths[1])
            pth_training3.append(imgpaths[2])
        #time.sleep(0.02)
    time.sleep(0.09)
    print("========== Create DataFrame ========== ")
    ## Prepare to create text files
    df_ = pd.DataFrame(
        {'Path1': pth_training1,
         'Path2': pth_training2,
         'Path3': pth_training3
        })
    print("========== Prepare to create text files ========== ")
    time.sleep(0.5)
    df_['Path_txt'] = ''
    for i in range(len(df_)):
        name1 = df_['Path1'][i]
        name2 = df_['Path2'][i]
        name3 = df_['Path3'][i]
        df_.loc[df_.index[i], 'Path_txt'] = str(name1)+' '+str(name2)+' '+str(name3)
    print(f'Filtered {subset} set with shape : {df_.shape}')
    # Save to text file
    time.sleep(0.5)
    print("On process to Save CSV. file")
#     list_path = df_['Path_txt'].tolist()
#     path2savetxt = f'{root2save}GLYrheology2023-{subset}.txt'
    path2savecsv = f'{root2save}GLYrheology2023-pathimg-{subset}.csv'
#     with open(path2savetxt, 'w') as f:
#          for line in list_path:
#                 f.write(f"{line}\n")
#     print(f'Done!! : Write text file name -> [ {path2savetxt} ] ')
    df_.to_csv(path2savecsv)
    print(f'Done!! : Write CSV. file name -> [ {path2savecsv} ] ')
    

### Start
trainlist0 = []
dfGlycerol = pd.read_csv(f'/media/tohn/SSD/rheology_data/Frame_Inter_rheology2023/_3FrameFilter/Glycerol2023_text-{subset}.csv')
print(f"{subset} set ==> {dfGlycerol.shape[0]} ")
data_fn = dfGlycerol['Path2text'].tolist()   ## Glycerol_Path
data_fn.sort()

## For training set
for fn in data_fn:
    with open(fn, 'r') as txt:
         meta_data = [line.strip() for line in txt]
    for seq in meta_data:
        img1_path, img2_path, img3_path = seq.split(' ')
        trainlist0.append([img1_path, img2_path, img3_path])
        #time.sleep(0.01)
print(f"**{subset} set** with Data size: {len(trainlist0)} batch")
time.sleep(0.5)     

#from multiprocessing import Process
from multiprocessing import Pool

# if __name__ == '__main__':
#     starttime = time.time()
#     Process(target=Filter_dataset, args=(datalist, args.dataset, root2save, starttime)).start()
    
inputs = [(trainlist0, subset, root2save)]

start = time.time()
with Pool(8) as p:
    results = p.starmap(Filter_dataset, inputs)
end = time.time()
#print(results)
# for r in results:
#     print(r)
print(f'Filter {subset} set, That took {round(end-start, 3)} seconds')
         
