import json 
import pandas as pd
import numpy as np
from pathlib import Path
import PIL
from PIL import Image
import cv2 
import tqdm
import os
import shutil as sh
import glob
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

def main():
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--set', type=str, help='[train , test]')
    args = my_parser.parse_args()

    subset = args.set
    
    ## ##Import data 
    dfGlycerol = pd.read_csv(f'/home/kannika/code/rheology2023_Glycerol_{subset}.csv')
    print(f"{subset} set ==> {dfGlycerol.shape[0]} ")
    z_ = dfGlycerol['Glycerol_Path'].tolist() 
    # Crete text files for prepare created sub directory
    lstName2Save = list()
    ##**-- Create text file. 
    for k in range(len(z_)) :
        pth = z_[k]
        txt_name = pth.split('/')[-1]
        f_name_toSave = pth.replace('rheology2023', "Frame_Inter_rheology2023/_3FrameFilter")
        f_name_toSave = f_name_toSave.split('/')[:-1]
        _f_name_toSave = '/'.join(f_name_toSave)
        txtname2save = _f_name_toSave+'/'+txt_name+'-'+subset+'3line.txt'
        ## Create Directory
        import imageio
        os.makedirs(_f_name_toSave, exist_ok=True)

        ###**-- get images list --**
        files = glob.glob(f"{pth}/*")
        files.sort()
        ## Create dataframe for text.
        df = pd.DataFrame(files, columns =['Path'])
        df_ = df[:-2].reset_index(drop=True)
        #print(df_.shape)
        # df_
        df2 = pd.DataFrame(files, columns =['Path'])
        df2_ = df2[1:-1].reset_index(drop=True)
        #print(df2_.shape)
        #df2_ 
        df3 = pd.DataFrame(files, columns =['Path'])
        df3_ = df3[2:].reset_index(drop=True)
        #print(df3_.shape)
        #df3_ 
        df_['Path_txt'] = ''
        for i in range(len(df_)):
            name1 = df_['Path'][i]
            name2 = df2_['Path'][i]
            name3 = df3_['Path'][i]
            df_.loc[df_.index[i], 'Path_txt'] = str(name1)+' '+str(name2)+' '+str(name3)  
        print(df_.shape)
        list_path = df_['Path_txt'].tolist()
        with open(txtname2save, 'w') as f:
                for line in list_path:
                     f.write(f"{line}\n")
        lstName2Save.append(txtname2save)
        print(f'On Process : Write text file name ==> [ {txtname2save} ] ')
    ## Save path text to dataframe.    
    df_train_text = pd.DataFrame(lstName2Save, columns =['Path2text'])
    df_train_text2csv = f"/media/tohn/SSD/rheology_data/Frame_Inter_rheology2023/_3FrameFilter/Glycerol2023_text-{subset}.csv"
    df_train_text.to_csv(df_train_text2csv)
    print(f"Save Dataframe for text path as ===> [ {df_train_text2csv} ]") 
    
    
## Run main Function 
if __name__ == '__main__':
    main()
    
    
    
