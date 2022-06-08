import pandas as pd
import numpy as np
import glob
from tqdm import tqdm

data_root = 'ensemble'

df_list = []
for path in glob.glob(data_root+'/*.csv'):
    df = pd.read_csv(path)
    df_list.append(df)

scores_list = []
for path in glob.glob(data_root+'/*.txt'):
    with open(path,'r') as f:
        scores = f.readline()
        scores = list(map(float,scores.split(',')))
    scores_list.append(scores)

def get_encoding(img,score):
    img_list = img.split(' ')
    img_np = np.array(img_list,dtype=int)
    encoding = np.eye(11)[img_np]
    encoding *= score
    return encoding


out_dict = {'image_id':[],'PredictionString':[]}
for i in tqdm(range(len(df_list[0]))):
    out_dict['image_id'].append(df_list[0]['image_id'][i])
    
    encoding = np.zeros((65536,11))
    for df,scores in zip(df_list,scores_list):
        encoding += get_encoding(df['PredictionString'][i],scores)

    out_img_np = np.argmax(encoding,axis=1)
    out_img_list = out_img_np.tolist()
    out_img = ' '.join(map(str,out_img_list))
    out_dict['PredictionString'].append(out_img)


out_df = pd.DataFrame(out_dict)
out_df.to_csv('ensemble.csv',index=False)