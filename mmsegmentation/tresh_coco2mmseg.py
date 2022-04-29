from pycocotools.coco import COCO
import numpy as np
import os
import shutil
import mmcv
import os.path as osp
import numpy as np
from PIL import Image
import tqdm

root_dir = '../data/mmseg'
val_dir = '../data/val.json'
train_dir = '../data/train.json'


os.makedirs(root_dir,exist_ok=True)
os.makedirs(os.path.join(root_dir,'labels'),exist_ok=True)
os.makedirs(os.path.join(root_dir,'images'),exist_ok=True)
os.makedirs(os.path.join(root_dir,'splits'),exist_ok=True)

print('copping batch_01_vt/* ...')
for path in tqdm.tqdm(os.listdir('../data/batch_01_vt/')):
    shutil.copy('../data/batch_01_vt/'+path,os.path.join(root_dir,'images','batch_01_vt_'+path))

print('copping batch_02_vt/* ...')
for path in tqdm.tqdm(os.listdir('../data/batch_02_vt/')):
    shutil.copy('../data/batch_02_vt/'+path,os.path.join(root_dir,'images','batch_02_vt_'+path))

print('copping batch_03/* ...')
for path in tqdm.tqdm(os.listdir('../data/batch_03/')):
    shutil.copy('../data/batch_03/'+path,os.path.join(root_dir,'images','batch_03_'+path))

val_coco = COCO(val_dir)
train_coco = COCO(train_dir)

def make_txt(coco):
    for idx in tqdm.tqdm(coco.getImgIds()):        # cocodata의 모든 image idx list
        image_id = coco.getImgIds(imgIds=idx)
        image_infos = coco.loadImgs(image_id)[0]            # 이미지 file_name, height, width를 담고있는 dict

        ann_ids = coco.getAnnIds(imgIds=image_infos['id'])  # 이미지 idx가 일치하는 ann id list
        anns = coco.loadAnns(ann_ids)

        masks = np.zeros((image_infos["height"], image_infos["width"]))
        for i in range(len(anns)):
            masks[coco.annToMask(anns[i]) == 1] = anns[i]['category_id']
        masks = masks.astype(np.int8).astype(np.str)

        with open(os.path.join('../data/mmseg/labels',image_infos['file_name'].replace('/','_').replace('.jpg','.txt')), 'w') as f:
            for mask in masks.tolist():
                f.write(' '.join(mask) + '\n')

print('making txt ann of val ...')
make_txt(val_coco)
print('making txt ann of train ...')
make_txt(train_coco)

cat_ids = val_coco.getCatIds()
cats = val_coco.loadCats(cat_ids)
data_root = root_dir
img_dir = 'images'
ann_dir = 'labels'
# define class and plaette for better visualization
classes = []
for c in cats:
    classes.append(c['name'])
palette = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34], 
           [0, 11, 123], [118, 20, 12], [122, 81, 25], [241, 134, 51],[0, 241, 0],[0, 241, 0]]

print('making ann drawing ...')
for file in tqdm.tqdm(mmcv.scandir(osp.join(data_root, ann_dir), suffix='.txt'),total=3272):
    seg_map = np.loadtxt(osp.join(data_root, ann_dir, file)).astype(np.uint8)
    seg_img = Image.fromarray(seg_map).convert('P')
    seg_img.putpalette(np.array(palette, dtype=np.uint8))
    seg_img.save(osp.join(data_root, ann_dir, file.replace('.txt','.png')))


print('train split ...')
with open(root_dir + '/splits/train.txt','w') as f:
    for idx in tqdm.tqdm(train_coco.getImgIds()):
        image_id = train_coco.getImgIds(imgIds=idx)
        image_infos = train_coco.loadImgs(image_id)[0]
        f.write(image_infos['file_name'].replace('/','_').replace('.jpg','') + '\n')


print('val split ...')
with open(root_dir + '/splits/val.txt','w') as f:
    for idx in tqdm.tqdm(val_coco.getImgIds()):
        image_id = val_coco.getImgIds(imgIds=idx)
        image_infos = val_coco.loadImgs(image_id)[0]    
        f.write(image_infos['file_name'].replace('/','_').replace('.jpg','') + '\n')

print('Done')