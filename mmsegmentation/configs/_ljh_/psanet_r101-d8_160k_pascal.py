_base_ = './psanet_r50-d8_160k_pascal.py'
model = dict(pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101))
