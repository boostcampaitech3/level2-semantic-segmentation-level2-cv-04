
# merge configs
_base_ = [
    '../models/upernet_swin.py',
    '../datasets/coco-trash.py',
    '../custom_runtime.py',
    # '../schedules/custom_schedule.py'
	'../schedules/schedule_20k.py'
]

# model = dict(
#     pretrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth',
#     backbone=dict(
#         pretrain_img_size=384,
#         embed_dim=192,
#         depths=[2, 2, 18, 2],
#         num_heads=[6, 12, 24, 48],
#         window_size=12,
#         use_abs_pos_embed=False,
#         drop_path_rate=0.,
#         convert_weights=True,
#         patch_norm=True),
#     decode_head=dict(in_channels=[192, 384, 768, 1536], num_classes=11),
#     auxiliary_head=dict(in_channels=768, num_classes=11))
"""
https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation
"""
model = dict(
	# pretrained='https://github.com/SwinTransformer/storage/releases/download/v1.0.1/upernet_swin_base_patch4_window7_512x512.pth',
	backbone=dict(
		embed_dim=128,
		depths=[2, 2, 18, 2],
		num_heads=[4, 8, 16, 32],
		window_size=7,
		ape=False,
		drop_path_rate=0.3,
		patch_norm=True,
		use_checkpoint=False
	),
	decode_head=dict(
		in_channels=[128, 256, 512, 1024],
		num_classes=11
	),
	auxiliary_head=dict(
		in_channels=512,
		num_classes=11
	)
)