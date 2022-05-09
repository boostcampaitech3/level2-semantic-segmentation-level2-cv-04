# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook"),
        dict(type="WandbLoggerHook", 
        init_kwargs=dict(
            project="seg_model_test", 
            entity = 'p0tpourri',
            name="mmseg_test_uperswin")),
    ],
)
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
