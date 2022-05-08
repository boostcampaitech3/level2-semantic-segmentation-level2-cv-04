# -- import
import os
import argparse
import yaml
from mmcv import Config
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor, set_random_seed
from mmseg.datasets import build_dataset    
from mmcv.runner import load_checkpoint

def main(config_train):
	# -- config file
	config_dir = config_train['config_dir']
	config_file = config_train['config_file']
	cfg = Config.fromfile(f'./configs/{config_dir}/{config_file}.py')

	# -- log config
	log_interval = config_train['log_interval']
	if config_train['wandb']:
		cfg.log_config = dict(interval = log_interval,
								hooks=[dict(type = 'TextLoggerHook'),
										dict(type = 'WandbLoggerHook',
											init_kwargs = dict(project = config_train['wandb_proj'], name = config_train['wandb_name'], entity='cv04'))
									])
	else:
		cfg.log_config = dict(interval = log_interval,
								hooks=[dict(type = 'TextLoggerHook')])

	# -- seed
	cfg.seed = config_train['seed']
	set_random_seed(cfg.seed)

	# -- gpu ids
	cfg.gpu_ids = [0]

	# -- hyperparameter
	cfg.data.samples_per_gpu = config_train['batch_size']
	cfg.optimizer['lr'] = config_train['lr']

	# -- work directory(for save *.pth, *.log)
	cfg.work_dir = os.path.join('./work_dirs', config_file)

	# -- save best model
	if config_train['save_best_model']:
		cfg.evaluation['save_best'] = 'mIoU'
		cfg.evaluation['interval'] = config_train['eval_interval']
<<<<<<< HEAD
		cfg.checkpoint_config['interval'] = config_train['eval_interval']
=======
		cfg.evaluation['img_log_interval'] = config_train['img_log_interval']
>>>>>>> a18b332... [fix] wandb image log for swin transformer

	# -- dataset
	if config_train['data_dir_root']:
		cfg.data['train']['img_dir']=os.path.join(config_train['data_dir_root'],'images/training')
		cfg.data['train']['ann_dir']=os.path.join(config_train['data_dir_root'],'annotations/training')
		cfg.data['val']['img_dir']=os.path.join(config_train['data_dir_root'],'images/validation')
		cfg.data['val']['ann_dir']=os.path.join(config_train['data_dir_root'],'annotations/validation')
	datasets = [build_dataset(cfg.data.train)]

	# -- build model
	model = build_segmentor(cfg.model)

	# -- init weight
	if config_train['use_ckpt']:
		ckpt_name = config_train['ckpt_name']
		checkpoint_path = os.path.join(cfg.work_dir, f'{ckpt_name}.pth')
		load_checkpoint(model, checkpoint_path, map_location = 'cpu')
	else:
		model.init_weights()

	# -- running
	train_segmentor(model, datasets, cfg, distributed=False, validate = config_train['validate'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--config_train', type = str, help = 'path of train configuration yaml file')

    args = parser.parse_args()

    # load yaml
    with open(args.config_train) as f:
        config_train = yaml.load(f, Loader = yaml.FullLoader)

    # running
    main(config_train)
    