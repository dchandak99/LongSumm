import os
import argparse

import torch

from train_extractive import train_ext

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-task', default='ext', type=str, choices=['ext', 'abs'])
	parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])
	#parser.add_argument('-use_gpu', default='y', type=str, choices=['y', 'n'])
	parser.add_argument('-seed', default=2020, type=int)
	parser.add_argument("-large", type=str2bool, nargs='?',const=True,default=False)
	parser.add_argument("-temp_dir", default='temp/')
	parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True)
	parser.add_argument("-train_from", default='')
	parser.add_argument("-bert_data_path", default='bert_data/cnndm')
	parser.add_argument("-model_path", default='models/')

	parser.add_argument("-batch_size", default=140, type=int)
	parser.add_argument("-test_batch_size", default=200, type=int)

	parser.add_argument("-encoder", default='bert', type=str, choices=['bert', 'baseline'])
	parser.add_argument("-max_pos", default=512, type=int)
		
	parser.add_argument("-max_tgt_len", default=140, type=int)
	parser.add_argument("-use_interval", type=str2bool, nargs='?',const=True,default=True)

	#Extractive summerizer args
	parser.add_argument("-ext_ff_size", default=2048, type=int)
	parser.add_argument("-ext_heads", default=8, type=int)
	parser.add_argument("-ext_dropout", default=0.2, type=float)
	parser.add_argument("-ext_layers", default=2, type=int)
	parser.add_argument("-ext_hidden_size", default=768, type=int)

	parser.add_argument("-optim", default='adam', type=str)
	parser.add_argument("-lr", default=1, type=float)
	parser.add_argument("-max_grad_norm", default=0, type=float)
	parser.add_argument("-param_init", default=0, type=float)
	parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)
	parser.add_argument("-beta1", default= 0.9, type=float)
	parser.add_argument("-beta2", default=0.999, type=float)
	parser.add_argument("-warmup_steps", default=8000, type=int)
	parser.add_argument("-warmup_steps_bert", default=8000, type=int)
	parser.add_argument("-warmup_steps_dec", default=8000, type=int)

	parser.add_argument("-save_checkpoint_steps", default=100, type=int)
	parser.add_argument("-accum_count", default=1, type=int)
	parser.add_argument("-report_every", default=1, type=int)
	parser.add_argument("-train_steps", default=1000, type=int)

	parser.add_argument('-visible_gpus', default='0', type=str)
	parser.add_argument('-gpu_ranks', default='0', type=str)

	args = parser.parse_args()

	args.device 	= 'cpu' if args.visible_gpus == '-1' else 'cuda'
	device_id		= 0 if args.device == 'cuda' else -1
	args.gpu_ranks	= [int(i) for i in range(len(args.visible_gpus.split(',')))]
	args.world_size = len(args.gpu_ranks)
	#os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus

	if args.task == 'abs':
		print('Not implemented yet')

	elif args.task == 'ext':
		if args.mode == 'train':
			train_ext(args, device_id)
