import os
import random
import argparse
from tqdm import tqdm

import torch

from models import data_loader
from models.data_loader import load_dataset
from models.trainer_ext import build_trainer 
from models.model_builder import ExtSummarizer

model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def train_ext(args):
	torch.manual_seed(args.seed)
	random.seed(args.seed)
	torch.backends.cudnn.deterministic = True

	#TODO -> add ability to load model from chkpt
	if args.train_from != '':
		checkpoint = torch.load(args.train_from, map_location=lambda storage, loc: storage)
		
		opt = vars(checkpoint['opt'])
		for k in opt.keys():
			if k in model_flags:
				setattr(args, k, opt[k])

	else:
		checkpoint = None

	model = ExtSummarizer(args, checkpoint)
	optim = model_builder.build_optim(args, model, checkpoint)

	train_iter = data_loader.load_text(args, args.text_src, args.text_tgt, device)

	for i in range(train_iter):
		print(i)



def test_text_ext(args):
	checkpoint = torch.load(args.test_from, map_location=lambda storage, loc: storage)
	
	opt = vars(checkpoint['opt'])
	for k in opt.keys():
		if (k in model_flags):
			setattr(args, k, opt[k])
	
	print(args)
	
	device = "cpu" if args.visible_gpus == '-1' else "cuda"
	device_id = 0 if device == "cuda" else -1

	model = ExtSummarizer(args, checkpoint)
	model.eval()

	test_iter = data_loader.load_text(args, args.text_src, args.text_tgt, device)

	trainer = build_trainer(args, device_id, model, None)
	trainer.test(test_iter, -1)


if __name__ == '__main__':
	parser = argparse.ArgumentParser() 

	parser.add_argument('-task', default='ext', type=str, choices=['ext', 'abs'])
	parser.add_argument("-mode", default='train', type=str, choices=['train', 'validate', 'test'])

	parser.add_argument("-model_path", default='models/')
	parser.add_argument("-result_path", default='models/')
	parser.add_argument("-test_from", default='models/bertext_cnndm_transformer.pt')

	parser.add_argument("-block_trigram", type=str2bool, nargs='?', const=True, default=True)

	parser.add_argument('-use_gpu', default=True, type=bool)
	parser.add_argument('-large', default=False, type=bool)
	parser.add_argument('-finetune_bert', default=True, type=bool)

	parser.add_argument("-ext_ff_size", default=2048, type=int)
	parser.add_argument("-ext_heads", default=8, type=int)
	parser.add_argument("-ext_dropout", default=0.2, type=float)
	parser.add_argument("-ext_layers", default=2, type=int)
	parser.add_argument("-ext_hidden_size", default=768, type=int)

	parser.add_argument("-max_pos", default=512, type=int) # Need to increase this

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

	parser.add_argument("-text_src", default='txt_data/source/test_1.txt')
	parser.add_argument("-text_tgt", default='')

	parser.add_argument("-accum_count", default=1, type=int)
	parser.add_argument("-report_every", default=1, type=int)
	parser.add_argument("-save_checkpoint_steps", default=100, type=int)
	parser.add_argument("-recall_eval", type=str2bool, nargs='?',const=True,default=False)

	args = parser.parse_args()

	args.device = 'cuda' if args.use_gpu else 'cpu'
	args.visible_gpus = '0'
	args.temp_dir = 'temp/'	

	device_id		= 0 if args.device == 'cuda' else -1
	args.gpu_ranks	= [int(i) for i in range(len(args.visible_gpus.split(',')))]
	args.world_size = len(args.gpu_ranks)

	if args.task == 'abs':
		print('Not implemented yet')

	elif args.task == 'ext':
		if args.mode == 'train':
			test_text_ext(args)


