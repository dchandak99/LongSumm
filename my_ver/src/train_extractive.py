import random

import torch

from models import data_loader, model_builder
from models.model_builder import ExtSummarizer
from models.trainer_ext import build_trainer 
from models.data_loader import load_dataset

model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']

def train_ext(args, device_id):
	train_single_ext(args, device_id)

def train_single_ext(args, device_id):
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

	def train_iter_fct():
		return data_loader.Dataloader(args, load_dataset(args, 'train', shuffle=True), args.batch_size, args.device, 
										shuffle=True, is_test=False)

	model = ExtSummarizer(args, checkpoint)
	optim = model_builder.build_optim(args, model, checkpoint)

	trainer = build_trainer(args, device_id, model, optim)
	trainer.train(train_iter_fct, args.train_steps)





	