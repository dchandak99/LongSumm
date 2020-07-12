

import torch
import torch.nn as nn

from models.optimizers import Optimizer
from transformers import BertModel, BertConfig
from models.encoder import Classifier, ExtTransformerEncoder

class Bert(nn.Module): #Calls BertModel
	def __init__(self, large, temp_dir, finetune=False):
		super(Bert, self).__init__()

		if large:
			self.model = BertModel.from_pretrained('bert-large-uncased')

		else:
			self.model = BertModel.from_pretrained('bert-base-uncased')

		self.finetune = finetune

	def forward(self, x, segs, mask):
		if self.finetune:
			self.train()
			top_vec, _ = self.model(input_ids=x, token_type_ids=segs, attention_mask=mask)

		else:
			self.eval()
			with torch.no_grad():
				top_vec, _ = self.model(input_ids=x, token_type_ids=segs, attention_mask=mask)

		return top_vec

class ExtSummarizer(nn.Module):
	def __init__(self, args, checkpoint):
		super(ExtSummarizer, self).__init__()

		self.args = args
		self.device = args.device
		
		self.bert = Bert(args.large, args.temp_dir, args.finetune_bert)
		self.ext_layer = ExtTransformerEncoder(self.bert.model.config.hidden_size, args.ext_ff_size,
												args.ext_heads, args.ext_dropout, args.ext_layers)

		if (args.encoder == 'baseline'):
			bert_config = BertConfig(self.bert.model.config.vocab_size, hidden_size=args.ext_hidden_size, 
										num_hidden_layers=args.ext_layers, num_attention_heads=args.ext_heads,
										intermediate_size=args.ext_ff_size)

			self.bert.model = BertModel(bert_config)
			self.ext_layer = Classifier(self.bert.model.config.hidden_size)


		if args.max_pos > 512:
			my_pos_embeddings = nn.Embeddings(args.max_pos, self.bert.model.config.hidden_size)
			my_pos_embeddings.weight.data[:512] = self.bert.model.embeddings.position_embeddings.weight.data
			my_pos_embeddings.weight.data[512:] = self.bert.model.embeddings.position_embeddings.weight.data[-1][None, :].repeat(args.max_pos - 512, 1)
			self.bert.model.embeddings.position_embeddings = my_pos_embeddings

		if checkpoint is not None:
			self.load_state_dict(checkpoint['model'], strict=True)
		
		else:
			if args.param_init != 0.0:
				for p in self.ext_layer.parameters():
					p.data.uniform_(-args.param_init, args.param_init)
			
			if args.param_init_glorot:
				for p in self.ext_layer.parameters():
					if p.dim() > 1:
						nn.init.xavier_uniform_(p)

		self.to(self.device)

	def forward(self, src, segs, clss, mask_src, mask_cls):
		top_vec = self.bert(src, segs, mask_src)
		sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]
		sents_vec = sents_vec * mask_cls[:, :, None].float()
		sent_scores = self.ext_layer(sents_vec, mask_cls).squeeze(-1)
		return sent_scores, mask_cls

def build_optim(args, model, checkpoint):
	""" Build optimizer """

	if checkpoint is not None:
		optim = checkpoint['optim'][0]
		saved_optimizer_state_dict = optim.optimizer.state_dict()
		optim.optimizer.load_state_dict(saved_optimizer_state_dict)
		if args.visible_gpus != '-1':
			for state in optim.optimizer.state.values():
				for k, v in state.items():
					if torch.is_tensor(v):
						state[k] = v.cuda()

		if (optim.method == 'adam') and (len(optim.optimizer.state) < 1):
			raise RuntimeError(
				"Error: loaded Adam optimizer from existing model" +
				" but optimizer state is empty")

	else:
		optim = Optimizer(
			args.optim, args.lr, args.max_grad_norm,
			beta1=args.beta1, beta2=args.beta2,
			decay_method='noam',
			warmup_steps=args.warmup_steps)

	optim.set_parameters(list(model.named_parameters()))


	return optim