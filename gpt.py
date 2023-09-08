import torch 
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
import json 
from dataclasses import dataclass 
import rotary_embedding 
import einops 

@dataclass 
class ModelArgs: 
	block_size = 512
	device = 'mps'
	n_embd = 768
	n_head = 4
	n_layer = 4
	dropout = 0.2
	batch_size = 64
	lr = 3e-4

config = ModelArgs() 

class attention_head(nn.Module): 
	def __init__(self, head_size=config.n_embd//config.n_head): 
		super().__init__() 
		self.key = nn.Linear(config.n_embd, head_size, bias=False) 
		self.query = nn.Linear(config.n_embd, head_size, bias=False) 
		self.value = nn.Linear(config.n_embd, head_size, bias=False) 
		self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
		self.dropout = nn.Dropout(config.dropout) 
		self.rotary_emb = rotary_embedding.rotary_embedding(dim=32)
	def forward(self, x): 
		B, T, C = x.shape 
		k = self.key(x) 
		q = self.query(x) 

		k = self.rotary_emb.rotate_queries_or_keys(k.unsqueeze(1)).squeeze(1)
		q = self.rotary_emb.rotate_queries_or_keys(q.unsqueeze(1)).squeeze(1)

		weight = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
		weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
		weight = F.softmax(weight, dim=-1)
		weight = self.dropout(weight) 
		v = self.value(x) 
		out = weight @ v 
		return out 

class multi_head_attention(nn.Module): 
	def __init__(self, num_heads=config.n_head, head_size=config.n_embd//config.n_head): 
		super().__init__() 
		self.heads = nn.ModuleList([attention_head() for _ in range(num_heads)])
		self.proj = nn.Linear(head_size * num_heads, config.n_embd)
		self.dropout = nn.Dropout(config.dropout)

	def forward(self, x): 
		out = torch.cat([h(x) for h in self.heads], dim=-1)
		out = self.dropout(self.proj(out)) 
		return out 

class feed_forward(nn.Module): 
	def __init__(self, n_embd=config.n_embd):
		super().__init__() 
		self.net = nn.Sequential(
			nn.Linear(n_embd, 4*n_embd), 
			nn.ReLU(), 
			nn.Linear(4*n_embd, n_embd), 
			nn.Dropout(config.dropout) 
		)

	def forward(self, x): 
		return self.net(x) 

class block(nn.Module): 
	def __init__(self, n_embd=config.n_embd, n_head=config.n_head): 
		super().__init__() 
		self.sa = multi_head_attention() 
		self.ff = feed_forward() 
		self.ln1 = nn.LayerNorm(n_embd) 
		self.ln2 = nn.LayerNorm(n_embd) 

	def forward(self, x): 
		x = x + self.sa(self.ln1(x)) 
		x = x + self.ff(self.ln2(x))
		return x 

class gpt_model(nn.Module): 
	def __init__(self, vocab_size): 
		super().__init__() 
		self.token_embedding = nn.Embedding(vocab_size, config.n_embd)
		self.blocks = nn.Sequential(*[block() for _ in range(config.n_layer)])
		self.ln_f = nn.LayerNorm(config.n_embd) 
		self.lm_head = nn.Linear(config.n_embd, vocab_size) 

		self.apply(self._init_weights)
	
	def _init_weights(self, module): 
		if isinstance(module, nn.Linear): 
			nn.init.normal_(module.weight, mean=0.0, std=0.02) 
			if module.bias is not None: 
				nn.init.zeros_(module.bias) 
		elif isinstance(module, nn.Embedding): 
			nn.init.normal_(module.weight, mean=0.0, std=0.02)

	def forward(self, idx):
		# sentence converted into indices of words
		B, T = idx.shape 

		tok_emb = self.token_embedding(idx) 
		x = self.blocks(tok_emb) 
		x = self.ln_f(x) 
		logits = self.lm_head(x) 

		return logits 
	
	# def generate(self, idx, max_new_tokens, limit_sentence=False): 
	# 	if limit_sentence: 
	# 		for _ in range(max_new_tokens): 
	# 			idx_cond = idx[:, -config.block_size:]
	# 			logits = self(idx_cond) 
	# 			logits = logits[:, -1, :]
	# 			probs = F.softmax(logits, dim=-1)
	# 			idx_next = torch.multinomial(probs, num_samples=1)
	# 			idx = torch.cat((idx, idx_next), dim=1) 
	# 			if idx[0, -1].item() == data.encode(['\n'])[0]: 
	# 				return idx 
	# 	else: 
	# 		for _ in range(max_new_tokens): 
	# 			idx_cond = idx[:, -config.block_size:]
	# 			logits = self(idx_cond) 
	# 			logits = logits[:, -1, :]
	# 			probs = F.softmax(logits, dim=-1)
	# 			idx_next = torch.multinomial(probs, num_samples=1)
	# 			idx = torch.cat((idx, idx_next), dim=1) 
	# 	return idx 
