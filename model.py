import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim

class Encoder(nn.Module):
  def __init__(self, vocab_size, emb_size, hid_size, embed=True, embedding_weights=None):
    super(Encoder, self).__init__() 
    self.embed = embed
    if self.embed:
      self.embedding = nn.Embedding(vocab_size, emb_size, padding_idx=1)
    if embedding_weights is not None:
      self.embedding.weight = embedding_weights
    self.encoder = nn.LSTM(emb_size, hid_size, bidirectional=True)
    self.hid_size = hid_size

  def forward(self, seqs, lens):
    # Embed
    if self.embed:
      emb_seqs = self.embedding(seqs)
    else:
      emb_seqs = seqs

    # Sort by length
    sort_idx = sorted(range(len(lens)), key=lambda i: -lens[i])
    emb_seqs = emb_seqs[:,sort_idx]
    lens = [lens[i] for i in sort_idx]

    # Pack sequence
    packed = torch.nn.utils.rnn.pack_padded_sequence(emb_seqs, lens)

    # Forward pass through LSTM
    outputs, hidden = self.encoder(packed)

    # Unpack outputs
    outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(outputs)
    
    # Unsort
    unsort_idx = sorted(range(len(lens)), key=lambda i: sort_idx[i])
    outputs = outputs[:,unsort_idx]
    hidden = (hidden[0][:,unsort_idx], hidden[1][:,unsort_idx])
    hidden = (torch.cat([hidden[0][0], hidden[0][1]], dim=-1).unsqueeze(0), torch.cat([hidden[1][0], hidden[1][1]], dim=-1).unsqueeze(0))


    return outputs, hidden

class DualEncoder(nn.Module):
  def __init__(self, context_encoder, response_encoder, w2i, i2w, args):
    super(DualEncoder, self).__init__()

    self.args = args

    # Model
    self.context_encoder = context_encoder
    self.response_encoder = response_encoder 

    self.r_proj = nn.Linear(context_encoder.hid_size, context_encoder.hid_size)
    self.c_proj = nn.Linear(context_encoder.hid_size, context_encoder.hid_size)

    # Vocab
    self.i2w = i2w
    self.w2i = w2i

    # Training
    self.criterion = nn.CrossEntropyLoss()
    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)

  def prep_batch(self, rows):
    def _pad(arr, pad=3):
      # Given an array of integer arrays, pad all arrays to the same length
      lengths = [len(e) for e in arr]
      max_len = max(lengths)
      return [e+[pad]*(max_len-len(e)) for e in arr], lengths

    contexts = [[self.w2i.get(w, self.w2i['_UNK']) for w in row[0]] for row in rows]
    input_seq, input_lens = _pad(contexts, pad=self.w2i['_PAD'])
    input_seq = torch.LongTensor(input_seq).t()

    cands = [[self.w2i.get(w, self.w2i['_UNK']) for w in resp] for row in rows for resp in row[1:]]
    cand_seq, cand_lens = _pad(cands, pad=self.w2i['_PAD'])
    cand_seq = torch.LongTensor(cand_seq).t()

    return input_seq, input_lens, cand_seq, cand_lens

  def prep_replies(self, replies):
    def _pad(arr, pad=3):
      # Given an array of integer arrays, pad all arrays to the same length
      lengths = [len(e) for e in arr]
      max_len = max(lengths)
      return [e+[pad]*(max_len-len(e)) for e in arr], lengths

    replies = [[self.w2i.get(w, self.w2i['_UNK']) for w in resp] for resp in replies]
    r_seq, r_lens = _pad(replies, pad=self.w2i['_PAD'])
    r_seq = torch.LongTensor(r_seq).t()

    return r_seq, r_lens

  def forward(self, ctx_seq, ctx_lens, resp_seq, resp_lens):
    # Encode
    ctx_outputs, ctx_hidden = self.context_encoder(ctx_seq, ctx_lens)
    rsp_outputs, rsp_hidden = self.response_encoder(resp_seq, resp_lens)

    _, batch_size, hid_size = ctx_hidden[0].size()
    ctx_hidden = ctx_hidden[0].permute(1,0,2)
    rsp_hidden = rsp_hidden[0].view(batch_size, -1, hid_size).permute(0,2,1)

    # Log softmax is done in criterion 
    return ctx_hidden.bmm(rsp_hidden)

  def train(self, ctx_seq, ctx_lens, resp_seq, resp_lens):
    self.optim.zero_grad()

    # Forward
    proba = self.forward(ctx_seq, ctx_lens, resp_seq, resp_lens)

    # Loss
    loss = self.criterion(proba.squeeze(1), torch.zeros(proba.size(0)).long().to(proba.device)) 

    # Backwards
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
    self.optim.step()

    return loss.item()

  def save(self, name):
    torch.save(self.context_encoder, name+'.ctx')
    torch.save(self.response_encoder, name+'.rsp')

  def load(self, name):
    self.context_encoder.load_state_dict(torch.load(name+'.ctx').state_dict())
    self.response_encoder.load_state_dict(torch.load(name+'.rsp').state_dict())

class MultiClassifierEncoder(nn.Module):
  def __init__(self, context_encoder, w2i, i2w, dial_acts, args):
    super(MultiClassifierEncoder, self).__init__()
    self.args = args

    # Model
    self.context_encoder = context_encoder
    self.linear = nn.Linear(2*args.hid_size, len(dial_acts))

    # Vocab
    self.i2w = i2w
    self.w2i = w2i
    self.dial_acts = dial_acts

    # Training
    self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([5.0]))
    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)

  def prep_batch(self, rows, dial_acts):
    def _pad(arr, pad=3):
      # Given an array of integer arrays, pad all arrays to the same length
      lengths = [len(e) for e in arr]
      max_len = max(lengths)
      return [e+[pad]*(max_len-len(e)) for e in arr], lengths

    contexts = [[self.w2i.get(w, self.w2i['_UNK']) for w in row[0]] for row in rows]
    input_seq, input_lens = _pad(contexts, pad=self.w2i['_PAD'])
    input_seq = torch.LongTensor(input_seq).t()

    dial_acts = [[int(da in e) for da in self.dial_acts] for e in dial_acts]
    dial_acts = torch.FloatTensor(dial_acts)

    return input_seq, input_lens, dial_acts

  def prep_replies(self, replies):
    def _pad(arr, pad=3):
      # Given an array of integer arrays, pad all arrays to the same length
      lengths = [len(e) for e in arr]
      max_len = max(lengths)
      return [e+[pad]*(max_len-len(e)) for e in arr], lengths

    replies = [[self.w2i.get(w, self.w2i['_UNK']) for w in resp] for resp in replies]
    r_seq, r_lens = _pad(replies, pad=self.w2i['_PAD'])
    r_seq = torch.LongTensor(r_seq).t()

    return r_seq, r_lens

  def forward(self, ctx_seq, ctx_lens):
    # Encode
    ctx_outputs, ctx_hidden = self.context_encoder(ctx_seq, ctx_lens)

    return self.linear(ctx_hidden[0])

  def train(self, ctx_seq, ctx_lens, dial_acts):
    self.optim.zero_grad()

    # Forward
    proba = self.forward(ctx_seq, ctx_lens)

    # Loss
    loss = self.criterion(proba.squeeze(0), dial_acts)

    # Backwards
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
    self.optim.step()

    return loss.item()

  def save(self, name):
    torch.save(self.context_encoder, name+'.ctx')
    torch.save(self.linear, name+'.lin')

  def load(self, name):
    self.context_encoder.load_state_dict(torch.load(name+'.ctx').state_dict())
    self.linear.load_state_dict(torch.load(name+'.lin').state_dict())

class CombMultiClassifierEncoder(nn.Module):
  def __init__(self, context_encoders, w2i, i2w, dial_acts, args):
    super(CombMultiClassifierEncoder, self).__init__()
    self.args = args

    # Model
    self.context_encoders = nn.ModuleList(context_encoders)
    self.linear = nn.Linear(len(context_encoders)*2*args.hid_size, len(dial_acts))

    # Vocab
    self.i2w = i2w
    self.w2i = w2i
    self.dial_acts = dial_acts

    # Training
    self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([5.0]))
    self.optim = optim.Adam(lr=args.lr, params=self.parameters(), weight_decay=args.l2_norm)

  def prep_batch(self, rows, dial_acts):
    def _pad(arr, pad=3):
      # Given an array of integer arrays, pad all arrays to the same length
      lengths = [len(e) for e in arr]
      max_len = max(lengths)
      return [e+[pad]*(max_len-len(e)) for e in arr], lengths

    contexts = [[self.w2i.get(w, self.w2i['_UNK']) for w in row[0]] for row in rows]
    input_seq, input_lens = _pad(contexts, pad=self.w2i['_PAD'])
    input_seq = torch.LongTensor(input_seq).t()

    dial_acts = [[int(da in e) for da in self.dial_acts] for e in dial_acts]
    dial_acts = torch.FloatTensor(dial_acts)

    return input_seq, input_lens, dial_acts

  def prep_replies(self, replies):
    def _pad(arr, pad=3):
      # Given an array of integer arrays, pad all arrays to the same length
      lengths = [len(e) for e in arr]
      max_len = max(lengths)
      return [e+[pad]*(max_len-len(e)) for e in arr], lengths

    replies = [[self.w2i.get(w, self.w2i['_UNK']) for w in resp] for resp in replies]
    r_seq, r_lens = _pad(replies, pad=self.w2i['_PAD'])
    r_seq = torch.LongTensor(r_seq).t()

    return r_seq, r_lens

  def forward(self, ctx_seq, ctx_lens):
    # Encode
    ctx_hidden = torch.cat([m(ctx_seq, ctx_lens)[1][0] for m in self.context_encoders], dim=-1)

    return self.linear(ctx_hidden)

  def train(self, ctx_seq, ctx_lens, dial_acts):
    self.optim.zero_grad()

    # Forward
    proba = self.forward(ctx_seq, ctx_lens)

    # Loss
    loss = self.criterion(proba.squeeze(0), dial_acts)

    # Backwards
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.parameters(), self.args.clip)
    self.optim.step()

    return loss.item()

  def save(self, name):
    torch.save(self.linear, name+'.lin')

  def load(self, name):
    self.linear.load_state_dict(torch.load(name+'.lin').state_dict())
