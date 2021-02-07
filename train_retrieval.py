import argparse
import csv
import json
import math
import model
import pandas as pd
import random
import time
import torch
import pickle
from collections import Counter
import torch.nn.functional as F
import pathlib


parser = argparse.ArgumentParser(description='Retrieval')
parser.add_argument('--batch_size', type=int, default=32, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--vocab_size', type=int, default=400, metavar='V')
parser.add_argument('--emb_size', type=int, default=300)
parser.add_argument('--hid_size', type=int, default=150)
parser.add_argument('--cell_type', type=str, default='bilstm')
parser.add_argument('--num_layers', type=int, default=1, help='number of layers')

parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--l2_norm', type=float, default=0.00001)
parser.add_argument('--clip', type=float, default=5.0, help='clip the gradient by norm')

parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 1)')
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--model_path', type=str, default='model/baseline')
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--data_path', type=str, default='multiwoz/')
parser.add_argument('--data_size', type=float, default=-1.0)

parser.add_argument('--num_buckets', type=int, default=0, metavar='K')
parser.add_argument('--glove', action='store_true')
parser.add_argument('--cpu', action='store_true')
parser.add_argument('--threshold', type=float,default=None)
parser.add_argument('--eval', action='store_true')
parser.add_argument('--logits', action='store_true',help="use logits instead of softmax")

args = parser.parse_args()
if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:1")

random.seed(args.seed)
torch.manual_seed(args.seed)

def get_glove(vocab):
  words = pd.read_table('../glove.42B.300d.zip', sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
  tensors = []
  for w in vocab:
    if w in words.index:
      tensors.append(torch.FloatTensor(words.loc[w].tolist()).to(device))
    else:
      tensors.append(torch.zeros(300).to(device))
  embedding = torch.stack(tensors).to(device)
  return embedding

def load_data(path,skipTrain=False):
  def _load(fname):
    if 'ubuntu' in path or 'dam' in path:
      return [[m.lower().split()[-160:] for m in row] for row in csv.reader(open(fname))]
    else:
      return [[m.lower().split() for m in row] for row in csv.reader(open(fname))]
    
  if skipTrain:
      return (_load(path + 'valid.csv'),
          _load(path + 'test.csv'))
  else:
      return (_load(path + 'train.csv'),
          _load(path + 'valid.csv'),
          _load(path + 'test.csv'))


# Build vocabulary
def build_vocab(data, size=10000,path=None):
  if data is None and path is not None:
      with open(path+"vocab_{}.pkl".format(size),"rb") as f:
          vocab = pickle.load(f)
      #vocab.insert(2,"_NOTA")
  else:
      all_words = [word for conv in data for msg in conv for word in msg]
      vocab = ['_UNK', '_PAD'] + [e[0] for e in Counter(all_words).most_common(size)]
      if path is not None:
          with open(path+"vocab_{}.pkl".format(size),"wb") as f:
              pickle.dump(vocab,f)
  return vocab, {w:i for i,w in enumerate(vocab)}


def evaluate_specific(data, model,suffix=""):
  # Evaluate
  if args.threshold is None:
    print("direct predict task")
  else:
    print("threshold task,threshold={}".format(args.threshold))
  if not args.logits:
    print("use softmax")
  correct = 0
  correct2 = 0
  correct3 = 0
  correct4 = 0
  total = 0
  total2 = 0
  mrr = 0
  nota_fp=nota_fp2=nota_tp=nota_tp2=nota_fn=nota_fn2=nota_tn=nota_tn2=0
  prob_nota = []
  prob_other = []
  num_batches = math.ceil(len(data)/args.batch_size)
  for i in range(num_batches):
    batch_rows = data[i*args.batch_size:(i+1)*args.batch_size]
    is_nota = [e[1][0]=='_nota' for e in batch_rows]
    if len(batch_rows) == 0:
      continue
    if args.threshold is not None:#remove _nota for threshold experiments
      batch_rows[:] = [list(filter((["_nota"]).__ne__, b)) for b in batch_rows]
    ctx_seq, ctx_lens, resp_seq, resp_lens = model.prep_batch(batch_rows)
    ctx_seq = ctx_seq.to(device)
    resp_seq = resp_seq.to(device)
    # Forward pass
    proba = model.forward(ctx_seq, ctx_lens, resp_seq, resp_lens)
    if not args.logits:
      proba = F.softmax(proba.squeeze(1), dim=-1)
    proba = proba.squeeze(1)
   
    # Compute metrics 
    alter_best = []
    for b in range(len(batch_rows)):
      if args.threshold is None:
        nota_pred = batch_rows[b][proba[b].argmax()+1][0]=="_nota"
        nota_pred2 = batch_rows[b][proba[b,:2].argmax()+1][0]=="_nota"
      else:
        nota_pred = proba[b].max().item() < args.threshold
        nota_pred2 =proba[b,:2].max().item() < args.threshold
      if is_nota[b]:
        prob_nota.append(proba[b].cpu().detach())
        correct3 += int(nota_pred)
        correct4 += int(nota_pred2)
        nota_tp += int(nota_pred)
        nota_tp2 += int(nota_pred2)
        nota_fn += int(not nota_pred)
        nota_fn2 += int(not nota_pred2)
      else:
        prob_other.append(proba[b].cpu().detach())
        correct += int(proba[b].argmax() == 0 and not nota_pred)
        correct2 += int(proba[b,:2].argmax() == 0 and not nota_pred2)
        correct3 += int(not nota_pred)
        correct4 += int(not nota_pred2)
        nota_fp += int(nota_pred)
        nota_fp2 += int(nota_pred2)
        nota_tn +=int(not nota_pred)
        nota_tn2 +=int(not nota_pred2)
        total += 1
    total2 += proba.size(0)
  with open(args.model_path+"{}_{}_distri{}.pkl".format("logits" if args.logits else "prob",99,suffix),"wb") as f:
    pickle.dump(prob_nota,f)
    pickle.dump(prob_other,f)
  #print("max non-NOTA",max(alter_best))
  print('Current Test R@1/100:{:.4f}'.format(correct/total))
  #print('Current Test R@1/2:{:.4f}'.format(correct2/total))
  print('Current Test NOTA@100 ACC:{:.4f}'.format(correct3/total2))
  #print('Current Test NOTA@2 ACC:{:.4f}'.format(correct4/total2))
  #print('NOTA percentage:'.format(total2/total))
  f1_nota,f1_other = calculate_f1_nota(nota_tp,nota_fp,nota_fn,nota_tn)
  #f1_nota2,f1_other2 = calculate_f1_nota(nota_tp2,nota_fp2,nota_fn2,nota_tn2)
  print('NOTA@100 F1:{:.4f}'.format(f1_nota))
  #print('NOTA@2 F1:{:.4f}'.format(f1_nota2))
  print('non-NOTA@100 F1:{:.4f}'.format(f1_other))
  #print('non-NOTA@2 F1:{:.4f}'.format(f1_other2))
  #print("weighted F1:{:.4f}".format((f1_nota+f1_nota2+f1_other+f1_other2)/4))
    
  return correct/total

def calculate_f1_nota(TP,FP,FN,TN):
  #print("TP: ",TP," FP: ",FP,"FN: ",FN,"TN: ",TN)
  #NOTA
  precision_nota = TP / (TP + FP)
  recall_nota = TP / (TP + FN)
  F1_nota = 2 * precision_nota * recall_nota / (precision_nota + recall_nota)
  #non-NOTA
  precision_other = TN / (TN + FN)
  recall_other = TN / (TN + FP)
  F1_other = 2 * precision_other * recall_other / (precision_other + recall_other)
  return F1_nota,F1_other


def load(name,i2w,w2i):
  ctx_enc = model.Encoder(vocab_size=len(i2w), 
                          emb_size=args.emb_size, 
                          hid_size=args.hid_size,
                          embedding_weights=None)
  r_enc = model.Encoder(vocab_size=len(i2w), 
                        emb_size=args.emb_size, 
                        hid_size=args.hid_size,
                        embedding_weights=None)
  d_enc = model.DualEncoder(context_encoder=ctx_enc,
                            response_encoder=r_enc,
                            w2i=w2i,
                            i2w=i2w,
                            args=args)
  d_enc = d_enc.to(device)
  d_enc.load(name)
  return d_enc

#glove_embed = get_glove(i2w)
#torch.save(glove_embed, 'glove_embed')
emb_w = None
if args.glove:
    print("Loading glove embedding....")
    glove_embed = torch.load('glove_embed')
    glove_embed2 = glove_embed.clone()
    emb_w = torch.nn.Parameter(glove_embed.clone())

start_epoch = 0
if args.load_model and not args.eval:
    dual_encoder = load(args.load_model)
    start_epoch = int(args.load_model.strip()[-1])

if args.eval:
    if not args.load_model:
        print("missing the model to load")
    valid, test = load_data(args.data_path,skipTrain=True)
    i2w, w2i = build_vocab(None,path=args.data_path)
    #train,valid, test = load_data(args.data_path)
    #i2w, w2i = build_vocab(train,path=args.data_path)
    dual_encoder = load(args.load_model,i2w, w2i)
    evaluate_specific(valid, dual_encoder)
    evaluate_specific(test,dual_encoder,suffix="test")
else:
    # Load all the data
    train, valid, test = load_data(args.data_path)

    if args.data_size >= 0:
      train = train[:int(len(train)*args.data_size)]

    print("Number of training instances:", len(train))
    print("Number of validation instances:", len(valid))
    print("Number of test instances:", len(test))
    i2w, w2i = build_vocab(train,path=args.data_path)
    context_encoder = model.Encoder(vocab_size=len(i2w), 
                                    emb_size=args.emb_size, 
                                    hid_size=args.hid_size,
                                    embedding_weights=emb_w)
    response_encoder = model.Encoder(vocab_size=len(i2w), 
                                     emb_size=args.emb_size, 
                                     hid_size=args.hid_size,
                                     embedding_weights=emb_w)
    dual_encoder = model.DualEncoder(context_encoder=context_encoder,
                                     response_encoder=response_encoder,
                                     w2i=w2i,
                                     i2w=i2w,
                                     args=args)
    dual_encoder = dual_encoder.to(device)
    best_valid = 0.0
    best_epoch = 0
    for epoch in range(start_epoch,args.num_epochs):
      # Train
      indices = list(range(len(train)))
      random.shuffle(indices)
      num_batches = math.ceil(len(train)/args.batch_size)
      cum_loss = 0
      for batch in range(num_batches):
        # Prepare batch
        batch_indices = indices[batch*args.batch_size:(batch+1)*args.batch_size]
        batch_rows = [train[i] for i in batch_indices]

        # Sample negs
        if args.num_buckets > 0: #and args.num_buckets < 10:
          negs = sample_negs(reply_encs, batch_indices, args.num_buckets, bucket_ind)
          batch_rows = [batch_row[:2] + [all_replies[i] for i in neg] for batch_row,neg in zip(batch_rows,negs)]
        

        if len(batch_rows) == 0:
          continue
        ctx_seq, ctx_lens, resp_seq, resp_lens = dual_encoder.prep_batch(batch_rows)

        ctx_seq = ctx_seq.to(device)
        resp_seq = resp_seq.to(device)

        # Train batch
        cum_loss += dual_encoder.train(ctx_seq, ctx_lens, resp_seq, resp_lens)

        # Log batch if needed
        if batch > 0 and batch % 50 == 0:
          print("Epoch {0}/{1} Batch {2}/{3} Avg Loss {4:.4f}".format(epoch+1, args.num_epochs, batch, num_batches, cum_loss/50))
          cum_loss = 0

      valid_score = evaluate_specific(valid, dual_encoder)
      if valid_score > best_valid:
          best_valid = valid_score
          best_epoch = epoch 
      evaluate_specific(test,dual_encoder)
      dual_encoder.save("{0}_{1}".format(args.model_path, epoch))

    print("Best R@1/10 score: {}, after epoch {}".format(best_valid,best_epoch))
