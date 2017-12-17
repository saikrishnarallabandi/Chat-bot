import os,sys
sys.path.append('/home3/srallaba/hacks/repos/clustergen_steroids')
import dynet as dy
from building_blocks.EncoderDecoderModels import EncoderDecoderModel 
from collections import defaultdict
import time, random

exp_name = sys.argv[1]
hidden = sys.argv[2]
debug = 0

arch = str(hidden) + 'SELU'
logfile_name = exp_name + '/logs/log_' + arch + '.log'
g = open(logfile_name,'w')
g.close()


# Declare paths to data
data_folder = '/home3/srallaba/data/chat_data'
train_encoder = data_folder  +'/train.enc'
train_decoder = data_folder + '/train.dec'
test_encoder = data_folder + '/test.enc'
test_decoder = data_folder + '/test.dec'

def read_file(fname,wids):
   f = open(fname)
   arr = []
   for line in f:
     line = line.split('\n')[0].split()
     s = []
     for k in line:
        s.append(wids[k])
     arr.append(s)
   return arr 
   
def get_embedding(arr):
   embed_arr = []
   for k in arr:
      #print k, M 
      embed_arr.append(dy.lookup(M,k).value())
   return embed_arr   

# Load data
global train_enc_wids, train_dec_wids, test_enc_wids, test_dec_wids
train_enc_wids = defaultdict(lambda:  len(train_enc_wids))
train_src = read_file(train_encoder, train_enc_wids)
train_dec_wids = defaultdict(lambda:  len(train_dec_wids))
train_tgt = read_file(train_decoder,train_dec_wids)

test_enc_wids = defaultdict(lambda:  len(test_enc_wids))
test_src = read_file(test_encoder, test_enc_wids)
test_dec_wids = defaultdict(lambda:  len(test_dec_wids))
test_tgt = read_file(test_decoder,test_dec_wids)

train_data = zip(train_src, train_tgt)
test_data = zip(test_src, test_tgt)
length = len(train_enc_wids) + len(train_dec_wids) + len(test_enc_wids) + len(test_dec_wids)
num_train = len(train_data)

# Hyperparameters 
units_hidden = int(hidden)
units_embedding = 128
num_layers = 1
units_attention = 128

# Instantiate DNN and define the loss
m = dy.Model()
global M
M = m.add_lookup_parameters((length, units_embedding))
edm = EncoderDecoderModel(m, [num_layers, units_embedding, units_hidden, units_attention, units_embedding, dy.tanh, M])
trainer = dy.AdamTrainer(m)
update_params = 32


for epoch in range(30):
  start_time = time.time()
  print " Epoch ", epoch
  train_loss = 0
  random.shuffle(train_data)
  count = 0
  l = 0
  for (src,tgt) in train_data:
     if len(src) == 0 or len(tgt) == 0 :
        continue
     start = time.time()
     dy.renew_cg()
     count += 1
     src_embed = get_embedding(src)    
     l += len(src)
     tgt_embed = get_embedding(tgt)
     if debug:
        print "Lengths: ", len(src_embed), len(tgt_embed)
     loss = edm.calculate_loss(src_embed, tgt_embed)
     train_loss += loss.value()
     if count % 1000 == 1:
        print "   Train Loss after processing ", count , " sequences: ", float( train_loss / l ), " in ", num_train
  print "Train Loss after epoch " +  str(epoch) + " : " +  str(float(train_loss/l)) 
