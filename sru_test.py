import os
import time as timer

#from apex import amp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from sru import SRU
from support import toy_batch, toy_batch16, default_params, write_results, print_results, check_results

# Experiment_type
bench = 'pytorch_cudnnGRU_half_precision'
version = torch.__version__
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
experiment = '1x320-LSTM_cross-entropy'

# Get data
bX, b_lenX, bY, classes = toy_batch()
batch_size, seq_len, inp_dims = bX.shape
rnn_size, learning_rate, batches = default_params()

# PyTorch compatibility: time first, batch second
bX = np.transpose(bX, (1, 0, 2))

print( inp_dims  )
# Create Network
#model, _ = amp.initialize(model, [], opt_level="O3")
gru = nn.GRU( input_size=inp_dims, hidden_size=rnn_size, num_layers=1, bias=True, bidirectional=False )
#sru = SRU( input_size=inp_dims, hidden_size=rnn_size, num_layers = 2, dropout = 0.0, bidirectional = False, layer_norm = False, highway_bias = 0, rescale = True )


# Start training
batch_time = []
batch_loss = []
train_start = timer.perf_counter()
for i in range(batches):
    torch.cuda.synchronize() # synchronize function call for precise time measurement
    batch_start = timer.perf_counter()

    bXt = Variable(torch.from_numpy(bX).cuda())
    bYt = Variable(torch.from_numpy(bY).cuda())

    output = gru(bXt)

    torch.cuda.synchronize() # synchronize function call for precise time measurement
    batch_end = timer.perf_counter()
train_end = timer.perf_counter() # end of training

print((train_end-train_start)/batched)
