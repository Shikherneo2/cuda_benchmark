import os
import time as timer

from apex import amp
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from support import toy_batch, toy_batch16, default_params, write_results, print_results, check_results

# Experiment_type
bench = 'pytorch_cudnnGRU_half_precision'
version = torch.__version__
torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
experiment = '1x320-LSTM_cross-entropy'

# Get data
bX, b_lenX, bY, classes = toy_batch()
batch_size, seq_len, inp_dims = bX.shape
rnn_size, learning_rate, batches = default_params()

# PyTorch compatibility: time first, batch second
bX = np.transpose(bX, (1, 0, 2))

# Create Network
class Net(nn.Module):
  def __init__(self):
      super(Net, self).__init__()
      self.gru = nn.GRU( input_size=inp_dims, hidden_size=rnn_size, num_layers=1, bias=True, bidirectional=False )
      self.fc1 = nn.Linear( rnn_size, rnn_size//2, bias=False )
      self.fc2 = nn.Linear( rnn_size//2, classes, bias=False )
      self.relu = nn.ReLU()

  def forward(self, x):
      h1, state = self.gru(x)
      h2 = h1[-1, :, :]
      h3 = self.relu( self.fc1( h2 ) )
      h4 = self.fc2( h3 )
      return h4


net = Net()
net.cuda()

# net = torch.jit.trace(net)

# Print parameter count
params = 0
for param in list(net.parameters()):
    sizes = 1
    for el in param.size():
        sizes = sizes * el
    params += sizes

print("\n")
print("-----------------------------------------------------------------------")
print('# network parameters: ' + str(params))

# Create optimizer
optimizer = optim.Adam(net.parameters(), lr=learning_rate,)

# 01
# , keep_batchnorm_fp32=None
# net, optimizer = amp.initialize(net, optimizer, opt_level='O1')
criterion = nn.CrossEntropyLoss()  # loss definition

# Check for correct sizes
assert ( net.fc1.in_features == rnn_size )  # final projection input size (rnn_size)
assert ( net.fc2.weight.cpu().data.numpy().shape == ( classes, rnn_size//2) )  # final projection output size (classes, rnn_size)

bXt = Variable(torch.from_numpy(bX).cuda())
torch.cuda.synchronize()
output = net(bXt)
output_numpy = output.data.cpu().numpy()
assert (output_numpy.shape == (batch_size, classes))

# Start training
batch_time = []
batch_loss = []
train_start = timer.perf_counter()
for i in range(batches):
    torch.cuda.synchronize() # synchronize function call for precise time measurement
    batch_start = timer.perf_counter()

    bXt = Variable(torch.from_numpy(bX).cuda())
    bYt = Variable(torch.from_numpy(bY).cuda())

    optimizer.zero_grad()
    output = net(bXt)
    loss = criterion(output, bYt.long())
    loss.backward()
    optimizer.step()

    torch.cuda.synchronize() # synchronize function call for precise time measurement
    batch_end = timer.perf_counter()
    batch_time.append(batch_end - batch_start)
    batch_loss.append(float(loss.data.cpu().numpy()))
train_end = timer.perf_counter() # end of training

# Write results
print_results(batch_time)
check_results(batch_loss, batch_time, train_start, train_end)
write_results(script_name=os.path.basename(__file__), bench=bench, experiment=experiment, parameters=params,
              run_time=batch_time, version=version, mode="cudnn_gru")
