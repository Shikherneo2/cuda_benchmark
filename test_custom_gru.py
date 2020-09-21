import math
import time
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

torch.backends.cudnn.enabled = False
fc1 = nn.Linear(512, 512).cuda()
fc2 = nn.Linear(512, 512).cuda()

class GRUCell(torch.jit.ScriptModule):

    """
    An implementation of GRUCell.

    """

    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias_ih = torch.nn.parameter.Parameter(torch.randn(3 * hidden_size))
        self.weight_ih = torch.nn.parameter.Parameter(torch.randn(hidden_size, 3 * hidden_size))
        self.weight_hh = torch.nn.parameter.Parameter(torch.randn( hidden_size, 3 * hidden_size))
        self.bias_hh = torch.nn.parameter.Parameter(torch.randn(3 * hidden_size))

        self.weight_ih_1, self.weight_ih_2, self.weight_ih_3 = self.weight_ih.chunk( 3, 1 )
        self.bias_ih_1, self.bias_ih_2, self.bias_ih_3 = self.bias_ih.chunk( 3 )

        self.weight_hh_1, self.weight_hh_2, self.weight_hh_3 = self.weight_hh.chunk( 3, 1 )
        self.bias_hh_1, self.bias_hh_2, self.bias_hh_3 = self.bias_hh.chunk( 3 )

        self.bias1 = nn.Parameter(self.bias_hh_1 + self.bias_ih_1)
        self.bias2 = nn.Parameter(self.bias_hh_2 + self.bias_ih_2)

        self.weight_hh_1 = nn.Parameter(self.weight_hh_1)
        self.weight_hh_2 = nn.Parameter(self.weight_hh_2)
        self.weight_hh_3 = nn.Parameter(self.weight_hh_3)

        self.weight_ih_1 = nn.Parameter(self.weight_ih_1)
        self.weight_ih_2 = nn.Parameter(self.weight_ih_2)
        self.weight_ih_3 = nn.Parameter(self.weight_ih_3)

        self.bias_ih_3 = nn.Parameter(self.bias_ih_3)
        self.bias_hh_3 = nn.Parameter(self.bias_hh_3)

        self.linear_matrix1 = nn.Parameter( torch.randn(512, 512) )
        self.linear_matrix2 = nn.Parameter( torch.randn(512, 512) )

        self.linear_bias1 = nn.Parameter( torch.randn(512) )
        self.linear_bias2 = nn.Parameter( torch.randn(512) )
        # self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        # self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()


    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)
    
    @torch.jit.script_method
    def forward(self, x, hidden):
        
        x = x.view(-1, x.size(1))
        
        # gate_x = self.x2h(x) 
        # gate_h = self.h2h(hidden)
        i_r = torch.mm( x, self.weight_ih_1 )
        i_i = torch.mm( x, self.weight_ih_2 )
        i_n = torch.mm( x, self.weight_ih_3 ) + self.bias_ih_3

        h_r = torch.mm( x, self.weight_hh_1 )
        h_i = torch.mm( x, self.weight_hh_2 )
        h_n = torch.mm( x, self.weight_hh_3 ) + self.bias_hh_3
        # gate_x = torch.mm( x, self.weight_ih ) + self.bias_ih
        # gate_h = torch.mm( hidden, self.weight_hh ) + self.bias_hh
        
        # i_r, i_i, i_n = gate_x.chunk(3, 1)
        # h_r, h_i, h_n = gate_h.chunk(3, 1)
    
        resetgate = torch.sigmoid(i_r + h_r + self.bias1)
        inputgate = torch.sigmoid(i_i + h_i + self.bias2)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        
        hy = newgate + inputgate * (hidden - newgate)
        
        return hy


#gru = GRUCell( 512, 512 ).cuda()
gru = nn.GRUCell(512,512).cuda()

a = torch.from_numpy( np.array(np.random.rand( 2, 512 ), dtype=np.float32))
b = torch.from_numpy( np.zeros((2, 512), dtype=np.float32) )
#gru = torch.jit.script(gru)
#batch, input_dim -- seq is 1
inputs = [np.array(np.random.rand( 10, 512 ), dtype=np.float32) for i in range(500)]

# number of layers, batch size and input dimensions
hidden = np.zeros((1, 10, 512), dtype=np.float32)
hn = torch.from_numpy(hidden[0,:,:]).cuda()

print("Warm up..")
start = time.time()
for i in range(10):
    for x in inputs:
        x = torch.from_numpy(x).cuda()
        x = fc1( x+hn )
        hn = gru( x, hn )
        hn = fc2(hn)

end = time.time()
print("Time: "+ str((end-start)/10))

print("Warm up..")
start = time.time()
for i in range(10):
    for x in inputs:
        x = torch.from_numpy(x).cuda()
        x = fc1( x+hn )
        hn = gru( x, hn )
        hn = fc2(hn)

end = time.time()
print("Time: "+ str((end-start)/10))

print("Running..")
start = time.time()
for i in range(10):
    for x in inputs:
        x = torch.from_numpy(x).cuda()
        x = fc1( x+hn )
        hn = gru( x, hn )
        hn = fc2(hn)

end = time.time()
print("Time: "+ str((end-start)/10))
