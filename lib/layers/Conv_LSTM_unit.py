import torch
import torch.nn as nn
from torch.autograd import Variable

class ConvLSTMCell(nn.Module):
    """
    ConvLSTMCell originates from the idea of:
        Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting
        https://arxiv.org/abs/1506.04214
    The intuition is to replace the vector inner product with 2d-convolution in classical LSTM cell.

    However, the details in original paper is unconsistent with the notation and implementation for
    torch.nn.LSTMCeil(include previous C state into input gate, forget gate). For consistency
    , we use following rules the same as the official document:
        https://pytorch.org/docs/master/nn.html#torch.nn.LSTMCell

    Only replace Wx with 2d-conv(W, x) in implementation. Good tutorial for LSTM:
        english: http://colah.github.io/posts/2015-08-Understanding-LSTMs/
        chinese: https://www.jianshu.com/p/9dc9f41f0b29
    """
    def __init__(self, D_in, D_hidden, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        self.D_in = D_in
        self.D_hidden = D_hidden
        self.bias = bias
        self.kernel_size = kernel_size

        # padding to keep the same shape except channel dimension
        self.padding = int((kernel_size - 1) / 2)

        # input gate (input info)
        self.Wii = nn.Conv2d(self.D_in, self.D_hidden, self.kernel_size, 1, self.padding, bias=bias)
        self.Whi = nn.Conv2d(self.D_hidden, self.D_hidden, self.kernel_size, 1, self.padding, bias=bias)
        # forget gate
        self.Wif = nn.Conv2d(self.D_in, self.D_hidden, self.kernel_size, 1, self.padding, bias=bias)
        self.Whf = nn.Conv2d(self.D_hidden, self.D_hidden, self.kernel_size, 1, self.padding, bias=bias)
        # input gate (cell info)
        self.Wig = nn.Conv2d(self.D_in, self.D_hidden, self.kernel_size, 1, self.padding, bias=bias)
        self.Whg = nn.Conv2d(self.D_hidden, self.D_hidden, self.kernel_size, 1, self.padding, bias=bias)
        # output gate
        self.Wio = nn.Conv2d(self.D_in, self.D_hidden, self.kernel_size, 1, self.padding, bias=bias)
        self.Who = nn.Conv2d(self.D_hidden, self.D_hidden, self.kernel_size, 1, self.padding, bias=bias)

        # save convolution kernels (to-be cudaed)
        self._kernels = [self.Wii, self.Whi, self.Wif, self.Whf, self.Wig, self.Whg, self.Wio, self.Who]

    def forward(self, x, internal_state):
        h, c = internal_state
        # input
        i = torch.sigmoid(self.Wii(x) + self.Whi(h))
        # forget
        f = torch.sigmoid(self.Wif(x) + self.Whf(h))
        # input (internel cell status)
        g = torch.tanh(self.Wig(x) + self.Whg(h))
        # output
        o = torch.sigmoid(self.Wio(x) + self.Who(h))
        # next cell status
        cx = f * c + i * g
        # next hidden/output status
        hx = o * torch.tanh(cx)
        return hx, cx

    def cuda(self, device=None):
        super(ConvLSTMCell, self).cuda(device)
        for k in self._kernels:
            k.cuda()

class CONV_lstm_unit(torch.nn.Module):
    def __init__(self, D_in, D_hidden, kernel_size, k, bias=True):
        """
        initilalize the traditional LSTM module by convLSTMCell(implemented above)
         with TBPTT

        Note1: TBPTT: truncated back propogation through time.The main difference is that
        it only computes the gradients on lastest k time steps.
        Note2: it does not support sequence input like build-in LSTM.
        :param D_in:
        :param D_hidden:
        :param kernel_size:
        :param k:
        """
        super(CONV_lstm_unit, self).__init__()
        self.D_in = D_in
        self.D_hidden = D_hidden
        self.k = k # reserve gradients of how many time steps
        self.LSTM_cell = ConvLSTMCell(D_in, D_hidden, kernel_size, bias)
        self.internal_state = []
        self.is_cuda = False # is on cuda?

    def init_hiddens(self, batch_num, input_shape):
        """
        initialize hidden status for conv LSTM is more challenging then classical one
        since convolution operation should consider the shape of the input
        :param batch_num:
        :param D_hidden: depth for hidden/output tensor
        :param input_shape: (width(int) x height(int))
        :return:
        """
        hx = torch.zeros(batch_num, self.D_hidden, input_shape[0], input_shape[1])
        cx = torch.zeros(batch_num, self.D_hidden, input_shape[0], input_shape[1])
        if self.is_cuda:
            hx = hx.cuda()
            cx = cx.cuda()
        self.internal_state.append([hx, cx, None])

    def forward(self, x):
        """
        For pytorch does not support dynamic compute graph(cannot set require_grad for non-leaf variable).
        It seem to be an overhead that we recompute previous k-steps. However, reduce the redundant requires
        us to modify the backward computing method which requires more advanced modification that will
        be not covered in my implemenetation.
        :param x:
        :return:
        """
        # require to call init_hiddens to intialialize the internal state
        # for the first time step
        if len(self.internal_state) == 0:
            self.init_hiddens(x.shape[0], (x.shape[2], x.shape[3]))
        hx, cx, _ = self.internal_state[0]
        hx = hx.detach()
        cx = cx.detach()
        for hxcx in self.internal_state:
            _, _, x_ = hxcx
            if x_ is None:
                x_ = x.detach()
                hxcx[2] = x
            hx, cx = self.LSTM_cell(x, (hx, cx))

        self.internal_state.append([hx, cx, None])

        if len(hxcx) >= self.k:
            evict_element = self.internal_state.pop(0)
            del(evict_element)

        return hx

    def get_hidden(self):
        for hxcx in self.internal_state:
            hx, cx, x = hxcx
            if self.is_cuda:
                hx.cpu()
                cx.cpu()
                if not x is None:
                    x.cpu()
        return self.internal_state

    def dump_hidden(self, internal_state):
        self.internal_state = internal_state
        for hxcx in self.internal_state:
            hx, cx, x = hxcx
            if self.is_cuda:
                hx.cuda()
                cx.cuda()
                if not x is None:
                    x.cuda()

    def cuda(self, device=None):
        super(CONV_lstm_unit, self).cuda(device)
        self.is_cuda = True
        self.LSTM_cell.cuda()

if __name__ == "__main__":
    # # unit tests
    # # 1. ConvLSTMCell
    # cell = ConvLSTMCell(10, 5, 3)
    #
    # # dim = (batch, chanel, shape)
    # x = torch.randn(5, 10, 10, 10)
    # h = torch.randn(5, 5, 10, 10)
    # c = torch.randn(5, 5, 10, 10)
    #
    # hx, cx = cell(x, h, c)
    # print(hx.size(), cx.size())
    #
    # # 2. ConvLSTMCell (with cuda)
    # x = x.cuda()
    # h = h.cuda()
    # c = c.cuda()
    # cell.cuda()
    # hx, cx = cell(x, h, c)
    # print(hx.size(), cx.size())

    # 3. Conv_lstm_unit
    #    def __init__(self, D_in, D_hidden, kernel_size, k, bias=True):
    foo = CONV_lstm_unit(5, 10, 3, 10)
    # dim = (batch, chanel, shape)
    x = torch.randn(5, 5, 10, 10)
    # foo.init_hiddens(5, (10, 10))
    out = foo(x) ** 2
    print(out.size())
    loss = torch.sum(out)
    loss.backward()

