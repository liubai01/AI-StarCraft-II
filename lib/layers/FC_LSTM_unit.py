"""
Refer to the idea of the post:
https://discuss.pytorch.org/t/implementing-truncated-backpropagation-through-time/15500/4
"""
import torch

class FC_lstm_unit(torch.nn.Module):
    def __init__(self, D_in, D_hidden, k):
        """
        initilalize the traditional LSTM module by built-in LSTMCell with TBPTT

        Note1: TBPTT: truncated back propogation through time.The main difference is that
        it only computes the gradients on lastest k time steps.
        Note2: it does not support sequence input like build-in LSTM.
        Note3: called FC here since I also aim at implement the conv LSTM version with TBPTT.
        :param D_in:
        :param D_hidden:
        :param k:
        """
        super(FC_lstm_unit, self).__init__()
        self.D_in = D_in
        self.D_hidden = D_hidden
        self.k = k # reserve gradients of how many time steps
        self.LSTM_cell = torch.nn.LSTMCell(D_in, D_hidden)
        self.internal_state = []
        self.is_cuda = False # is on cuda?

    def init_hiddens(self, batch_num):
        hx = torch.zeros(batch_num, self.D_hidden)
        cx = torch.zeros(batch_num, self.D_hidden)
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
            self.init_hiddens(x.size()[0])
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

    def cuda(self, device=None):
        super(FC_lstm_unit, self).cuda(device)
        self.is_cuda = True
        self.LSTM_cell.cuda()

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


class FC_lstm_stacked(torch.nn.Module):
    def __init__(self, arch, k):
        super(FC_lstm_stacked, self).__init__()
        self.arch = arch
        self.k = k
        self.ceils = []

        assert(isinstance(arch, list))
        assert(len(arch) > 0)
        prev = None
        for a in arch:
            if prev is None:
                prev = a
                continue
            self.ceils.append(FC_lstm_unit(prev, a, k))
            prev = a
        if len(arch) == 1:
            self.ceils.append(FC_lstm_unit(prev, prev, k))

    def init_hiddens(self, batch_num):
       for c in self.ceils:
            c.init_hiddens(batch_num)

    def forward(self, x):
        tmp = x
        for c in self.ceils:
            tmp = c(tmp)
        return tmp

    def cuda(self, device=None):
        super(FC_lstm_stacked, self).cuda(device)
        for c in self.ceils:
            c.cuda()

    def get_hidden(self):
        ret_hidden = []
        for c in self.ceils:
            ret_hidden.append(c.get_hidden())
        return ret_hidden


    def dump_hidden(self, internal_states):
        for i, c in enumerate(self.ceils):
            c.dump_hidden(internal_states[i])

if __name__ == "__main__":
    foo = FC_lstm_stacked([10, 20, 10], 10)
    foo.cuda()
    # foo.init_hiddens(2)
    print(foo.ceils)
    for i in range(10):
        x = torch.randn(2, 10).cuda()
        out = foo(x) ** 2
        loss = torch.sum(out)
        loss.backward()
        # print(foo.ceils[0].LSTM_cell.bias_hh.grad)

