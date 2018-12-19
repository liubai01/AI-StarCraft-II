import torch

class Replicate_unit0d(torch.nn.Module):
    def __init__(self, depth, width, height):
        super(Replicate_unit0d, self).__init__()
        self.depth = depth
        self.width = width
        self.height = height

    def forward(self, x):
        # accept only scalar, 1 dimensional input, first dimension is batch
        assert(len(x.size()) == 1)
        batch_num = x.size()[0]
        tmp1 = torch.cat([x.view((batch_num, 1,)) for _ in range(self.depth)], dim=1)
        tmp2 = torch.cat([tmp1.view((batch_num, -1, 1)) for _ in range(self.width)], dim=2)
        ret = torch.cat([tmp2.view((batch_num, tmp2.size()[1], tmp2.size()[2], 1)) for _ in range(self.height)], dim=3)
        return ret

class Replicate_unit1d(torch.nn.Module):
    def __init__(self, width, height):
        super(Replicate_unit1d, self).__init__()
        self.width = width
        self.height = height

    def forward(self, x):
        # accept only scalar, 0 dimensional input
        assert(len(x.size()) == 2)
        batch_num = x.size()[0]
        tmp = torch.cat([x.view((batch_num, -1, 1)) for _ in range(self.width)], dim=2)
        ret = torch.cat([tmp.view((batch_num, tmp.size()[1], tmp.size()[2], 1)) for _ in range(self.height)], dim=3)
        return ret

if __name__ == "__main__":
    x = torch.randn((4, 1), requires_grad=True)
    ru = Replicate_unit0d(4, 10, 10)
    print(ru(x[:, 0]).size())

    x = torch.randn((4, 4,))
    ru = Replicate_unit1d(10, 10)
    print(ru(x).size())
