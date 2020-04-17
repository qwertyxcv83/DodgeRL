import torch


class ResNet(torch.nn.Module):
    def __init__(self, layers):
        super(ResNet, self).__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class DenseNet(torch.nn.Module):

    def __init__(self, layers):
        super(DenseNet, self).__init__()
        self.depth = len(layers)
        self.layers = torch.nn.ModuleList(layers)
        self.pars = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.FloatTensor().new_full((i+1, ), 1/float(i+1)), requires_grad=True)
             for i in range(self.depth + 1)])

    def forward(self, inp):
        # inp: N*...*n
        shape = inp.shape
        acts = [inp.reshape(shape + torch.Size([1]))]

        for i, layer in enumerate(self.layers):
            # pars: size x
            layer_acts = acts[:i+1]  # size x*...
            # multiplication: (x) * (...*x)

            a = self.pars[i] * torch.cat(acts, dim=len(shape))
            layer_in = a.sum(dim=len(shape))
            acts.append(layer(layer_in).reshape(shape + torch.Size([1])))

        out = (self.pars[self.depth] * torch.cat(acts, dim=len(shape))).sum(dim=len(shape))

        return out


# probably better than its alternatives, n_softmax = 2 or 3 is enough
class WeightedAverageLayer(torch.nn.Module):
    def __init__(self, n_in, n_out, n_softmax):
        super(WeightedAverageLayer, self).__init__()
        self.n_out = n_out
        self.n_softmax = n_softmax
        self.lin = torch.nn.Linear(n_in, n_out * n_softmax)
        self.lin_soft = torch.nn.Linear(n_in, n_out * n_softmax)

    def forward(self, inp):
        shape = inp.shape[:-1]
        lin = self.lin(inp).reshape(shape + torch.Size([self.n_out, self.n_softmax]))
        soft = self.lin_soft(inp).reshape(shape + torch.Size([self.n_out, self.n_softmax])).softmax(dim=len(shape)+1)

        out = (lin * soft).sum(dim=len(shape)+1)

        return out


class SelfWeightedAverageLayer(torch.nn.Module):
    def __init__(self, n_in, n_out, n_softmax):
        super(SelfWeightedAverageLayer, self).__init__()
        self.n_out = n_out
        self.n_softmax = n_softmax
        self.lin = torch.nn.Linear(n_in, n_out * n_softmax)

    def forward(self, inp):
        shape = inp.shape[:-1]
        lin = self.lin(inp).reshape(shape + torch.Size([self.n_out, self.n_softmax]))
        soft = lin.softmax(dim=len(shape)+1)

        out = (lin * soft).sum(dim=len(shape)+1)

        return out


class MaxLayer(torch.nn.Module):
    def __init__(self, n_in, n_out, n_max):
        super(MaxLayer, self).__init__()
        self.n_out = n_out
        self.n_max = n_max
        self.lin = torch.nn.Linear(n_in, n_out * n_max)

    def forward(self, inp):
        shape = inp.shape[:-1]
        lin = self.lin(inp).reshape(shape + torch.Size([self.n_out, self.n_max]))
        out, _ = lin.max(dim=len(shape)+1)

        return out


class FullyConnected(torch.nn.Module):
    def __init__(self, sizes):
        super(FullyConnected, self).__init__()

        layer_list = []
        for i in range(len(sizes)-1):
            layer_list.append(torch.nn.Linear(sizes[i], sizes[i+1]))

        self.layers = torch.nn.ModuleList(layer_list)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x).relu()
        return x
