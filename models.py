from torch import nn


class IdNet(nn.Module):
    def __init__(self, nfeatures, nhidden, nlabels):
        super(IdNet, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(nfeatures, nhidden),
            nn.ReLU(),
            nn.Linear(nhidden, int(nhidden/2)),
            nn.ReLU(),
            nn.Linear(int(nhidden/2), int(nhidden/4)),
            nn.ReLU(),
            nn.Linear(int(nhidden/4), int(nhidden/8)),
            nn.ReLU(),
            nn.Linear(int(nhidden/8), nlabels)
        )

    def forward(self, x):
        return self.main(x)
