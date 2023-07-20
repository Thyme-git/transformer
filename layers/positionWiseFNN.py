import torch
from torch import nn

__all__ = [
    'PositionWiseFeedForward'
]

class PositionWiseFeedForward(nn.Module):
    '''simple mlp converting token'''
    def __init__(self, tokenSize, hiddenSize, drop = 0.1, **kwargs) -> None:
        '''
        input
            tokenSize: token's length in sequence
                et. seqs: (batchSize, seqlength, tokenSize)
            hiddenSize: 2048 is recommanded
        output
            converted seqs (batchSize, seqlength, tokenSize)
        '''
        super().__init__()
        self.fc0 = nn.Linear(tokenSize, hiddenSize)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop)
        self.fc1 = nn.Linear(hiddenSize, tokenSize)
    
    def forward(self, X):
        '''
        input
            X: (batchSize, seqlength, tokenSize)
        output
            converted X: (batchSize, seqlength, tokenSize)
        '''
        out = self.fc0(X)
        out = self.relu(out)
        out = self.dropout(out)
        return self.fc1(out)


if __name__ == '__main__':
    fnn = PositionWiseFeedForward(512, 2048)
    embeded = torch.randn((100, 320, 512))
    out = fnn(embeded)
    print(out.shape)