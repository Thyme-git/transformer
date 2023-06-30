import torch
from torch import nn

__all__ = [
    'PositionalEncoder'
]


class PositionalEncoder(nn.Module):
    '''positional encoder implementation'''
    def __init__(self, maxLength:int, tokenSize:int, dropout = 0.1, **kwargs) -> None:
        '''
        input
            maxLength: max sequence length of a sentence or ...
            tokenSize: token vector's length
        calc
            posInfo(i_seq, 2*j_token) = sin(i_seq / 10000 ** (2*j_token / tokenSize))
            posInfo(i_seq, 2*j_token+1) = cos(i_seq / 10000 ** (2*j_token / tokenSize))
        '''
        super().__init__()
        self.encoding = torch.zeros((1, maxLength, tokenSize)) # first dimension for batch size
        self.encoding.requires_grad_(False)
        
        tmp = torch.arange(0, maxLength).reshape((-1, 1))
        tmp = tmp / torch.pow(10000, torch.arange(0, tokenSize, 2) / tokenSize)
        
        self.encoding[:, :, 0::2] = torch.sin(tmp)
        self.encoding[:, :, 1::2] = torch.cos(tmp)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, X:torch.Tensor):
        '''
        input
            X: (batchSize, seqLength, tokenSize) 
        '''
        return self.dropout(X + self.encoding[:, :X.shape[0], :])

if __name__ == '__main__':
    X = torch.randn((100, 30, 512))
    encoder = PositionalEncoder(30, 512)
    out = encoder(X)
    print(out.shape)