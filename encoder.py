import torch
from torch import nn
from layers.attention import MutiHeadAttention
from layers.positionWiseFNN import PositionWiseFeedForward

__all__ = [
    'EncoderBlock'
]


class EncoderBlock(nn.Module):
    '''transformer encoder block implementation'''
    def __init__(self, tokenSize:int, hiddenSize:int, numHeads:int, fnnHiddenSize:int, dropout=0.1, **kwargs) -> None:
        '''
        input
            tokenSize: token vector's length, 512 is recommanded
            hiddenSize: attention output token size, 512 is recommanded
            numHeads: headers' amount for multi-header attention, 8 is recommanded
            dropout: STFW if you don't understand what it is
            fnnHiddenSize: hidden size for 2-layer MLP fnn, 2048 is recommanded
        note
            tokenSize must be the same as hiddenSize to guarantee residual connection going smoothly
        '''
        super().__init__()
        self.selfAttention = MutiHeadAttention(tokenSize, tokenSize, tokenSize, hiddenSize, numHeads)
        self.layerNorm0 = nn.LayerNorm(hiddenSize)
        self.fnn = PositionWiseFeedForward(tokenSize, fnnHiddenSize, dropout)
        self.layerNorm1 = nn.LayerNorm(hiddenSize)
    
    def forward(self, X:torch.Tensor):
        '''
        input:
            X: (batchSize, seqLength, tokenSize) for sentences after postional encoding or from last encoder block
        '''
        out = self.layerNorm0(self.selfAttention(X, X, X) + X)
        return self.layerNorm1(self.fnn(out) + out)


if __name__ == "__main__":
    X = torch.randn((100, 30, 512))
    encoder = EncoderBlock(512, 512, 8, 2048)
    out = encoder(X)
    print(out.shape)