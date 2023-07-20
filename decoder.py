import torch
from torch import nn
from layers.attention import MutiHeadAttention
from layers.positionWiseFNN import PositionWiseFeedForward


__all__ = [
    'DecoderBlock'
]

class DecoderBlock(nn.Module):
    '''simple transformer decoder block implementation'''
    def __init__(self, hiddenSize:int, numHeads:int, fnnHiddenSize:int, dropout=0.1, **kwargs) -> None:
        '''
        input
            hiddenSize: attention output token size, 512 is recommanded
            numHeads: headers' amount for multi-header attention, 8 is recommanded
            dropout: STFW if you don't understand what it is
            fnnHiddenSize: hidden size for 2-layer MLP fnn, 2048 is recommanded
        '''
        super().__init__()
        self.selfAttention = MutiHeadAttention(hiddenSize, hiddenSize, hiddenSize, hiddenSize, numHeads)
        self.layerNorm0 = nn.LayerNorm(hiddenSize)

        self.encDecAttention = MutiHeadAttention(hiddenSize, hiddenSize, hiddenSize, hiddenSize, numHeads)
        self.layerNorm1 = nn.LayerNorm(hiddenSize)

        self.fnn = PositionWiseFeedForward(hiddenSize, fnnHiddenSize, dropout)
        self.layerNorm2 = nn.LayerNorm(hiddenSize)
    
    def forward(self, src, tgt, tgt_mask = None):
        '''
        input
            src: (batchSize, seqLength, hiddenSize) as encoder output
            tgt: (batchSize, seqLength, hiddenSize) as positional encoded target sequence
            tgt_mask: mask for tgt
        '''
        tgt = self.layerNorm0(self.selfAttention(tgt, tgt, tgt, tgt_mask) + tgt)
        out = self.layerNorm1(self.encDecAttention(tgt, src, src) + tgt)
        return self.layerNorm2(self.fnn(out) + out)


if __name__ == '__main__':
    src = torch.randn((100, 30, 512))
    tgt = torch.randn((100, 30, 512))
    decoder = DecoderBlock(512, 8, 2048)
    out = decoder(src, tgt)
    print(out.shape)