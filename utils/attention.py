import torch
from torch import nn

__all__ = [
    'DotProductAttention'
,   'AdditiveAttention'
]

class Attention(nn.Module):
    '''base model for attention'''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.maskValue = -1e10

    def forward(self, *args, **kwargs):
        '''should be implemented in child class'''
        raise NotImplementedError

    def _shapeCheck(self, inputTensor:torch.Tensor, name:str, dim:int):
        '''check if inputTensor's dimension is equal to dim'''
        try:
            assert(len(inputTensor.shape) == dim)
        except AssertionError:
            raise TypeError(f'{name} must be a {dim}D tensor but got shape of {inputTensor.shape}')

    def _equalCheck(self, val0:any, name0:str, val1:any, name1:str):
        '''check if val0 is equal to val1'''
        if isinstance(val0, torch.Tensor):
            try:
                assert(torch.equal(val0, val1))
            except AssertionError:
                raise ValueError(f'{name0} should be the same as {name1}, but got {val0} and {val1}')
        else:
            try:
                assert(val0 == val1)
            except AssertionError:
                raise ValueError(f'{name0} should be the same as {name1}, but got {val0} and {val1}')
        
    def _mask(self, score, mask):
        '''
            mask is a 0-1 tensor with the same shape of score
            set score[where mask = 0] to self.maskValue(-1e10)
        '''
        self._equalCheck(score.shape, 'shape of score', mask.shape, 'shape of mask')
        return score.masked_fill(mask == 0, self.maskValue)


class DotProductAttention(Attention):
    '''dot product attention implementation'''
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(args, kwargs)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, queries:torch.Tensor, keys:torch.Tensor, values:torch.Tensor, mask = None):
        '''
        input
            queries: (batchSize, n, dk) note that dimension of query must be same as key
            keys: (batchSize, m, dk)
            values: (batchSize, m, dv)
            mask: (batchSize, n, m)
        return
            attention output: (batchSize, n, dv) for n queries
        '''

        self._shapeCheck(queries, 'queries', dim=3)
        self._shapeCheck(keys, 'keys', dim=3)
        self._shapeCheck(values, 'values', dim=3)

        self._equalCheck(queries.shape[2], 'query\'s length', keys.shape[2], 'key\'s length')
        self._equalCheck(keys.shape[1], 'amount of keys', values.shape[1], 'amount of values')

        attentionScore = torch.bmm(queries, keys.transpose(1, 2)) / keys.shape[2]**0.5 # (batchSize, n, m)

        if mask is not None:
            attentionScore = self._mask(attentionScore, mask)

        attentionWeight = self.softmax(attentionScore)
        return torch.bmm(attentionWeight, values)


class AdditiveAttention(Attention):
    '''additive attention implementation'''
    def __init__(self, querySize:int, keySize:int, numHidden:int, **kwargs) -> None:
        super().__init__(querySize, keySize, numHidden, **kwargs)
        self.queryTransform = nn.Linear(querySize, numHidden) # transform query dim from querySize to numHidden
        self.keyTransform = nn.Linear(keySize, numHidden) # transform key dim from keySize to numHidden
        self.tanh = nn.Tanh()
        self.valueTransform = nn.Linear(numHidden, 1) # transform key-query coincidence to scalar for attention score
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries:torch.Tensor, keys:torch.Tensor, values:torch.Tensor, mask = None):
        '''
        input
            queries: (batchSize, n, dq)
            keys: (batchSize, m, dk)
            values: (batchSize, m, dv)
        return
            attention output: (batchSize, n, dv) for n queries
        '''

        self._shapeCheck(queries, 'queries', dim=3)
        self._shapeCheck(keys, 'keys', dim=3)
        self._shapeCheck(values, 'values', dim=3)

        self._equalCheck(keys.shape[1], 'amount of keys', values.shape[1], 'amount of values')

        transformedQueries = self.queryTransform(queries) # (batchSize, n, h)
        transformedKeys = self.keyTransform(keys) # (batchSize, m, h)

        transformedQueries = transformedQueries.unsqueeze(2)  # (batchSize, n, 1, h)
        transformedKeys = transformedKeys.unsqueeze(1) # (batchSize, 1, m, h)
        
        coincidence = self.tanh(transformedKeys+transformedQueries) # (batchSize, n, m, h)
        
        attentionScore = self.valueTransform(coincidence) # (batchSize, n, m, 1)
        attentionScore = attentionScore.squeeze(-1) # (batchSize, n, m)
        
        if mask is not None:
            attentionScore = self._mask(attentionScore, mask)
        
        attentionWeight = self.softmax(attentionScore) # (batchSize, n, m)
        return torch.bmm(attentionWeight, values)


class MutiHeadAttention(Attention):
    '''muti-head attention  implementation with dot product attention'''
    def __init__(self, querySize:int, keySize:int, valueSize:int, numHidden:int, numHeads:int, **kwargs) -> None:
        super().__init__(querySize, keySize, valueSize, numHidden, numHeads, **kwargs)
        self.numHeads = numHeads
        self.attention = DotProductAttention(querySize, keySize, valueSize, numHidden, **kwargs)
        self.queryTransform = nn.Linear(querySize, numHidden) # transform query dim from querySize to numHidden
        self.keyTransform = nn.Linear(keySize, numHidden) # transform key dim from keySize to numHidden
        self.valueTransform = nn.Linear(valueSize, numHidden) # transform value dim from valueSize to numHidden
        self.outputTransform = nn.Linear(numHidden, numHidden)

    def forward(self, queries:torch.Tensor, keys:torch.Tensor, values:torch.Tensor, mask = None):
        '''
        input
            queries: (batchSize, n, dq)
            keys: (batchSize, m, dk)
            values: (batchSize, m, dv)
            mask: (batchSize, n, m)
        return
            attention output: (batchSize, n, numHidden) for n queries
        '''
        self._shapeCheck(queries, 'queries', dim=3)
        self._shapeCheck(keys, 'keys', dim=3)
        self._shapeCheck(values, 'values', dim=3)

        self._equalCheck(keys.shape[1], 'amount of keys', values.shape[1], 'amount of values')

        transformedQueries = self.queryTransform(queries) # (batchSize, n, hidden)
        transformedKeys = self.keyTransform(keys) # (batchSize, m, hidden)
        transformedValues = self.valueTransform(values) # (batchSize, m, hidden)

        transformedQueries = self.split(transformedQueries) # (b*numHeads, n, hidden/numHeads)
        transformedKeys = self.split(transformedKeys) # (b*numHeads, m, hidden/numHeads)
        transformedValues = self.split(transformedValues) # (b*numHeads, m, hidden/numHeads)

        if mask is not None:
            # On axis 0, copy the first item (scalar or vector) for num_heads
            # times, then copy the next item, and so on
            mask = torch.repeat_interleave(mask, repeats=self.numHeads, dim=0)

        out = self.attention(transformedQueries, transformedKeys, transformedValues, mask)
        out = self.concat(out) # (b, n, hidden)
        return self.outputTransform(out) # (b, n, hidden)

    def split(self, X:torch.Tensor):
        '''
        input
            X: (b, n, d)
        return
            out: (b*numHeads, n, d/numHeads)
        '''
        try:
            assert(X.shape[2] % self.numHeads == 0)
        except AssertionError:
            raise ValueError(f'length of vector should be no. of heads\' multiple, but got {X.shape[2]} % numHeads != 0')
        
        X = X.reshape(X.shape[0], X.shape[1], self.numHeads, -1) # (b, n, numHeads, d/numHeads)
        X = X.transpose(1, 2) # (b, numHeads, n, d/numHeads)
        return X.reshape(X.shape[0]*X.shape[1], X.shape[2], X.shape[3])
    
    def concat(self, X:torch.Tensor):
        '''
        input
            X: (b*numHeads, n, d/numHeads)
        return
            out: (b, n, d)
        '''
        X = X.reshape(-1, self.numHeads, X.shape[1], X.shape[2])
        X = X.transpose(1, 2)
        return X.reshape(X.shape[0], X.shape[1], -1)


if __name__ == '__main__':
    attension = MutiHeadAttention(128, 512, 256, 128, 8)
    
    queries = torch.randn((10, 100, 128))
    keys = torch.randn((10, 1024, 512))
    values = torch.randn((10, 1024, 256))

    output = attension(queries, keys, values, torch.randint(0, 1, (queries.shape[0], queries.shape[1], keys.shape[1])))
    print(output.shape)