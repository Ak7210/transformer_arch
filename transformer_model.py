'''
working on the architecture of the transformer model:
- 1. Input Embedding
- 2. positional encoding
- 3. feed forward network
- 4. add and norm {layer normalization}

- 5. multi-head attention
- 6. masked multi-head attention
- 7. output embedding

- transformer encoder
- transformer decoder
'''
# Importing Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np  
import math
# 1. Input Embedding

class input_embedding(nn.Module):

    ''' This class is used to create the input embedding layer for the transformer model which is used to convert the input tokens into the embedding vectors of the same size as the model dimension 
    Args:
        d_model: int: the dimension of the model (default= 512) also known as the embedding size
        vocab_size: int: the size of the vocabulary (depends on the dataset)
        padding_idx: int: the padding index for the padding token (default= 0)

    Returns:
        input_embedding: tensor: the input embedding layer for the transformer model

    Note: currently we are not using the padding index, but we can use it in the future for that we need to pass the padding index as an argument
    ''''
    def __init__(self, d_model: int, vocab_size : int, padding_idx: int = 0):
        super(input_embedding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size    
        self.embedding = nn.Embedding(self.vocab_size, self.d_model, padding_idx= padding_idx) # creating the embedding layer using the nn.Embedding class 
        '''
        nn.Embedding: A simple lookup table that stores embeddings of a fixed dictionary and size.
        This module is often used to store word embeddings and retrieve them using indices. The input to the module is a list of indices, and the output is the corresponding word embeddings.
        '''

    def forward(self, x):
        return self.embedding(x)*math.sqrt(self.d_model)
        

        
