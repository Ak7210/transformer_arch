'''
working on the architecture of the transformer model:
- 1. Input Embedding
- 2. positional encoding
- 3. Layer Normalization
- 4. Feed Forward Network

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

    ''' This class is used to create the input embedding layer for the transformer model
         which is used to convert the input tokens into the embedding vectors of the same size as the model dimension 
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
        '''
        This function is used to pass the input tokens through the embedding layer to get the embedding vectors
        Args:
            x: tensor: the input tokens of shape (batch_size, seq_len)
            batch_size: int: the size of the batch
            seq_len: int: the length of each input sequence
        Returns:
            embedding: tensor: the embedding vectors of shape (batch_size, seq_len, d_model)
            multiplying the embedding vectors with the square root of the d_model is prevent the gradients from vanishing or exploding
        '''
        return self.embedding(x)*math.sqrt(self.d_model) 

# 2. Positional Encoding

class positional_encoding(nn.Module):
    
    ''' Transformers process input tokens in parallel and lack order information. 
        To address this, a positional encoding layer adds positional information using a positional encoding matrix, ensuring the model understands token order.
    Args:
        d_model: int: the dimension of the model (default= 512) also known as the embedding size
        max_len: int: the maximum length of the input sequence (default= 512)

    Returns:
        positional_encoding: tensor: the positional encoding layer for the transformer model

    Note: the positional encoding is added to the input embedding vectors to add the positional information to the input tokens
    ''''
    def __init__(self, d_model: int, max_len: int = 512, dropout: float) -> None:
        super(positional_encoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout) # adding the dropout layer to the positional encoding to prevent overfitting
        self.positional_encoding = self.get_positional_encoding() # getting the positional encoding matrix



    def get_positional_encoding(self):
        '''
        This function is used to create the positional encoding matrix
        Returns:
            positional_encoding: tensor: the positional encoding matrix of shape (max_len, d_model)
        '''
        positional_encoding = torch.zeros(self.max_len, self.d_model) # creating a matrix of zeros of shape (max_len, d_model)

        position = torch.arange(0, self.max_len, dtype= torch.float).unsqueeze(1) # creating a position matrix of shape (max_len, 1) 
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float()*(-math.log(10000.0)/self.d_model)) # creating a division term

        positional_encoding[:, 0::2] = np.sin(position*div_term) # adding the sin values to the even indices
        positional_encoding[:, 1::2] = np.cos(position*div_term) # adding the cos values to the odd indices

        positional_encoding = positional_encoding.unsqueeze(0)
        return positional_encoding

    def forward(self, x):
        '''
        This function is used to add the positional encoding to the input embedding vectors
        Args:
            x: tensor: the input embedding vectors of shape (batch_size, seq_len, d_model)
            batch_size: int: the size of the batch
            seq_len: int: the length of each input sequence
        Returns:
            x + positional_encoding: tensor: the input embedding vectors with the positional encoding added of shape (batch_size, seq_len, d_model)
        '''
        x = x + self.positional_encoding[:, :x.shape[1], :].requires_grad_(False) #requires_grad_(False) is used to prevent the positional encoding matrix from being updated during the training
        x = self.dropout(x)
        return x

# 3. Layer Normalization

class LayerNormalization(nn.Module):
    ''' This class is used to create the layer normalization layer for the transformer model
    Args:
        eps: float: a value added to the denominator for numerical stability (default= 1e-6)
            so that the layer normalization does not divide by zero

    Returns:
        layer_norm: tensor: the layer normalization layer for the transformer model
    '''
    def __init__(self,  eps: float = 1e-6) -> None:
        super(LayerNormalization, self).__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # creating a learnable parameter alpha: multiplicative
        self.bias = nn.Parameter(torch.zeros(1)) # creating a learnable parameter bias: additive

    def forward(self, x):
        '''
        This function is used to normalize the input embedding vectors
        Args:
            X: tensor: the input embedding vectors of shape (batch_size, seq_len, d_model)
        returns:
            layer_norm: tensor: normalized vectors (x*alpha + bias) of shape (batch_size, seq_len, d_model)

        '''
        mean = x.mean(dim = -1, keepdim=True) # dim = -1 is used to calculate the mean along the last dimension
        std = x.std(dim = -1, keepdim=True)   
        layer_norm = self.alpha*(x - mean)/(std + self.eps) + self.bias
        return layer_norm

# 4. Feed Forward Network

class FFN(nn.module):
    '''
    This class is used to create the feed forward network for the transformer model
    used the ReLU activation function and two linear layers
    Args:
        d_model: int: the dimension of the model (default= 512) also known as the embedding size
        d_ff: int: the dimension of the feed forward network (default= 2048)
        dropout: float: the dropout rate (default= 0)
            
    Returns:
        ffn: tensor: the feed forward network for the transformer model
        
    '''
    def __init__(self, d_model: int , d_ff : int = 2048, dropout: float = 0) -> None:
        super(FFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.d_model, self.d_ff) # creating the first fully connected linear layer 
        self.fc2 = nn.Linear(self.d_ff, self.d_model) # creating the second fully connected linear layer

    def forward(self, x):
        '''
        This function is used to pass the input embedding vectors through the feed forward network
        Args:
            x: tensor: the input embedding vectors of shape (batch_size, seq_len, d_model)
            batch_size: int: the size of the batch
            seq_len: int: the length of each input sequence
            d_model: int: the dimension of the model (default= 512) also known as the embedding size
        Returns:
            ffn: tensor: the output of the feed forward network of shape (batch_size, seq_len, d_model)
        '''
        x = nn.functional.relu(self.fc1(x)) # passing the input through the first linear layer and the ReLU activation function
        x = self.dropout(x) # applying the dropout
        ffn = self.fc2(x)
        return ffn




        

        
