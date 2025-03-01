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

# 5. Multi-Head Attention

class MultiheadAttention(nn.Module):
    '''
    - This class is used to create the multi-head attention layer for the transformer model
    - used to calculate the attention scores between the input embedding vectors
    Args:
        d_model: int: the dimension of the model (default= 512) also known as the embedding size
        num_heads: int: the number of attention heads (default= 4)
        dropout: float: the dropout rate (default= 0)
    Returns:
        multihead_attention: tensor: the multi-head attention layer for the transformer model
    
    '''
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0) -> None:
        super(MultiheadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        # to make sure the d_model is divisible by the number of heads
        assert d_model % self.num_heads == 0
        self.d_k = d_model // self.num_heads # d_k is the dimension of the key and value vectors
        '''
            - d_k is the dimension of the key and value vectors
            - w_q is the weight matrix for the query vectors of shape (d_model, d_model)
            - w_k is the weight matrix for the key vectors of shape (d_model, d_model)
            - w_v is the weight matrix for the value vectors of shape (d_model, d_model)
            - w_o is the weight matrix for the output vectors of shape (d_model, d_model)
            nn.Linear is used to create a linear layer and it applies a linear transformation y = xW^T + b
        '''
        self.w_q = nn.Linear(self.d_model, self.d_model)
        self.w_k = nn.Linear(self.d_model, self.d_model)
        self.w_v = nn.Linear(self.d_model, self.d_model)
        self.w_o = nn.Linear(self.d_model, self.d_model) #d_model = d_k*num_heads and d_k == d_v

        @staticmethod
        def attention_block(query, key, value, mask, dropout: nn.Dropout):
            '''
            shape of the query is (batch_size, num_heads, seq_len, d_k)
            extract the d_k dimension from the query tensor
            '''
            d_k = query.shape[-1]
            attention_score = torch.matmul(query, key.transpose(-2, -1))/math.sqrt(d_k) # calculating the attention scores
            if mask is not None:
                attention_score = attention_score.masked_fill(mask == 0, -1e9) # applying the mask to the attention scores
            attention_score = nn.softmax(attention_score, dim = -1) # applying the softmax function to the attention scores
            if dropout is not None:
                aention_score = dropout(attention_score) # applying the dropout to the attention scores

            attention = torch.matmul(attention_score, value) # calculating the attention
            return attention, attention_score


        def forward(self, q, k, v, mask):
            '''
            self.w_q(q) is used to calculate the query vectors or applies the linear transformation to the query vectors
            query = q @ w_q^T + b_q   
            q: tensor: the query vectors of shape (batch_size, seq_len, d_model)
            w_q: tensor: the weight matrix for the query vectors of shape (d_model, d_model)
            b_q: tensor: the bias for the query vectors of shape (d_model)
            query: tensor: the query vectors after the linear transformation of shape (batch_size, seq_len, d_model)
                
            '''
            query = self.w_q(q) 
            key = self.w_k(k)
            value = self.w_v(v)
            '''
            - splitting the query, key, and value vectors into a number of heads so that we can calculate the attention scores in parallel
            - splitingn the heads allows the model to focus on different aspects of the input sequence
            - view is used to change the shape of the tensor to (batch_size, seq_len, num_heads, d_k)
            - permute is used to change the dimensions of the tensor to (batch_size, num_heads, seq_len, d_k) : basically it's changing the position dimensions
            '''
            query = query.view(q.shape[0], q.shape[1], self.num_heads, self.d_k).permute(0, 2, 1, 3) 
            key = key.view(k.shape[0], k.shape[1], self.num_heads, self.d_k).permute(0, 2, 1, 3)
            value = value.view(v.shape[0], v.shape[1], self.num_heads, self.d_k).permute(0, 2, 1, 3)

            x, self.attention_score = MultiheadAttention.attention_block(query, key, value, mask, self.dropout)

            x = x.permute(0, 2, 1, 3).contiguous().view(q.shape[0], -1, self.d_model) # changing the dimensions of the tensor back to (batch_size, seq_len, d_model)
            x = self.w_o(x)
            return x
        
# 6. Residual Connection

class ResidualConnection(nn.Module):
    '''
        
    '''



            

    