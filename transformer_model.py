class InputEmbedding(nn.Module):

    ''' This class is used to create the input embedding layer for the transformer model
         which is used to convert the input tokens into the embedding vectors of the same size as the model dimension 
    Args:
        d_model: int: the dimension of the model (default= 512) also known as the embedding size
        vocab_size: int: the size of the vocabulary (depends on the dataset)
        padding_idx: int: the padding index for the padding token (default= 0)

    Returns:
        input_embedding: tensor: the input embedding layer for the transformer model

    Note: currently we are not using the padding index, but we can use it in the future for that we need to pass the padding index as an argument
    '''
    def __init__(self, d_model: int, vocab_size : int):
        super(InputEmbedding, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size    
        self.embedding = nn.Embedding(vocab_size, d_model) # creating the embedding layer using the nn.Embedding class 
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

class PositionalEmbedding(nn.Module):
    
    ''' Transformers process input tokens in parallel and lack order information. 
        To address this, a positional encoding layer adds positional information using a positional encoding matrix, ensuring the model understands token order.
    Args:
        d_model: int: the dimension of the model (default= 512) also known as the embedding size
        max_len: int: the maximum length of the input sequence (default= 512)

    Returns:
        positional_encoding: tensor: the positional encoding layer for the transformer model

    Note: the positional encoding is added to the input embedding vectors to add the positional information to the input tokens
    '''
    def __init__(self, d_model: int, max_seq_len: int, dropout: float):
        super(PositionalEmbedding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(dropout) # adding the dropout layer to the positional encoding to prevent overfitting
        # self.positional_encoding = self.get_positional_encoding() # getting the positional encoding matrix

        pos_emb = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()*(-math.log(10000.0)/d_model))
        pos_emb[:, 0::2] = torch.sin(position*div_term)
        pos_emb[:, 1::2] = torch.cos(position*div_term)

        self.register_buffer('pos_emb', pos_emb.unsqueeze(0))

    # def get_positional_encoding(self):
    #     '''
    #     This function is used to create the positional encoding matrix
    #     Returns:
    #         positional_encoding: tensor: the positional encoding matrix of shape (max_len, d_model)
    #     '''
    #     positional_encoding = torch.zeros(self.max_len, self.d_model) # creating a matrix of zeros of shape (max_len, d_model)

    #     position = torch.arange(0, self.max_len, dtype= torch.float).unsqueeze(1) # creating a position matrix of shape (max_len, 1) 
    #     div_term = torch.exp(torch.arange(0, self.d_model, 2).float()*(-math.log(10000.0)/self.d_model)) # creating a division term

    #     positional_encoding[:, 0::2] = np.sin(position*div_term) # adding the sin values to the even indices
    #     positional_encoding[:, 1::2] = np.cos(position*div_term) # adding the cos values to the odd indices

    #     positional_encoding = positional_encoding.unsqueeze(0)
    #     return positional_encoding

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
        x = x + self.pos_emb[:, :x.size(1), :].requires_grad_(False) #requires_grad_(False) is used to prevent the positional encoding matrix from being updated during the training
        x = self.dropout(x)
        return x

# 3. Layer Normalization

# class LayerNormalization(nn.Module):
#     ''' This class is used to create the layer normalization layer for the transformer model
#     Args:
#         eps: float: a value added to the denominator for numerical stability (default= 1e-6)
#             so that the layer normalization does not divide by zero

#     Returns:
#         layer_norm: tensor: the layer normalization layer for the transformer model
#     '''
#     def __init__(self,  eps: float = 1e-6) -> None:
#         super(LayerNormalization, self).__init__()
#         self.eps = eps
#         self.alpha = nn.Parameter(torch.ones(1)) # creating a learnable parameter alpha: multiplicative
#         self.bias = nn.Parameter(torch.zeros(1)) # creating a learnable parameter bias: additive

#     def forward(self, x):
#         '''
#         This function is used to normalize the input embedding vectors
#         Args:
#             X: tensor: the input embedding vectors of shape (batch_size, seq_len, d_model)
#         returns:
#             layer_norm: tensor: normalized vectors (x*alpha + bias) of shape (batch_size, seq_len, d_model)

#         '''
#         mean = x.mean(dim = -1, keepdim=True) # dim = -1 is used to calculate the mean along the last dimension
#         std = x.std(dim = -1, keepdim=True)   
#         layer_norm = self.alpha*(x - mean)/(std + self.eps) + self.bias
#         return layer_norm

# 4. Feed Forward Network

class FFN(nn.Module):
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
    def __init__(self, d_model: int , d_ff : int, dropout: float):
        super(FFN, self).__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.d_model, self.d_ff) # creating the first fully connected linear layer 
        self.fc2 = nn.Linear(self.d_ff, self.d_model) # creating the second fully connected linear layer
        self.relu = nn.ReLU() # creating the ReLU activation function

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
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x



# 5. Multi-Head Attention

class MultiHeadAttention(nn.Module):
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
    def __init__(self, d_model: int, num_heads: int, dropout: float):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        # to make sure the d_model is divisible by the number of heads
        assert d_model % num_heads == 0

        self.d_k = d_model // self.num_heads # d_k is the dimension of the key and value vectors
        '''
            - d_k is the dimension of the key and value vectors
            - w_q is the weight matrix for the query vectors of shape (d_model, d_model)
            - w_k is the weight matrix for the key vectors of shape (d_model, d_model)
            - w_v is the weight matrix for the value vectors of shape (d_model, d_model)
            - w_o is the weight matrix for the output vectors of shape (d_model, d_model)
            nn.Linear is used to create a linear layer and it applies a linear transformation y = xW^T + b
        '''
        self.w_q = nn.Linear(self.d_model, self.d_model, bias = False)
        self.w_k = nn.Linear(self.d_model, self.d_model, bias = False)
        self.w_v = nn.Linear(self.d_model, self.d_model, bias = False)
        self.w_o = nn.Linear(self.d_model, self.d_model, bias= False) #d_model = d_k*num_heads and d_k == d_v

    @staticmethod
    def self_attention(q, k, v, mask, dropout: nn.Dropout):
        '''
        shape of the query is (batch_size, num_heads, seq_len, d_k)
        extract the d_k dimension from the query tensor
        '''
        d_k = q.size(-1)

        # calculating the attention scores
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k) # calculating the dot product of the query and key vectors
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -1e9)
        attn_score = attn_score.softmax(dim = -1)

        if dropout is not None:
            attn_score = dropout(attn_score)

        output = torch.matmul(attn_score, v) # calculating the dot product of the attention scores and value vectors
        return output, attn_score


    def forward(self, q, k, v, mask):
        '''
        self.w_q(q) is used to calculate the query vectors or applies the linear transformation to the query vectors
        query = q @ w_q^T + b_q   
        q: tensor: the query vectors of shape (batch_size, seq_len, d_model)
        w_q: tensor: the weight matrix for the query vectors of shape (d_model, d_model)
        b_q: tensor: the bias for the query vectors of shape (d_model)
        query: tensor: the query vectors after the linear transformation of shape (batch_size, seq_len, d_model)
            
        '''
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)

        #  split the d_model into num_heads
        batch_size = Q.size(0)
        max_seq_len = Q.size(1)
        

        '''
        - splitting the query, key, and value vectors into a number of heads so that we can calculate the attention scores in parallel
        - splitingn the heads allows the model to focus on different aspects of the input sequence
        - view is used to change the shape of the tensor to (batch_size, seq_len, num_heads, d_k)
        - permute is used to change the dimensions of the tensor to (batch_size, num_heads, seq_len, d_k) : basically it's changing the position dimensions
        '''
        Q = Q.view(batch_size, max_seq_len, self.num_heads, self.d_k).transpose(1,2)
        K = K.view(batch_size, max_seq_len, self.num_heads, self.d_k).transpose(1,2)
        V = V.view(batch_size, max_seq_len, self.num_heads, self.d_k).transpose(1,2) 

        attn_output, attnt_score = MultiHeadAttention.self_attention(Q, K, V, mask, self.dropout)

        # concatenating the heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, max_seq_len, self.d_model)
        # apply the final linear layer
        attn_output = self.w_o(attn_output)
        return attn_output

       
        
# 6. Residual Connection

class ResidualConnection(nn.Module):
    '''
        It's also know as skip connection
        This class is used to create the residual connection for the transformer model

        Args:
            d_model: int: the dimension of the model (default= 512) also known as the embedding size
            dropout: float: the dropout rate (default= 0)
            sublayer: nn.Module: the sublayers (Multihead self-attention layer or ffn) to be added to the residual connection

        Returns:
            residual_connection: tensor: the residual connection for the transformer model
    '''
    def __init__(self, features: int, dropout: float) -> None:
        super(ResidualConnection, self).__init__()
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(features)

    def forward(self, x, sublayer):
        '''
        This function is used to add the sublayer to the residual connection
        Args:
            x: tensor: the input embedding vectors of shape (batch_size, seq_len, d_model)
            sublayer: nn.Module: the sublayers (Multihead self-attention layer or ffn) to be added to the residual connection
        Returns:
            residual_connection: tensor: the output of the residual connection of shape (batch_size, seq_len, d_model)
        '''
        x = self.norm(x)
        x = x + self.dropout(sublayer(x))
        return x

# Encoder Block
class EncoderBlock(nn.Module):
    '''
    Encoder block consists of:
    - Multihead self-attention layer
    - Residual connection
    - Feed forward network
    - Residual connection
    Args:
        self_attention: nn.Module: the multihead self-attention layer
        ffn: nn.Module: the feed forward network
        dropout: float: the dropout rate (default= 0)
    Returns:
        encoder_block: tensor: the encoder block for the transformer model
    '''        
    def __init__(self, features: int, self_attn: MultiHeadAttention,feed_forward: FFN, dropout: float):
        super(EncoderBlock, self).__init__()
        self.features = features
        self.self_attention = self_attn
        self.feed_forward_network = feed_forward
        self.residual_connection_1 = ResidualConnection(features, dropout)
        self.residual_connection_2 = ResidualConnection(features, dropout)



    def forward(self, x, mask):
        '''
        args:
            x: tensor: the input embedding vectors of shape (batch_size, seq_len, d_model)
            mask: tensor: the mask to be applied to the input embedding vectors of shape (batch_size, seq_len, seq_len)
        Returns:
            encoder_block: tensor: the output of the encoder block of shape (batch_size, seq_len, d_model)
        '''
        x = self.residual_connection_1(x, lambda x: self.self_attention(x, x, x, mask))
        x = self.residual_connection_2(x, self.feed_forward_network)
        return x


# Encoder consists of N encoder blocks 
class Encoder(nn.Module):
    '''
    Encoder consists of N encoder blocks
    Args:
        layers or encoder_blocks: nn.ModuleList: the list of encoder blocks
    Returns:
        encoder: tensor: the encoder for the transformer model
    '''
    def __init__(self, features: int, layers: nn.ModuleList) -> None:
        super(Encoder, self).__init__()
        self.features = features
        self.layers = layers
        self.norm = nn.LayerNorm(features)

    def forward(self, x, source_mask):
        '''
        args:
            x: tensor: the input embedding vectors of shape (batch_size, seq_len, d_model)
            mask: tensor: the mask to be applied to the input embedding vectors of shape (batch_size, seq_len, seq_len)
        Returns:
            encoder: tensor: the output of the encoder of shape (batch_size, seq_len, d_model)
        '''
        for encoder_block in self.layers: # iterating through the encoder blocks (layers)
            x = encoder_block(x, source_mask)
        return self.norm(x)

# Decoder Block
class DecoderBlock(nn.Module):
    '''
    Decoder block consists of:
    - Multihead self-attention layer
    - Residual connection
    - Multihead masked self-attention layer
    - Residual connection
    - Feed forward network
    - Residual connection
    Args:
        self_attention: MultiheadAttention: the multihead self-attention layer
        cross_attention: MultiheadAttention: the multihead masked self-attention layer
        ffn: nn.Module: the feed forward network
        dropout: float: the dropout rate (default= 0)
    Returns:
        decoder_block: tensor: the decoder block for the transformer model
    '''
    def __init__(self, features: int, self_attn: MultiHeadAttention, cross_attn: MultiHeadAttention, feed_forward: FFN, dropout: float):
        super(DecoderBlock, self).__init__()
        self.features = features
        self.self_attn = self_attn
        self.cross_attn = cross_attn
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)

        self.residual_connection1 = ResidualConnection(features, dropout)
        self.residual_connection2 = ResidualConnection(features, dropout)
        self.residual_connection3 = ResidualConnection(features, dropout)
 


    def forward(self, x, enc_out, src_mask, tgt_mask):
        '''
        args:
            x: tensor: the input embedding vectors of shape (batch_size, seq_len, d_model)
            encoder_output: tensor: the output of the encoder of shape (batch_size, seq_len, d_model)
            src_mask: tensor: the mask to be applied to the input embedding vectors of shape (batch_size, seq_len, seq_len)
            tgt_mask: tensor: the mask to be applied to the input embedding vectors of shape (batch_size, seq_len, seq_len)
        Returns:
            decoder_block: tensor: the output of the decoder block of shape (batch_size, seq_len, d_model)
        '''
        x = self.residual_connection1(x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.residual_connection2(x, lambda x: self.cross_attn(x, enc_out, enc_out, src_mask))
        x = self.residual_connection3(x, self.feed_forward)
        return x


# Decoder consists of N decoder blocks
class Decoder(nn.Module):
    '''
    Decoder consists of N decoder blocks
    Args:
        layers or decoder_blocks: nn.ModuleList: the list of decoder blocks
    Returns:
        decoder: tensor: the decoder for the transformer model
    '''
    def __init__(self, features: int, layers: nn.ModuleList):
        super(Decoder, self).__init__()
        self.features = features
        self.decoder_blocks = layers
        self.norm = nn.LayerNorm(features)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        '''
        args:
            x: tensor: the input embedding vectors of shape (batch_size, seq_len, d_model)
            encoder_output: tensor: the output of the encoder of shape (batch_size, seq_len, d_model)
            src_mask: tensor: the mask to be applied to the input embedding vectors of shape (batch_size, seq_len, seq_len)
            tgt_mask: tensor: the mask to be applied to the input embedding vectors of shape (batch_size, seq_len, seq_len)
        Returns:
            decoder: tensor: the output of the decoder of shape (batch_size, seq_len, d_model)
        '''
        for decoder_block in self.decoder_blocks: # iterating through the decoder blocks (layers)
            x = decoder_block(x, enc_out, src_mask, tgt_mask)
        return self.norm(x)
          

# Linear Layer
class LinearLayer(nn.Module):
    '''
    Linear layer is used to convert the output of the transformer model to the output vocabulary size
    Args:
        d_model: int: the dimension of the model (default= 512) also known as the embedding size
        vocab_size: int: the size of the vocabulary (depends on the dataset)
    Returns:
        linear_layer: tensor: the linear layer for the transformer model
    '''
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super(LinearLayer, self).__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        '''
        args:
            x: tensor: the input embedding vectors of shape (batch_size, seq_len, d_model)
        Returns:
            linear_layer: tensor: the output of the linear layer of shape (batch_size, seq_len, vocab_size)
            also apply the softmax function to the output to get the probabilities of the output tokens
        '''
        return self.linear(x)


# Now connect all the components to create the transformer model
class Transformer(nn.Module):
    '''
    This class is used to create the transformer model
    Args:
        src_embedding: InputEmbedding : the input embedding layer
        trg_embedding: InputEmbedding: the target embedding layer
        src_postional_encoding: PositionalEncoding: the positional encoding layer for the source
        trg_postional_encoding: PositionalEncoding: the positional encoding layer for the target
        encoder: Encoder: the encoder
        decoder: Decoder: the decoder
        linear_layer: LinearLayer: the linear layer
    Returns:
        transformer: tensor: the transformer model
    '''   
    def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos_embed: PositionalEmbedding, tgt_pos_embed: PositionalEmbedding, linearlayer: LinearLayer) -> None:
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embed
        self.trg_embedding = tgt_embed
        self.src_postional_encoding = src_pos_embed
        self.trg_postional_encoding = tgt_pos_embed
        self.linear = linearlayer


    def encode(self, src, src_mask):
        '''
        args:
            src: tensor: the input tokens of shape (batch_size, seq_len)
            src_mask: tensor: the mask to be applied to the input embedding vectors of shape (batch_size, seq_len, seq_len)
        Returns:
            encoder_output: tensor: the output of the encoder of shape (batch_size, seq_len, d_model)
        '''

        src = self.src_embedding(src)
        src = self.src_postional_encoding(src)
        return self.encoder(src, src_mask)
    
    def decode(self, target, enc_out, src_mask, tgt_mask):
        '''
        args:
            trg: tensor: the target tokens of shape (batch_size, seq_len)
            encoder_output: tensor: the output of the encoder of shape (batch_size, seq_len, d_model)
            src_mask: tensor: the mask to be applied to the input embedding vectors of shape (batch_size, seq_len, seq_len)
            tgt_mask: tensor: the mask to be applied to the input embedding vectors of shape (batch_size, seq_len, seq_len)
        Returns:
            decoder_output: tensor: the output of the decoder of shape (batch_size, seq_len, d_model)
        '''
        target = self.trg_embedding(target)
        target = self.trg_postional_encoding(target)
        return self.decoder(target, enc_out, src_mask, tgt_mask)
    

    def linearlayer(self, x):
        '''
        args:
            x: tensor: the input embedding vectors of shape (batch_size, seq_len, d_model)
        Returns:
            linear_layer: tensor: the output of the linear layer of shape (batch_size, seq_len, vocab_size)
        '''
        return self.linear(x)


# building the transformer model
def trans_model(src_vocab, tgt_vocab, src_seq_len, tgt_seq_len, d_model:int = 512, num_heads: int = 8, d_ff: int = 2048, num_layers: int = 6, dropout: float = 0.1):
    '''
    This function is used to build the transformer model
    Args:
        src_vocab: int: the size of the source vocabulary
        tgt_vocab: int: the size of the target vocabulary
        src_seq_len: int: the length of the source sequence
        tgt_seq_len: int: the length of the target sequence
        d_model: int: the dimension of the model (default= 512) also known as the embedding size
        num_heads: int: the number of attention heads (default= 8)
        d_ff: int: the dimension of the feed forward network (default= 2048)
        num_layers: int: the number of encoder and decoder blocks (default= 6)
        dropout: float: the dropout rate (default= 0.1)
    '''
    # creating the input embedding layer for the source and target
    source_embedding = InputEmbedding(d_model, src_vocab)
    target_embedding = InputEmbedding(d_model, tgt_vocab)
    src_pos_embed = PositionalEmbedding(d_model, src_seq_len, dropout)
    tgt_pos_embed = PositionalEmbedding(d_model, tgt_seq_len, dropout)

    # creating the encoder and decoder blocks
    encoder_blocks = []
    for i in range(num_layers):
        enc_self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        ffn = FFN(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, enc_self_attn, ffn, dropout)
        encoder_blocks.append(encoder_block)

    # creating decoder blocks
    decoder_blocks = []
    for i in range(num_layers):
        dec_self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        dec_cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        dec_ffn = FFN(d_model, d_ff, dropout)
        dec_block = DecoderBlock(d_model, dec_self_attn, dec_cross_attn, dec_ffn, dropout)
        decoder_blocks.append(dec_block)

    # creating the encoder and decoder
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
    decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))

    # creating the linear layer
    linearlayer = LinearLayer(d_model, tgt_vocab)
    # create the transformer model
    transformer = Transformer(encoder, decoder, source_embedding, target_embedding, src_pos_embed, tgt_pos_embed, linearlayer)

    for parameter in transformer.parameters():
        if parameter.dim() > 1:
            nn.init.xavier_uniform_(parameter)

    return transformer