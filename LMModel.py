import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from LMDecoder import *


class LMModel(nn.Module):
    """ The Neural Machine Translation Model """
    def __init__(self, target_vocab_size, target_embedding_size, decoding_size, pretrained_embeddings, 
                 target_bos_index):
        """
        Args:
            source_vocab_size (int): number of unique words in source language
            source_embedding_size (int): size of the source embedding vectors
            target_vocab_size (int): number of unique words in target language
            target_embedding_size (int): size of the target embedding vectors
            encoding_size (int): the size of the encoder RNN.  
        """
        super(LMModel, self).__init__() 
        self.decoder = LMDecoder(num_embeddings=target_vocab_size, 
                                  embedding_size=target_embedding_size, 
                                  rnn_hidden_size=decoding_size,
                                  pretrained_embeddings=pretrained_embeddings,
                                  bos_index=target_bos_index)
    
    def forward(self,target_sequence):
        """The forward pass of the model
        
        Args:
            x_source (torch.Tensor): the source text data tensor. 
                x_source.shape should be (batch, vectorizer.max_source_length)
            x_source_lengths torch.Tensor): the length of the sequences in x_source 
            target_sequence (torch.Tensor): the target text data tensor
        Returns:
            decoded_states (torch.Tensor): prediction vectors at each output step
        """
        decoded_states = self.decoder(target_sequence=target_sequence)
        return decoded_states