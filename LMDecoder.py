import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LMDecoder(nn.Module):
    def __init__(self, num_embeddings, embedding_size, rnn_hidden_size, bos_index , pretrained_embeddings=None):
        """
        Args:
            num_embeddings (int): number of embeddings is also the number of 
                unique words in target vocabulary 
            embedding_size (int): the embedding vector size
            rnn_hidden_size (int): size of the hidden rnn state
            bos_index(int): begin-of-sequence index
        """
        super(LMDecoder, self).__init__()

        if pretrained_embeddings is None:

            self.target_embedding = nn.Embedding(         num_embeddings=num_embeddings, 
                                             embedding_dim=embedding_size, 
                                             padding_idx=0)        
        else:
            pretrained_embeddings = torch.from_numpy(pretrained_embeddings).float()
            self.target_embedding = nn.Embedding(num_embeddings=num_embeddings, 
                                                 embedding_dim=embedding_size, 
                                                 padding_idx=0,
                                                 _weight=pretrained_embeddings)
                                    
        #self._rnn_hidden_size = rnn_hidden_size                  ########################
        self.gru_cell = nn.GRUCell(embedding_size , 
                                   rnn_hidden_size)               #########################
        self.hidden_map = nn.Linear(rnn_hidden_size, rnn_hidden_size)
        self.classifier = nn.Linear(rnn_hidden_size , num_embeddings)
        self.bos_index = bos_index
        self._sampling_temperature = 3
    
    def _init_indices(self, batch_size):
        """ return the BEGIN-OF-SEQUENCE index vector """
        return torch.ones(batch_size, dtype=torch.int64) * self.bos_index
    
    # def _init_context_vectors(self, batch_size):                               ###################
        # """ return a zeros vector for initializing the context """
        # return torch.zeros(batch_size, self._rnn_hidden_size)
            
    def forward(self, target_sequence , sample_probability = 0.0):
        """The forward pass of the model
        
        Args:
            encoder_state (torch.Tensor): the output of the LMEncoder
            initial_hidden_state (torch.Tensor): The last hidden state in the  LMEncoder
            target_sequence (torch.Tensor): the target text data tensor
        Returns:
            output_vectors (torch.Tensor): prediction vectors at each output step
        """    
        if target_sequence is None:
            sample_probability = 1.0
        else:
            target_sequence = target_sequence.permute(1, 0)
            output_sequence_size = target_sequence.size(0)
        batch_size = 64
        y_t_index = self._init_indices(batch_size)
        output_vectors = []
        self._cached_ht = []
        
        for i in range(output_sequence_size):
            use_sample= np.random.random() < sample_probability
            if not use_sample:
               y_t_index = target_sequence[i]
                
            y_input_vector = self.target_embedding(y_t_index)
            h_t = self.gru_cell(y_input_vector)
            self._cached_ht.append(h_t.cpu().detach().numpy())

            score_for_y_t_index = self.classifier(F.dropout(h_t, 0.3))
            
            if use_sample:
                p_y_t_index = F.Softmax(score_for_y_t_index * self._sampling_temperature)
                y_t_index = torch.multinomial(p_y_t_index , 1).squeeze()
            
            output_vectors.append(score_for_y_t_index)
            
        output_vectors = torch.stack(output_vectors).permute(1, 0, 2)
        
        return output_vectors