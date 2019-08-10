from SequenceVocabulary import *
from Vocabulary import *
import collections
import string
import numpy as np
from collections import Counter
class LMVectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""     
    def __init__(self, target_vocab, max_target_length):       
        """         Args:             review_vocab (Vocabulary): maps words to integers     
        rating_vocab (Vocabulary): maps class labels to integers         """ 
        self.target_vocab = target_vocab
        self.max_target_length = max_target_length
        
    def _vectorize(self, indices, vector_length=-1, mask_index=0):
        """Vectorize the provided indices
        
        Args:
            indices (list): a list of integers that represent a sequence
            vector_length (int): an argument for forcing the length of index vector
            mask_index (int): the mask_index to use; almost always 0
        """
        if vector_length < 0:
            vector_length = len(indices)
        
        vector = np.zeros(vector_length, dtype=np.int64)
        vector[:len(indices)] = indices
        vector[len(indices):] = mask_index

        return vector
    
    def _get_source_indices(self, text):
        """Return the vectorized source text
        
        # # # Args:
            # # # text (str): the source text; tokens should be separated by spaces
        # # # Returns:
            # # # indices (list): list of integers representing the text
        # # # """
    def _get_target_indices(self, text):
        """Return the vectorized source text
        
        Args:
            text (str): the source text; tokens should be separated by spaces
        Returns:
            a tuple: (x_indices, y_indices)
                x_indices (list): list of integers representing the observations in target decoder 
                y_indices (list): list of integers representing predictions in target decoder
        """
        indices = [self.target_vocab.lookup_token(token) for token in text.split()]
        x_indices = [self.target_vocab.begin_seq_index] + indices
        y_indices = indices + [self.target_vocab.end_seq_index]
        return x_indices, y_indices
    # # # def vectorize(self, source_text, target_text, use_dataset_max_lengths=True):
    def vectorize(self,  target_text, use_dataset_max_lengths=True):

        # # # source_vector_length = -1
        target_vector_length = -1
        
        if use_dataset_max_lengths:
            # # # source_vector_length = self.max_source_length + 2
            target_vector_length = self.max_target_length + 1
            
        # # # source_indices = self._get_source_indices(source_text)
        # # # source_vector = self._vectorize(source_indices, 
                                        # # # vector_length=source_vector_length, 
                                        # # # mask_index=self.source_vocab.mask_index)
        
        target_x_indices, target_y_indices = self._get_target_indices(target_text)
        target_x_vector = self._vectorize(target_x_indices,
                                        vector_length=target_vector_length,
                                        mask_index=self.target_vocab.mask_index)
        target_y_vector = self._vectorize(target_y_indices,
                                        vector_length=target_vector_length,
                                        mask_index=self.target_vocab.mask_index)
        return {
                "target_x_vector": target_x_vector, 
                "target_y_vector": target_y_vector, 
                "target_length": len(target_x_indices)}
    @classmethod    
    def from_dataframe(cls, bitext_df, cutoff = 3):   
        target_vocab = SequenceVocabulary()
        word_counts = Counter()
        max_target_length = 0
        for _,row in bitext_df.iterrows():
            target_tokens = row["source_language"].split()
            if len(target_tokens) > max_target_length:
               max_target_length = len(target_tokens)
            for token in target_tokens:
                word_counts[token] += 1
        for word, count in word_counts.items():
            if count > cutoff:
                target_vocab.add_token(word)
        return cls(target_vocab,max_target_length)
    @classmethod    
    def from_serializable(cls, contents):
        target_vocab = SequenceVocabulary.from_serializable(contents["target_vocab"])
        return cls(
                   target_vocab=target_vocab,
                   max_target_length=contents["max_target_length"])

    def to_serializable(self):
        return {"target_vocab": self.target_vocab.to_serializable(), 
                "max_target_length": self.max_target_length}
