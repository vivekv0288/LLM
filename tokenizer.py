import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from torch.utils.data import Dataset
import config
import os

class CharDataset(Dataset):

    def __init__(self, input_file):
        with open(input_file, 'r') as f:
            data = f.read()
        words = data.splitlines()
        words = [w.strip() for w in words] # get rid of any leading or trailing white space
        words = [w for w in words if w] # get rid of any empty strings
        chars = sorted(list(set(''.join(words)))) # all the possible characters
        chars.append('.')
        max_word_length = max(len(w) for w in words)
        print(f"number of examples in the dataset: {len(words)}")
        print(f"max word length: {max_word_length}")
        print(f"number of unique characters in the vocabulary: {len(chars)}")
        print("vocabulary:")
        print(''.join(chars))
        self.words = words
        self.chars = chars
        print("chars: {chars}")
        self.stoi = {ch:i for i,ch in enumerate(chars)}
        self.itos = {i:s for s,i in self.stoi.items()} 
        #self.generate_tokens(input_file,max_word_length)
        self.createFixedLengthDataSet(input_file)
     

    def add_period_if_short(self,item):
        for i in range(17):
            if len(item) < i:
                item += '.'
        return item

   
    def generate_tokens(self,input_file,max_word_length):
        tokens=[]
        for item in self.words:
            wrd = self.add_period_if_short(item)
            tokens.extend([self.stoi[c] for c in wrd]) 
        # merged_tokens = torch.cat(all_tokens, dim=0)
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
        tokens_np_uint16 = tokens_np.astype(np.uint16)
        file_name = input_file + '_tokens'
        np.save(file_name, tokens_np_uint16)
    
    def __len__(self):
        return len(self.words)

    def contains(self, word):
        return word in self.words

    def get_vocab_size(self):
        return len(self.chars) + 1 # all the possible characters and special 0 token

    def encode(self, word):
        ix = torch.tensor([self.stoi[w] for w in word], dtype=torch.long)
        return ix

    def decode(self, ix):
        word = ''.join(self.itos[i] for i in ix)
        return word
    
    def createFixedLengthDataSet(self,input_file):
        with open(input_file, 'r') as f:
            data = f.read()
        words = data.splitlines()
        words = [w.strip() for w in words] # get rid of any leading or trailing white space
        words = [w for w in words if w]
        result = "."+ ".".join(words)
        # Break the result into chunks of length 64
        # result_chunks = [result[i:i+64] for i in range(0, len(result), 64)]
        batch_number = 0
        index = 0
        batches = []
        while True:
            isAvailaible, data, index = self.fetchNextFromDot(result, index)
            if not isAvailaible or len(data) < 64:
                break
            batches.append(data)
        
        tokens = []
        for batch in batches:
            tokens.extend([self.stoi[c] for c in batch]) 
            
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
        tokens_np_uint16 = tokens_np.astype(np.uint16)
        file_name_without_extension = os.path.splitext(os.path.basename(input_file))[0]
        file_name = config.data_tokens_files +'/'+ file_name_without_extension + '_tokens'
        np.save(file_name, tokens_np_uint16)
    
    def fetchNextFromDot(self, result, index):
        batch_size = 64
        if index == 0:
            return  True, result[:batch_size], batch_size
        
        data = result[index: index+batch_size]
        
        index_of_first_dot = data.find('.')
        if index_of_first_dot == -1:
            return False, None, None
        
        data = data[index_of_first_dot:]
        remaining = result[index+batch_size: index+batch_size+index_of_first_dot]
        data += remaining
        
        return True, data, index+batch_size+index_of_first_dot
        

    def __getitem__(self, idx):
        word = self.words[idx]
        ix = self.encode(word)
        tkns= torch.tensor(ix, dtype=torch.long)#x, y
        return tkns
    

