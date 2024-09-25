from datasets import load_dataset
import numpy as np
import config

def load_dataset():
    # Load the dataset
    dataset = load_dataset("jbrazzy/baby_names")
    return dataset['train']['Names']

def fetchNextFromDot( result, index):
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

def generate_tokens():
    words = load_dataset()
    words = [w.strip() for w in words] # get rid of any leading or trailing white space
    words = [w for w in words if w] # get rid of any empty strings
    chars = sorted(list(set(''.join(words)))) # all the possible characters
    chars.append('.')
    stoi = {ch:i for i,ch in enumerate(chars)}

    index = 0
    batches = []
    result = "."+ ".".join(words)
    while True:
        isAvailaible, data, index = fetchNextFromDot(result, index)
        if not isAvailaible or len(data) < 64:
            break
        batches.append(data)

    tokens = []
    for batch in batches:
        tokens.extend([stoi[c] for c in batch]) 
        
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    np.save(config.train_token_file, tokens_np_uint16)
    