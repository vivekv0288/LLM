
from tokenizer import CharDataset
import config

def main():
    CharDataset(config.train_file_name) #generate tokens for training
    CharDataset(config.val_file_name) #generate tokens for validation
    
    
    
if __name__ == "__main__":
    main()