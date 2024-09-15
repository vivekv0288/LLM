
from baby_names_tokenizer import CharDataset
import config

def main():
    CharDataset(config.train_file_name)
    CharDataset(config.val_file_name)
    
    
    
if __name__ == "__main__":
    main()