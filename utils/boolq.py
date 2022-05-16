import data_utils
import torch

from torch.utils.data import Dataset


class BoolQDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_seq_length=256):
        self.encoded_data = data_utils.encode_data(dataframe,
                                                    tokenizer,
                                                    max_seq_length)
        self.label_list = data_utils.extract_labels(dataframe)

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, i):
        example = {'input_ids': self.encoded_data[0][i],
                   'attention_mask': self.encoded_data[1][i],
                   'labels': self.label_list[i]
                   }
        
        return example
