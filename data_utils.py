import torch


def encode_data(dataset, tokenizer, max_seq_length=128):
    input_ids = torch.tensor(tokenizer(dataset['question'].tolist(),
                                       dataset['passage'].tolist(),
                                       padding=True,
                                       truncation=True,
                                       max_length=max_seq_length)
                             ['input_ids'])
    attention_mask = torch.tensor(tokenizer(dataset['question'].tolist(),
                                            dataset['passage'].tolist(),
                                            padding=True,
                                            truncation=True,
                                            max_length=max_seq_length)
                                  ['attention_mask'])
    return input_ids, attention_mask


def extract_labels(dataset):
    labels = dataset['label'].replace([False,True],[0,1]).values.tolist()
    return labels
