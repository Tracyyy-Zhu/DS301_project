import pandas as pd
import torch
import unittest

from boolq import BoolQDataset
from transformers import RobertaTokenizerFast


class TestBoolQDataset(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.dataset = pd.DataFrame.from_dict(
            {
                "question": ["question 0", "question 1"],
                "passage": ["passage 0", "passage 1"],
                "idx": [0, 1],
                "label": [True, False],
            }
        )
        self.max_seq_len = 4
        self.boolq_dataset = BoolQDataset(
            self.dataset, self.tokenizer, self.max_seq_len
        )

    def test_len(self):
        self.assertEqual(len(self.boolq_dataset), len(self.dataset))

    def test_item(self):
        for i in range(len(self.boolq_dataset)):
            self.assertEqual(list(self.boolq_dataset[i].keys()),
                             ['input_ids', 'attention_mask', 'labels'])
            self.assertEqual(len(self.boolq_dataset[i]['input_ids']), self.max_seq_len)
            self.assertEqual(len(self.boolq_dataset[i]['attention_mask']), self.max_seq_len)
            self.assertEqual(self.boolq_dataset[i]['input_ids'].dtype, torch.long)
            self.assertEqual(self.boolq_dataset[i]['attention_mask'].dtype, torch.long)
            self.assertEqual(type(self.boolq_dataset[i]['labels']), int)


if __name__ == "__main__":
    unittest.main()
