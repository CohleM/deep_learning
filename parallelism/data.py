from datasets import load_dataset
import copy
# from RL2.datasets.base import BaseDataset, load_dataset
from torch.utils.data import Dataset

class RLDataset(Dataset):

    def __init__(self, data_path, responses_per_prompt):

        self.dataset = load_dataset(data_path, split='train')
        self.responses_per_prompt = responses_per_prompt

    def __getitem__(self, idx):

        ex = self.dataset[idx]
        messages = ex["messages"]
        answer = ex["answer"]

        return {
            "messages": messages,
            "answer": answer
        }
    def __len__(self):
        return len(self.dataset)



    def collate_fn(self, batch):

        return [
            copy.deepcopy(ex)
            for ex in batch
            for _ in range(self.responses_per_prompt)
        ]