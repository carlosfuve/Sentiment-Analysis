import torch
from torch.utils.data import Dataset

class DataLoaderBert(Dataset):
  def __init__(self, reviews, targets, tokenizer, max_len, include_raw_text=False):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
    self.include_raw_text = include_raw_text

  def __len__(self):
    return len(self.reviews)

  def __getitem__(self, index):
    review = self.reviews[index]
    target = self.targets[index]

    encoding = self.tokenizer.encode_plus(review,
                                          add_special_tokens=True,
                                          padding='max_length',
                                          truncation=True,
                                          max_length=self.max_len,
                                          return_token_type_ids=False,
                                          return_attention_mask=True,
                                          return_tensors='pt')

    output = {
        'input_ids': encoding['input_ids'].flatten(), #Tamaño de [512], antes del flatten [1, 512]
        'attention_mask': encoding['attention_mask'].flatten(), #Tamaño de [512], antes del flatten [1, 512]
        'targets': torch.tensor(target,dtype=torch.long)
    }
    if self.include_raw_text:
      output['review_text'] = review

    return output