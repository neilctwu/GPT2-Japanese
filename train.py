from transformers import GPT2LMHeadModel, GPT2Config, BertJapaneseTokenizer
from torch.utils.data import DataLoader, Dataset

tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

# TODO:
class MyData(Dataset):
    def __init__(self):
        return


n=1
dataset = MyData()
data_loader = DataLoader(dataset, batch_size=n)

config = GPT2Config(vocab_size=tokenizer.vocab_size,
                    n_positions=n)
model = GPT2LMHeadModel(config)

for n_epoch in range(epoches):
    for text, label in