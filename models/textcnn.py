import os

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):    
    def __init__(self, vocab_size, cls_num):
        super(TextCNN, self).__init__()
        emb_dim = 128
        Ci = 1
        Co = 100
        Ks = [3, 4, 5]

        self.embed = nn.Embedding(vocab_size, emb_dim)
        self.convs1 = nn.ModuleList([nn.Conv2d(Ci, Co, (K, emb_dim)) for K in Ks])
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(len(Ks)*Co, cls_num)
        self.criterion = torch.nn.CrossEntropyLoss().cuda()

    def save_pretrained(self, saved_path):
        torch.save(self.state_dict(), os.path.join(saved_path, 'model.pt'))

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        x = self.embed(input_ids)  # (N, seqlen, emb_dim)

        x = torch.mul(x, attention_mask.unsqueeze(-1).float())

        x = x.unsqueeze(1)  # (N, Ci, seqlen, emb_dim)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, seqlen), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logits = self.fc1(x)  # (N, C)
        if labels is None:
            return [logits]

        loss = self.criterion(logits, labels)
        return loss, logits