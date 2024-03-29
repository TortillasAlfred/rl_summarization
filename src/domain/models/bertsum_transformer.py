import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from os.path import join
from transformers import BertModel

from .encoder import TransformerInterEncoder

D_FFN = 2048
D_MODEL_BERT = 768
HEAD = 8
DROPOUT = 0.1
NUM_TLAYER = 2
DEFAULT_BERT_MAX_LEN = 512


class Classifier(nn.Module):
    def __init__(self, hidden_size=768, output_size=3):
        super(Classifier, self).__init__()
        self.linear1 = nn.Linear(hidden_size, 64)
        self.act1 = F.relu
        self.linear2 = nn.Linear(64, 64)
        self.act2 = F.relu
        self.linear3 = nn.Linear(64, 1)

    def forward(self, x, mask_cls):
        x = self.act1(self.linear1(x))
        x = self.act2(self.linear2(x))
        x = self.linear3(x).squeeze(-1)

        sent_scores = x * mask_cls.float()
        return sent_scores


class Summarizer(nn.Module):
    def __init__(self, device, config):
        super(Summarizer, self).__init__()
        self.device = device
        if not config.bert_cache:
            self.bert = BertModel.from_pretrained("bert-base-uncased")
        else:
            self.bert = BertModel.from_pretrained(
                join(config.bert_cache, "bertmodel_save_pretrained"), local_files_only=True
            )
        self.bert = self.bert.to(self.device)

        tokens_per_doc = config.max_tokens_per_doc
        if tokens_per_doc > DEFAULT_BERT_MAX_LEN:
            pos_embeddings = nn.Embedding(tokens_per_doc, self.bert.config.hidden_size)
            bert_weight = self.bert.embeddings.position_embeddings.weight.data
            bert_weight_dup = torch.cat((bert_weight,) * int(tokens_per_doc // 512 + 1))
            pos_embeddings.weight.data = bert_weight_dup[:tokens_per_doc]
            self.bert.embeddings.position_embeddings = pos_embeddings

        self.position_ids = torch.arange(tokens_per_doc, device=self.device)

        # Choosing encoder type
        if config.encoder == "Transformer":
            self.encoder = TransformerInterEncoder(
                d_model=D_MODEL_BERT, d_ff=D_FFN, heads=HEAD, dropout=DROPOUT, num_inter_layers=NUM_TLAYER
            )
        elif config.encoder == "Classifier":
            self.encoder = Classifier(hidden_size=768)
        self.to(self.device)

    def load_cp(self, pt):
        self.load_state_dict(pt["model"], strict=True)

    def forward(self, contents):
        x, segs, clss, mask, mask_cls = (
            contents["token_ids"],
            contents["segs"],
            contents["clss"],
            contents["mark"],
            contents["mark_clss"],
        )

        len_seq = contents["token_ids"].size(-1)
        top_vec = self.bert(x, mask, segs, position_ids=self.position_ids[:len_seq]).last_hidden_state

        # /input_ids, attention_mask, token_type_ids
        # sents_vec: lay [CLS] cua moi cau. shape top_vec (5, 17, 768) (5 docs, 17 sentences, 768: hiden state of word)
        if not torch.is_tensor(clss):
            max_len = max([len(x) for x in clss])

            for mask_cls_, clss_ in zip(mask_cls, clss):
                clss_.extend([0] * (max_len - len(clss_)))
                mask_cls_.extend([0] * (max_len - len(mask_cls_)))
            clss = torch.Tensor(clss).long()
            mask_cls = torch.Tensor(mask_cls).bool()

        mask_cls = mask_cls.to(self.device)
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss]

        mark_cls_ = mask_cls[:, :, None].float().to(self.device)  # mark_cls_ shape (2,24,1)
        sents_vec = sents_vec * mark_cls_
        sent_scores = self.encoder(sents_vec, mask_cls).squeeze(-1)  # sents_vec(2, 54, 768) mask_cls(1, 54)

        return sent_scores, mask_cls
