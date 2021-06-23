import torch
import torch.nn as nn

from os import getcwd
from os.path import join
from transformers import BertModel

from .encoder import TransformerInterEncoder
from ..dataset_bert import MAX_LEN_DOCUMENT

D_FFN = 2048
HEAD = 8
DROPOUT = 0.1
NUM_TLAYER = 2


class Summarizer(nn.Module):
    def __init__(self, device, config):
        super(Summarizer, self).__init__()
        self.device = device
        if config.bert_cache:
            bert_cache_dir = join(getcwd(), "bert_cache/bertmodel_save_pretrained")
            self.bert = BertModel.from_pretrained(
                bert_cache_dir, local_files_only=True
            ).to(self.device)
        else:
            self.bert = BertModel.from_pretrained("bert-base-uncased").to(self.device)

        if MAX_LEN_DOCUMENT > 512:
            pos_embeddings = nn.Embedding(
                MAX_LEN_DOCUMENT, self.bert.config.hidden_size
            )
            bert_weight = self.bert.embeddings.position_embeddings.weight.data
            bert_weight_dup = torch.cat(
                (bert_weight,) * int(MAX_LEN_DOCUMENT // 512 + 1)
            )
            pos_embeddings.weight.data = bert_weight_dup[:MAX_LEN_DOCUMENT]
            self.bert.embeddings.position_embeddings = pos_embeddings

        self.encoder = TransformerInterEncoder(768, D_FFN, HEAD, DROPOUT, NUM_TLAYER)
        self.position_ids = torch.arange(MAX_LEN_DOCUMENT).to(self.device)
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
        top_vec = self.bert(
            x, mask, segs, position_ids=self.position_ids[:len_seq]
        ).last_hidden_state

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

        mark_cls_ = (mask_cls[:, :, None].float().to(self.device))  # mark_cls_ shape (2,24,1)
        sents_vec = sents_vec * mark_cls_
        sent_scores = self.encoder(sents_vec, mask_cls).squeeze(-1)  # sents_vec(2, 54, 768) mask_cls(1, 54)
        return sent_scores, mask_cls
