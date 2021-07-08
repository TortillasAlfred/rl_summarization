from src.domain.loader_utils import TextDataCollator

import time
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical
from torch.optim.lr_scheduler import ReduceLROnPlateau


# TODO: Créer un fichier bertbanditsum.py, on ne veut pas perdre le contenu de ce fichier
class BanditSum(pl.LightningModule):
    def __init__(self, dataset, reward, hparams):
        # TODO: Devrait basically être remplacé par ce qui est dans BertCombiSum
        # Il faut seulement s'assurer de garder self.epsilon et self.n_repeat_per_sample,
        # qui sont des hparams propres à BanditSum
        super(BanditSum, self).__init__()
        self.hparams = hparams
        self.fields = dataset.fields
        self.pad_idx = dataset.pad_idx
        self.reward_builder = reward

        self.embedding_dim = dataset.embedding_dim
        self.pad_idx = dataset.pad_idx
        self.splits = dataset.get_splits()
        self.n_epochs_done = 0

        self.train_batch_size = hparams.train_batch_size
        self.num_workers = hparams.num_workers
        self.test_batch_size = hparams.test_batch_size
        self.hidden_dim = hparams.hidden_dim
        self.decoder_dim = hparams.decoder_dim
        self.n_repeats_per_sample = hparams.n_repeats_per_sample
        self.learning_rate = hparams.learning_rate
        self.epsilon = hparams.epsilon
        self.n_sents_per_summary = hparams.n_sents_per_summary
        self.weight_decay = hparams.weight_decay
        self.batch_idx = 0

        self.__build_model(dataset)
        self.model = RLSummModel(hparams.hidden_dim, hparams.decoder_dim)

    def __build_model(self, dataset):
        # TODO: Delete
        self.embeddings = torch.nn.Embedding.from_pretrained(
            dataset.vocab.vectors, freeze=False, padding_idx=self.pad_idx
        )
        self.wl_encoder = torch.nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

    def word_level_encoding(self, contents):
        # TODO: Delete
        valid_tokens = ~(contents == self.pad_idx)
        sentences_len = valid_tokens.sum(dim=-1)
        valid_sentences = sentences_len > 0
        contents = self.embeddings(contents)
        orig_shape = contents.shape
        contents = self.wl_encoder(contents.view(-1, *orig_shape[2:]))[0].reshape(*orig_shape[:3], -1)
        contents = contents * valid_tokens.unsqueeze(-1)
        contents = contents.sum(-2)
        word_level_encodings = torch.zeros_like(contents)
        word_level_encodings[valid_sentences] = contents[valid_sentences] / sentences_len[valid_sentences].unsqueeze(-1)
        return word_level_encodings, valid_sentences

    def __extract_features(self, contents):
        # TODO: Delete
        contents, valid_sentences = self.word_level_encoding(contents)
        sent_contents = self.model.sentence_level_encoding(contents)
        affinities = self.model.produce_affinities(sent_contents)
        affinities = affinities * valid_sentences

        return affinities, valid_sentences

    def select_idxs(self, affinities, valid_sentences):
        # TODO: Rouler Banditsum et noter la forme de affinities et valid_sentences.
        # C'est le contrat que tu devrais respecter en changeant l'encodeur par
        # BERT au lieu des LSTMs.
        #
        # Si affinites et valid_sentences sont toujours de taille 50 par défaut, ce
        # n'est pas grave que BERT retourne un nombre flexible de phrases par doc
        # comme il fait dans BertCombiSum. Tant que affinities et valid_sentences
        # sont de la même shape, on est OK.
        #
        # IMPORTANT: CETTE MÉTHODE NE DEVRAIT PAS ÊTRE TOUCHÉE. C'EST UNE MÉTHODE
        # UN PEU CONTRE-INTUITIVE MAIS TRÈS EFFICACE DE PARRALLÉLISER BANDITSUM
        # ET C'EST 100 % INDÉPENDANT DE L'ENCODEUR UTILISÉ
        n_docs = affinities.shape[0]

        affinities = affinities.repeat_interleave(self.n_repeats_per_sample, 0)
        valid_sentences = valid_sentences.repeat_interleave(self.n_repeats_per_sample, 0)

        all_idxs = torch.zeros(
            (n_docs, self.n_repeats_per_sample, self.n_sents_per_summary),
            dtype=torch.int,
            device=affinities.device,
        )
        all_logits = torch.zeros(
            (n_docs, self.n_repeats_per_sample, self.n_sents_per_summary),
            dtype=torch.float,
            device=affinities.device,
        )

        uniform_distro = torch.ones_like(affinities) * valid_sentences
        for step in range(self.n_sents_per_summary):
            step_affinities = affinities * valid_sentences
            step_affinities = step_affinities / step_affinities.sum(-1, keepdim=True)

            step_unif = uniform_distro * valid_sentences
            step_unif = step_unif / step_unif.sum(-1, keepdim=True)

            step_probas = self.epsilon * step_unif + (1 - self.epsilon) * step_affinities

            c = Categorical(step_probas)

            idxs = c.sample()
            logits = c.logits.gather(1, idxs.unsqueeze(1))

            valid_sentences = valid_sentences.scatter(1, idxs.unsqueeze(1), 0)

            all_idxs[:, :, step] = idxs.view(n_docs, self.n_repeats_per_sample)
            all_logits[:, :, step] = logits.view(n_docs, self.n_repeats_per_sample)

        return all_idxs, all_logits.sum(-1)

    def forward(self, batch, subset):
        raw_contents, contents, raw_abstracts, abstracts, ids, scorers = batch
        batch_size = len(contents)

        # TODO: Delete
        self.wl_encoder.flatten_parameters()
        self.model.sl_encoder.flatten_parameters()

        # TODO: Remplacer par le code de prédiction via BERT et un encodeur
        # présent dans BertCombiSum
        action_vals, valid_sentences = self.__extract_features(contents)
        _, greedy_idxs = torch.topk(action_vals, self.n_sents_per_summary, sorted=False)

        if subset == "train":
            # Toute cette section ne devrait pas être touchée.
            # Elle marche très bien avec les LSTMs et les modifs de BERT
            # n'ont pas d'impact ici.
            greedy_rewards = []
            for scorer, sent_idxs in zip(scorers, greedy_idxs):
                greedy_rewards.append(scorer(sent_idxs.tolist()))
            greedy_rewards = torch.tensor(greedy_rewards).unsqueeze(1)

            selected_idxs, selected_logits = self.select_idxs(action_vals, valid_sentences)

            generated_rewards = []
            for scorer, batch_idxs in zip(scorers, selected_idxs):
                generated_rewards.append(torch.tensor([scorer(sent_idxs.tolist()) for sent_idxs in batch_idxs]))
            generated_rewards = torch.stack(generated_rewards)

            greedy_rewards = greedy_rewards.repeat_interleave(self.n_repeats_per_sample, 1)

            # Cette ligne a l'air bizarre, mais elle doit être restée comme ça
            # PTI: on doit détacher le vecteur de rewards pcq il agit seulement à titre
            # de "weighting" dans REINFORCE. Voir le livre de Sutton et surtout le policy gradient
            # theorem. Si tu veux en savoir plus, je suis toujours content d'en parler.
            rewards = ((greedy_rewards - generated_rewards)).clone().detach().to(device=selected_logits.device)
            loss = rewards * selected_logits
            loss = loss.mean()

            # PTI:
            # generated_rewards = la moyenne de ROUGE en PIGEANT de la distribution prédite
            # greedy_rewards = la moyenne de ROUGE en SÉLECTIONNANT LES 3 MEILLEURS IDX de la distro prédite
            # loss = pas aussi directement interprétable qu'habituellement. Indique seulement dans quelle direction
            # le gradient pointe.
            return generated_rewards, loss, greedy_rewards
        else:
            # TODO: Prendre la version de BertCombiSum
            greedy_rewards = scorers.get_scores(greedy_idxs, raw_contents, raw_abstracts)

            return torch.from_numpy(greedy_rewards)

    def training_step(self, batch, batch_idx):
        #
        start = time.time()
        generated_rewards, loss, greedy_rewards = self.forward(batch, subset="train")
        end = time.time()

        log_dict = {
            "greedy_rouge_mean": greedy_rewards.mean(),
            "generated_rouge_mean": generated_rewards.mean(),
            "loss": loss.detach(),
            "batch_time": end - start,
        }

        for key, val in log_dict.items():
            self.log(key, val, prog_bar="greedy" in key)

        return loss

    def validation_step(self, batch, batch_idx):
        greedy_rewards = self.forward(batch, subset="val")

        reward_dict = {
            "val_greedy_rouge_1": greedy_rewards[:, 0],
            "val_greedy_rouge_2": greedy_rewards[:, 1],
            "val_greedy_rouge_L": greedy_rewards[:, 2],
            "val_greedy_rouge_mean": greedy_rewards.mean(-1),
        }

        for name, val in reward_dict.items():
            self.log(name, val, prog_bar="mean" in name)

        return reward_dict["val_greedy_rouge_mean"].mean()

    def validation_epoch_end(self, outputs):
        mean_rouge = torch.stack(outputs).mean()
        self.lr_scheduler.step(mean_rouge)

        current_lr = self.trainer.optimizers[0].param_groups[1]["lr"]
        self.log("lr", current_lr)

    def test_step(self, batch, batch_idx):
        greedy_rewards = self.forward(batch, subset="test")

        reward_dict = {
            "test_greedy_rouge_1": greedy_rewards[:, 0],
            "test_greedy_rouge_2": greedy_rewards[:, 1],
            "test_greedy_rouge_L": greedy_rewards[:, 2],
            "test_greedy_rouge_mean": greedy_rewards.mean(-1),
        }

        for name, val in reward_dict.items():
            self.log(name, val)

    def configure_optimizers(self):
        # TODO: Remplacer par la version que tu as dans BertCombiSum
        optimizer = torch.optim.Adam(
            [
                {
                    "params": self.embeddings.parameters(),
                    "lr": self.learning_rate * 0.1,
                },
                {"params": self.wl_encoder.parameters()},
                {"params": self.model.sl_encoder.parameters()},
                {"params": self.model.decoder.parameters()},
            ],
            lr=self.learning_rate,
            betas=[0, 0.999],
            weight_decay=self.weight_decay,
        )

        self.lr_scheduler = ReduceLROnPlateau(optimizer, mode="max", patience=10, factor=0.2, verbose=True)

        return optimizer

    def train_dataloader(self):
        # TODO: Remplacer par la version que tu as dans BertCombiSum
        dataset = self.splits["train"]
        return DataLoader(
            dataset,
            collate_fn=TextDataCollator(self.fields, self.reward_builder, subset="train"),
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=False,
        )

    def val_dataloader(self):
        # TODO: Remplacer par la version que tu as dans BertCombiSum
        dataset = self.splits["val"]
        return DataLoader(
            dataset,
            collate_fn=TextDataCollator(self.fields, self.reward_builder, subset="val"),
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def test_dataloader(self):
        # TODO: Remplacer par la version que tu as dans BertCombiSum
        dataset = self.splits["test"]
        return DataLoader(
            dataset,
            collate_fn=TextDataCollator(self.fields, self.reward_builder, subset="test"),
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    @staticmethod
    def from_config(dataset, reward, config):
        return BanditSum(
            dataset,
            reward,
            config,
        )


# TODO: Delete
class RLSummModel(torch.nn.Module):
    def __init__(self, hidden_dim, decoder_dim):
        super().__init__()
        self.sl_encoder = torch.nn.LSTM(
            input_size=2 * hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, decoder_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(decoder_dim, 1),
            torch.nn.Sigmoid(),
        )

    def sentence_level_encoding(self, contents):
        sent_contents, _ = self.sl_encoder(contents)

        return sent_contents

    def produce_affinities(self, sent_contents):
        affinities = self.decoder(sent_contents).squeeze(-1)

        return affinities

    def forward(self, x):
        pass
