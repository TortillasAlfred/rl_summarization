import torch

from torch.utils.data import DataLoader, RandomSampler, BatchSampler


class RepeatingBatchSampler(BatchSampler):
    def __init__(self, data_source, num_samples_per_batch=1, batch_size=1, drop_last=True):
        super().__init__(RandomSampler(data_source), batch_size, drop_last)
        self._get_batch_format(num_samples_per_batch, batch_size)

    def _get_batch_format(self, num_samples_per_batch, batch_size):
        num_each_sample, extras = divmod(batch_size, num_samples_per_batch)
        self._batch_format = extras * [num_each_sample + 1] + \
            (num_samples_per_batch - extras) * [num_each_sample]

    def __iter__(self):
        batch = []
        for idx, sample in enumerate(self.sampler):
            batch.extend([sample] * self._batch_format[idx % len(self._batch_format)])
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


def get_repeating_data_loader(dataset, num_samples_per_batch=1, batch_size=1, drop_last=True):
    batch_sampler = RepeatingBatchSampler(
        dataset, num_samples_per_batch, batch_size, drop_last)

    return DataLoader(dataset, batch_sampler=batch_sampler, collate_fn=lambda x: x)


if __name__ == '__main__':
    from dataset import CnnDmDataset
    from embeddings import PretrainedEmbeddings

    emb = PretrainedEmbeddings('./data/embeddings/glove/glove.6B.50d.txt')

    cnn_dm_dataset = CnnDmDataset('dev')
    cnn_dm_dataset.preprocess(emb)

    loader = get_repeating_data_loader(cnn_dm_dataset, 3, 8)
    for batch in loader:
        print(len(batch))
        print(loader.batch_sampler._batch_format)
