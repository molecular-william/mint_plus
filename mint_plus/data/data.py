import zstandard as zstd
import random
import torch
from mint_plus.models.alphabet import Alphabet


class STRINGDataset(torch.utils.data.IterableDataset):  # like a generator
    def __init__(
        self,
        links_path: str,
        seqs_path: str,
        global_rank: int = 0,
        world_size: int = 1,
        max_examples: int = 0,
      #  split_name: str = 'training',
    ):  # are overfit, seek, max_len, concat necessary?
        super().__init__()
        self.links_path = links_path
        self.seqs_path = seqs_path
        self.global_rank = global_rank  # index overall
        self.world_size = world_size  # total gpus overall

        if max_examples:  # if each gpu has this iter, then together will get max_examples
            self.max_iters = int(max_examples // world_size)
        else:
            self.max_iters = None

    def __len__(self):
        return self.max_iters

    def __iter__(self):
        it = self.__iter_helper__()
        for i, n in enumerate(it):
            yield n  # seek is used here to skip the first "seek" iters?

    def __iter_helper__(self):
        self.seqs = {}
        links_f = iter(zstd.open(self.links_path, "rt", encoding="ascii"))
        seqs_f = iter(zstd.open(self.seqs_path, "rt", encoding="ascii"))
        i, j = 0, 0
        while True:
            try:
                name1, name2 =  next(links_f).strip().split()[:2]
                if name1 not in self.seqs:
                    name, seq = next(seqs_f).strip().split()
                    self.seqs[name] = seq
                if name2 not in self.seqs:
                    name, seq = next(seqs_f).strip().split()
                    self.seqs[name] = seq

                if i % self.world_size == self.global_rank:
                    yield self.seqs[name1], self.seqs[name2]
                    j += 1
                i += 1
                if j == self.max_iters:
                    break
            except StopIteration:
                links_f = iter(zstd.open(self.links_path, "rt", encoding="ascii"))  # keeps looping the links
                # seqs_f = iter(zstd.open(self.seqs_path, "rt", encoding="ascii"))
                # i, j = 0, 0


class CollateFn:
    def __init__(self, truncation_seq_length=None):
        self.alphabet = Alphabet.from_architecture("ESM-1b")
        self.truncation_seq_length = truncation_seq_length

    def __call__(self, batches):  # this part is interesting
        chains = zip(*batches)  # tuple of two iterables, one for seq1s, one for seq2s
        chains = [self.convert(c) for c in chains]  # convert to tokens
        chain_ids = [torch.ones(c.shape, dtype=torch.int32) * i for i, c in enumerate(chains)]  # create chain-id tensors
        chains = torch.cat(chains, -1)  # chain seq tokens concat
        chain_ids = torch.cat(chain_ids, -1)  # chain ids concat
        return chains, chain_ids

    def convert(self, seq_str_list):
        batch_size = len(seq_str_list)
        seq_encoded_list = [
            self.alphabet.encode("<cls>" + seq_str.replace("J", "L") + "<eos>")
            for seq_str in seq_str_list
        ]
        #if self.truncation_seq_length:  # choose random starting index, takes a slice of length truncation_seq_length
        #    for i in range(batch_size):
        #        seq = seq_encoded_list[i]
                #if len(seq) > self.truncation_seq_length:
                #    start = random.randint(0, len(seq) - self.truncation_seq_length + 1)
                #    seq_encoded_list[i] = seq[start : start + self.truncation_seq_length]

        #max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)

        #if self.truncation_seq_length:
        #    assert max_len <= self.truncation_seq_length
        #tokens = torch.empty((batch_size, max_len), dtype=torch.int64)
        #tokens.fill_(self.alphabet.padding_idx)  # padding

#        for i, seq_encoded in enumerate(seq_encoded_list):
#            seq = torch.tensor(seq_encoded, dtype=torch.int64)
#            tokens[i, : len(seq_encoded)] = seq
#        return tokens

        if self.truncation_seq_length:  
            for i in range(batch_size):
                seq = seq_encoded_list[i]
                if len(seq) > self.truncation_seq_length:
                    # FIX: Preserve <cls> (index 0) and <eos> (index -1)
                    inner_trunc_len = self.truncation_seq_length - 2
                    # Start index must be between 1 (after <cls>) and valid bounds
                    start = random.randint(1, len(seq) - 1 - inner_trunc_len)

                    # Reconstruct keeping structural tokens intact
                    seq_encoded_list[i] = [seq[0]] + seq[start : start + inner_trunc_len] + [seq[-1]]

        # STABILITY UPGRADE:
        if self.truncation_seq_length:
            # Always pad to the exact truncation length so every batch has the identical shape
            stable_max_len = self.truncation_seq_length
        else:
            # Round up to the nearest multiple of 128 to drastically reduce recompilations
            batch_max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
            stable_max_len = ((batch_max_len + 127) // 128) * 128

        tokens = torch.empty((batch_size, stable_max_len), dtype=torch.int64)
        tokens.fill_(self.alphabet.padding_idx)  # padding

        for i, seq_encoded in enumerate(seq_encoded_list):
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            tokens[i, : len(seq_encoded)] = seq

        return tokens
