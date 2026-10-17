"""Read approved English token files as lossless causal-LM blocks.

This module uses NumPy only. It does not read raw text, write corpus files, use a
GPU, or start training. Arrays can be collated by a PyTorch DataLoader directly.
Every within-source next-token transition appears once per complete epoch.
"""
from __future__ import annotations

from bisect import bisect_right
import json
import operator
import os
from pathlib import Path
from typing import Iterator

import numpy as np

SOURCES = ('wikipedia', 'gutenberg', 'web_articles', 'arxiv')
SPLITS = ('train', 'validation', 'test')
PAD_TOKEN_ID = 0
IGNORE_TARGET = -100


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _nonnegative_integer(value, label):
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0,
             label + ' must be a nonnegative integer')
    return value


class PackedTokenDataset:
    """Map-style dataset with source-local, consecutive T+1 token windows.

The last block of each source is padded, never dropped. ``valid_tokens`` counts
targets that contribute to loss; ``targets == -100`` excludes padding. Token EOS
IDs remain ordinary targets. ``source_id`` indexes the stable ``sources`` tuple.
Only source boundaries stop packing; EOS already separates document boundaries.

Memmaps open lazily in read-only mode in each worker. Returned arrays own their
memory, so closing a dataset cannot invalidate a previously returned block.
"""

    def __init__(self, root, split, sequence_length=1024):
        self.root = Path(root).resolve()
        _require(split in SPLITS, 'Unknown split: ' + str(split))
        self.split = split
        try:
            self.sequence_length = operator.index(sequence_length)
        except TypeError as error:
            raise ValueError('sequence_length must be a positive integer') from error
        _require(not isinstance(sequence_length, bool) and self.sequence_length > 0,
                 'sequence_length must be a positive integer')
        self.manifest = json.loads((self.root / 'manifest.json').read_text(encoding='utf-8'))
        self.release_validation = json.loads(
            (self.root / 'release_validation.json').read_text(encoding='utf-8'))
        manifest, release = self.manifest, self.release_validation
        _require(manifest.get('status') == 'completed', 'A completed preparation is required')
        _require(manifest.get('language') == 'English', 'Only the approved English corpus is supported')
        _require(release.get('status') == 'passed', 'Independent release validation must have passed')
        for field in ('input_corpus_hashes_unchanged', 'raw_metadata_unchanged',
                      'next_token_window_check', 'parent_and_duplicate_components_no_cross_split'):
            _require(release.get(field) is True, 'Release validation is missing: ' + field)
        _require(manifest.get('inputs_sha256_unchanged') is True
                 and manifest.get('raw_metadata_unchanged') is True,
                 'Preparation input integrity checks are missing')
        _require(manifest['tokenizer'].get('training_set_only_verified') is True,
                 'Tokenizer training-set-only validation is missing')
        _require(manifest['tokenizer']['special_token_ids'].get('<|pad|>') == PAD_TOKEN_ID,
                 'The dataset requires PAD token id 0')
        _require(manifest['tokens'].get('dtype') == 'little-endian uint16'
                 and manifest['tokens'].get('index_dtype') == 'little-endian uint64',
                 'Unexpected token storage dtype')
        _require(set(manifest['splits']['sources']) == set(SOURCES),
                 'Unexpected or missing approved source')
        _require(all(manifest['splits']['leakage_checks'].values()),
                 'Preparation split leakage checks did not pass')
        statistics = manifest['tokens']['sources']
        expected_keys = {partition + '/' + source for partition in SPLITS for source in SOURCES}
        _require(set(statistics) == expected_keys, 'Unexpected or missing split/source token statistics')
        self.sources = SOURCES
        self.source_metadata = []
        self.block_ends = []
        self.effective_targets = 0
        total_blocks = 0
        # All three partitions must have complete file triples; this also catches
        # a partial corpus directory before DataLoader workers are spawned.
        for partition in SPLITS:
            for source_id, source in enumerate(SOURCES):
                key = partition + '/' + source
                stats = statistics[key]
                tokens = _nonnegative_integer(stats['tokens'], key + ' tokens')
                documents = _nonnegative_integer(stats['documents'], key + ' documents')
                _require(stats['bin_bytes'] == tokens * 2, 'Manifest token byte count mismatch: ' + key)
                _require(documents == manifest['splits']['sources'][source].get(partition + '_records', 0),
                         'Split and token document counts disagree: ' + key)
                base = self.root / 'tokens' / partition / source
                paths = [base.with_suffix(suffix) for suffix in ('.bin', '.idx', '.documents.jsonl')]
                for path in paths:
                    resolved = path.resolve()
                    _require(resolved.is_relative_to(self.root) and path.is_file(),
                             'Missing or external prepared token file: ' + str(path))
                _require(paths[0].stat().st_size == tokens * 2, 'Token file size mismatch: ' + key)
                _require(paths[1].stat().st_size == (documents + 1) * 8,
                         'Document index file size mismatch: ' + key)
                if partition == split:
                    targets = max(0, tokens - 1)
                    blocks = (targets + self.sequence_length - 1) // self.sequence_length
                    self.source_metadata.append({'source_id': source_id, 'source': source,
                        'path': paths[0], 'tokens': tokens, 'effective_targets': targets,
                        'blocks': blocks, 'first_block_id': total_blocks, 'statistics': dict(stats)})
                    self.effective_targets += targets
                    total_blocks += blocks
                    self.block_ends.append(total_blocks)
        self._length = total_blocks
        self._maps = {}
        self._map_pid = None

    def __len__(self):
        return self._length

    @property
    def total_effective_targets(self):
        return self.effective_targets

    def _tokens(self, source_id):
        pid = os.getpid()
        if self._map_pid != pid:
            self.close()
            self._map_pid = pid
        if source_id not in self._maps:
            entry = self.source_metadata[source_id]
            # A selected source always has at least one transition/two tokens.
            self._maps[source_id] = np.memmap(entry['path'], mode='r', dtype='<u2',
                                               shape=(entry['tokens'],))
        return self._maps[source_id]

    def __getitem__(self, index):
        try:
            index = operator.index(index)
        except TypeError as error:
            raise IndexError('Block index must be an integer') from error
        if index < 0 or index >= len(self):
            raise IndexError(index)
        source_id = bisect_right(self.block_ends, index)
        entry = self.source_metadata[source_id]
        local_block = index - entry['first_block_id']
        offset = local_block * self.sequence_length
        valid = min(self.sequence_length, entry['effective_targets'] - offset)
        values = np.array(self._tokens(source_id)[offset:offset + valid + 1], dtype=np.int64, copy=True)
        _require(len(values) == valid + 1, 'Token file changed after dataset validation')
        inputs = np.full(self.sequence_length, PAD_TOKEN_ID, dtype=np.int64)
        targets = np.full(self.sequence_length, IGNORE_TARGET, dtype=np.int64)
        inputs[:valid] = values[:-1]
        targets[:valid] = values[1:]
        return {'input_ids': inputs, 'targets': targets, 'valid_tokens': int(valid),
                'block_id': int(index), 'source_id': int(source_id)}

    def canonical_indices(self, start=0):
        """Unshuffled validation/test iteration, with an optional fixed offset."""
        start = operator.index(start)
        _require(0 <= start <= len(self), 'Canonical start is outside dataset')
        return range(start, len(self))

    def close(self):
        for mapped in self._maps.values():
            mapped._mmap.close()
        self._maps = {}
        self._map_pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_maps'] = {}
        state['_map_pid'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._maps = {}
        self._map_pid = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        # Initialization can fail before _maps exists.
        try:
            self.close()
        except (AttributeError, TypeError):
            pass


class EpochBlockSampler:
    """Deterministic permutation with a FIXED resume start, never a live cursor.

DataLoader prefetch can request future blocks before they are consumed. The
trainer must record consumed blocks separately, then construct the next sampler
with ``start=consumed_blocks``. Repeated iteration yields the same suffix.
    """

    def __init__(self, dataset_size, seed=42, epoch=0, start=0, shuffle=True):
        try:
            self.dataset_size = operator.index(dataset_size)
            self.seed = operator.index(seed)
            self.epoch = operator.index(epoch)
            self.start = operator.index(start)
        except TypeError as error:
            raise ValueError('Sampler size, seed, epoch and start must be integers') from error
        _require(self.dataset_size >= 0 and self.seed >= 0 and self.epoch >= 0,
                 'Sampler size, seed and epoch must be nonnegative')
        _require(0 <= self.start <= self.dataset_size, 'Resume start is outside permutation')
        self.shuffle = bool(shuffle)
        self._order = (np.random.default_rng(self.seed + self.epoch).permutation(self.dataset_size)
                       if self.shuffle else np.arange(self.dataset_size, dtype=np.int64))

    def __len__(self):
        return self.dataset_size - self.start

    def __iter__(self) -> Iterator[int]:
        # __iter__ deliberately does not update self.start during prefetch.
        return (int(index) for index in self._order[self.start:])
