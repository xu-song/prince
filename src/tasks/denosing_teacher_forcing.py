# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
We feed corrupt tokens instead of teacher forcing with golden token.

## policy
- Denosing Generative Pre-Training
- denosing
- decoder dropout

## type
- random noise
- entity noise
"""

import os
import numpy as np
import logging
import torch

from collections import OrderedDict
from fairseq import utils
from fairseq.data import (
    data_utils,
    PrependTokenDataset,
    LanguagePairDataset,
    TokenBlockDataset,
    AppendTokenDataset,
    data_utils,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    PadDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TokenBlockDataset,
)

from fairseq.tasks import FairseqTask, register_task

from ..data.denosing_language_pair_dataset import DenosingLanguagePairDataset

logger = logging.getLogger(__name__)


@register_task('denosing_tf')
class DenosingTeacherForcingTask(FairseqTask):
    """
    Train a masked sequence-to-sequence model.

    Args:
        dictionary (~fairseq.data.Dictionary): the dictionary for the input of
            the language model
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', help='colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner')
        parser.add_argument('--sample-break-mode', default='none',
                            choices=['none', 'complete', 'complete_doc', 'eos'],
                            help='If omitted or "none", fills each sample with tokens-per-sample '
                                 'tokens. If set to "complete", splits samples only at the end '
                                 'of sentence, but may include multiple sentences per sample. '
                                 '"complete_doc" is similar but respects doc boundaries. '
                                 'If set to "eos", includes only one sentence per sample.')
        parser.add_argument('--tokens-per-sample', default=512, type=int,
                            help='max number of total tokens over all segments '
                                 'per sample for BERT dataset')
        # parser.add_argument('--max-source-positions', default=512, type=int, metavar='N',
        #                     help='max number of tokens in the source sequence')
        # parser.add_argument('--max-target-positions', default=512, type=int, metavar='N',
        #                     help='max number of tokens in the target sequence')
        parser.add_argument('--mask-prob', default=0.15, type=float,
                            help='probability of replacing a token with mask')
        parser.add_argument('--leave-unmasked-prob', default=0.1, type=float,
                            help='probability that a masked token is unmasked')
        parser.add_argument('--random-token-prob', default=0.1, type=float,
                            help='probability of replacing a token with a random token')
        parser.add_argument('--freq-weighted-replacement', default=False, action='store_true',
                            help='sample random replacement words based on word frequencies')
        parser.add_argument('--mask-whole-words', default=False, action='store_true',
                            help='mask whole words; you may also want to set --bpe')
        parser.add_argument('--shorten-method', default='none',
                            choices=['none', 'truncate', 'random_crop'],
                            help='if not none, shorten sequences that exceed --tokens-per-sample')
        parser.add_argument('--shorten-data-split-list', default='',
                            help='comma-separated list of dataset splits to apply shortening to, '
                                 'e.g., "train,valid" (default: all dataset splits)')
        parser.add_argument('--shuffle', action='store_true',
                            help='shuffle each dataset while training')
        parser.add_argument('--num-classes', type=int, default=-1,
                            help='number of classes or regression targets')
        parser.add_argument('--tagging-head-name', type=str, default=None,
                            help='')
        parser.add_argument('--tag-num-classes', type=int, default=-1,
                            help='number of tagging classes')
        parser.add_argument('--apply-decoder-mask', default=False, action='store_true',
                            help='mask words for decoder input')
        parser.add_argument('--decoder-mask-prob', default=0.15, type=float,
                            help='probability of replacing a token with mask in decoder')
        parser.add_argument('--only-mask-entity-in-decoder', default=False, action='store_true',
                            help='only_mask_entity_in_decoder')
        # fmt: on

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.mask_idx = getattr(dictionary, 'mask_index', None) or dictionary.add_symbol("<mask>")

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)

        if args.tagging_head_name:
            num_class = self.cfg.tag_num_classes if self.cfg.tag_num_classes > 0 else self.cfg.num_classes
            model.register_tagging_head(
                args.tagging_head_name,
                num_classes=num_class,
            )
        return model

    @classmethod
    def setup_task(cls, args, **kwargs):
        paths = utils.split_paths(args.data)
        assert len(paths) > 0
        dictionary = cls.load_dictionary(os.path.join(paths[0], 'dict.txt'))
        logger.info('dictionary: {} types'.format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, epoch=0, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[epoch % len(paths)]
        split_path = os.path.join(data_path, split)
        dataset = data_utils.load_indexed_dataset(
            split_path,
            self.dictionary,
            self.cfg.dataset_impl,
            combine=combine,
        )
        if dataset is None:
            raise FileNotFoundError('Dataset not found: {} ({})'.format(split, split_path))

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        dataset = PrependTokenDataset(dataset, self.source_dictionary.bos())

        s2s_dataset = DenosingLanguagePairDataset.apply_mask(
            dataset,
            dataset.sizes,
            self.source_dictionary,
            shuffle=True,
            mask_idx=self.mask_idx,
            mask_prob=self.cfg.mask_prob,
            leave_unmasked_prob=self.cfg.leave_unmasked_prob,
            random_token_prob=self.cfg.random_token_prob,
            apply_decoder_mask=self.cfg.apply_decoder_mask,
            only_mask_entity_in_decoder=self.cfg.only_mask_entity_in_decoder,
            decoder_mask_prob=self.cfg.decoder_mask_prob,
        )
        self.datasets[split] = s2s_dataset

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    def max_positions(self):
        max_positions = 1024
        if hasattr(self.cfg, 'max_positions'):
            max_positions = min(max_positions, self.cfg.max_positions)
        if hasattr(self.cfg, 'max_source_positions'):
            max_positions = min(max_positions, self.cfg.max_source_positions)
        if hasattr(self.cfg, 'max_target_positions'):
            max_positions = min(max_positions, self.cfg.max_target_positions)
        return (max_positions, max_positions)

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        dataset = StripTokenDataset(
            TokenBlockDataset(
                src_tokens,
                src_lengths,
                block_size=None,  # ignored for "eos" break mode
                pad=self.source_dictionary.pad(),
                eos=self.source_dictionary.eos(),
                break_mode="eos",
            ),
            # remove eos from (end of) target sequence
            self.source_dictionary.eos(),
        )
        src_dataset = PrependTokenDataset(
            dataset,
            token=(
                self.source_dictionary.bos()
                if getattr(self.cfg, "add_bos_token", False)
                else self.source_dictionary.eos()
            ),
        )
        tgt_dataset = AppendTokenDataset(
            dataset,
            token=self.source_dictionary.pad()
        )
        return NestedDictionaryDataset(
            {
                "id": IdDataset(),
                "net_input": {
                    "src_tokens": PadDataset(src_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False),
                    "src_lengths": NumelDataset(src_dataset, reduce=False),
                },
                "target": PadDataset(tgt_dataset, pad_idx=self.source_dictionary.pad(), left_pad=False),
            },
            sizes=[np.array(src_lengths)],
        )
