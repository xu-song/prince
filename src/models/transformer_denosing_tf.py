# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from fairseq import utils
from fairseq.data import data_utils

from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.transformer import (
    TransformerEncoder,
    TransformerDecoder,
    TransformerModel,
    base_architecture,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models.transformer import TransformerModel  # 支持任务：seq2seq
from fairseq.models.bart.model import BARTModel  # 支持任务：seq2seq cls
from .transformer_mass import TransformerMASSModel  # 支持任务：seq2seq mlm clm
from .transformer_kplug import TransformerKplugModel  # 支持任务：seq2seq mlm clm cls ner
# current model                                          # 支持任务：seq2seq mlm clm cls ent_cls rel_cls
from .transformer_hub_interface import TransformerHubInterface
from .transformer_plus import TransformerEncoderPLUS, TransformerDecoderPLUS
from ..modules.output_heads import SentenceClassificationHead, MaskedLMHead, SequenceTaggingHead
from fairseq.models.bart.model import bart_base_architecture, bart_large_architecture


@register_model('transformer_dtf')
class DenosingTeacherForcingModel(TransformerKplugModel):
    """
    denosing teacher forcing / dropout
    """

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="gpt2",
        sample_break_mode="eos",
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            sample_break_mode=sample_break_mode,
            **kwargs,
        )
        return BARTHubInterface(x["args"], x["task"], x["models"][0])



    def forward(self, src_tokens=None, src_lengths=None, prev_output_tokens=None, prev_output_positions=None,
                masked_tokens=None, features_only=False, classification_head_name=None):
        """
        """
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths)
        encoder_feature = encoder_out['encoder_out'][0].transpose(0, 1)  # T x B x C -> B x T x C

        # 1. encoder model
        if features_only:
            return encoder_feature

        # 2. encoder-decoder model
        decoder_out = None
        extra = {}

        if prev_output_tokens is not None and len(prev_output_tokens) > 0:
            if masked_tokens is not None and masked_tokens.get('decoder_mask', None) is not None:
                encoder_out = self.slice_encoder_out(encoder_out, masked_tokens['decoder_mask'])
            decoder_out, extra = self.decoder(prev_output_tokens, encoder_out=encoder_out,
                                              prev_output_positions=prev_output_positions)
            if torch.isnan(decoder_out).any():
                print('catch decoder nan')

        if masked_tokens:
            if masked_tokens.get('clm', None) is not None:
                extra['clm_out'] = self.get_clm_output(decoder_out, masked_tokens['clm'])
            if masked_tokens.get('mlm', None) is not None:
                extra['mlm_out'] = self.get_mlm_output(encoder_feature, masked_tokens['mlm'])

        return decoder_out, extra



@register_model_architecture("transformer_dtf", "transformer_dtf_base")
def kgnet_base_architecture(args):
    args.share_encoder_input_output_embed = getattr(args, 'share_encoder_input_output_embed', True)
    bart_base_architecture(args)


@register_model_architecture("transformer_dtf", "transformer_dtf_large")
def kgnet_large_architecture(args):
    args.share_encoder_input_output_embed = getattr(args, 'share_encoder_input_output_embed', True)
    bart_large_architecture(args)
