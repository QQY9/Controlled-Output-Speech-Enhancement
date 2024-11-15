# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adiyoss

import json
import logging
from pathlib import Path
import os
import time

import torch
import torch.nn.functional as F

from . import augment, distrib, pretrained
from .enhance import enhance
from .evaluate import evaluate
from .stft_loss import MultiResolutionSTFTLoss
from .utils import bold, copy_state, pull_metric, serialize_model, swap_state, LogProgress

logger = logging.getLogger(__name__)


class Test(object):
    def __init__(self, data, model, laminput, optimizer, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.tt_loader = data['tt_loader']
        self.model = model
        self.laminput = laminput
        self.dmodel = distrib.wrap(model)
        self.optimizer = optimizer

        # data augment
        augments = []
        if args.remix:
            augments.append(augment.Remix())
        if args.bandmask:
            augments.append(augment.BandMask(args.bandmask, sample_rate=args.sample_rate))
        if args.shift:
            augments.append(augment.Shift(args.shift, args.shift_same))
        if args.revecho:
            augments.append(
                augment.RevEcho(args.revecho))
        self.augment = torch.nn.Sequential(*augments)

        # Training config
        self.device = args.device
        self.epochs = args.epochs

        # Checkpoints
        self.continue_from = args.continue_from
        self.eval_every = args.eval_every
        self.checkpoint = args.checkpoint
        if self.checkpoint:
            self.checkpoint_file = Path(args.checkpoint_file)
            self.best_file = Path(args.best_file)
            logger.debug("Checkpoint will be saved to %s", self.checkpoint_file.resolve())
        self.history_file = args.history_file

        self.best_state = None
        self.restart = args.restart
        self.history = []  # Keep track of loss
        self.samples_dir = args.samples_dir  # Where to save samples
        self.num_prints = args.num_prints  # Number of times to log per epoch
        self.args = args
        self.mrstftloss = MultiResolutionSTFTLoss(factor_sc=args.stft_sc_factor,
                                                  factor_mag=args.stft_mag_factor).to(self.device)
 

    def train(self):
        enhance(self.args, self.model, self.laminput, self.samples_dir)



