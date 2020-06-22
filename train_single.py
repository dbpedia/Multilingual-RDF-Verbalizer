""" Script to train the selected model
    Used to train a single language model ( Teacher model ) 
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from src.arguments import get_args
from src.trainers.GATtrainer import _train_gat_trans
from src.trainers.RNNtrainer import _train_rnn
from src.trainers.TransformerTrainer import _train_transformer
import tensorflow as tf
import numpy as np
import random

if __name__ == "__main__":
  args = get_args()
  global step

if args.seed is not None:
  tf.reset_default_graph()
  tf.set_random_seed(args.seed)
  random.seed(args.seed)
  tf.random.set_seed(args.seed)
  np.random.seed(args.seed)

if args.enc_type == 'rnn' and args.dec_type == "rnn":
  _train_rnn(args)

elif args.enc_type == 'transformer' and args.dec_type == "transformer":
  _train_transformer(args)

elif ((args.enc_type == "gat") and (args.dec_type == "transformer")):
  _train_gat_trans(args)
