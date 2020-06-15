from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import time

import tensorflow as tf
from tqdm import tqdm

from src.DataLoader import GetDataset
from src.layers.Encoders import RNNEncoder
from src.layers.Decoders import RNNDecoder
from src.utils.Optimizers import LazyAdam
from src.utils.model_utils import CustomSchedule, _set_up_dirs


def _train_rnn(args):
  # set up dirs
  (OUTPUT_DIR, EvalResultsFile,
   TestResults, log_file, log_dir) = _set_up_dirs(args)

  OUTPUT_DIR += '/{}_{}'.format(args.enc_type, args.dec_type)

  dataset, eval_set, test_set, BUFFER_SIZE, BATCH_SIZE, \
  steps_per_epoch, src_vocab_size, vocab, dataset_size, max_seq_len = GetDataset(args)

  if args.epochs is not None:
    steps = args.epochs * steps_per_epoch
  else:
    steps = args.steps

  # Save model parameters for future use
  if os.path.isfile('{}/{}_{}_params'.format(log_dir, args.lang, args.model)):
    with open('{}/{}_{}_params'.format(log_dir, args.lang, args.model), 'rb') as fp:
      PARAMS = pickle.load(fp)
      print('Loaded Parameters..')
  else:
    if not os.path.isdir(log_dir):
      os.makedirs(log_dir)
    PARAMS = {
      "args": args,
      "vocab_size": src_vocab_size,
      "dataset_size": dataset_size,
      "max_tgt_length": max_seq_len,
      "step": 0
    }

  if args.decay is not None:
    learning_rate = CustomSchedule(args.emb_dim, warmup_steps=args.decay_steps)
    optimizer = LazyAdam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
  else:
    optimizer = LazyAdam(learning_rate=args.learning_rate,
                         beta_1=0.9, beta_2=0.98, epsilon=1e-9)

  encoder = RNNEncoder(src_vocab_size, args.emb_dim,
                       args.enc_units, args.batch_size)
  decoder = RNNDecoder(src_vocab_size, args.emb_dim,
                       args.enc_units, args.batch_size)

  ckpt = tf.train.Checkpoint(
    encoder=encoder,
    decoder=decoder,
    optimizer=optimizer
  )
  ckpt_manager = tf.train.CheckpointManager(ckpt, OUTPUT_DIR, max_to_keep=5)
  if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')

  if args.learning_rate is not None:
    optimizer._lr = args.learning_rate

  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

  def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

  def train_step(inp, targ, enc_hidden):
    loss = 0
    with tf.GradientTape() as tape:
      enc_output, enc_hidden = encoder(inp, enc_hidden)
      dec_hidden = enc_hidden
      if args.sentencepiece == 'True':
        dec_input = tf.expand_dims([vocab.PieceToId('<start>')] * BATCH_SIZE, 1)
      else:
        dec_input = tf.expand_dims([vocab.word_index['<start>']] * BATCH_SIZE, 1)

      # Teacher forcing - feeding the target as the next input
      for t in range(1, targ.shape[1]):
        # passing enc_output to the decoder
        predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
        loss += loss_function(targ[:, t], predictions)
        # using teacher forcing
        dec_input = tf.expand_dims(targ[:, t], 1)
    batch_loss = (loss / int(targ.shape[1]))
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

  def test_step():
    encoder.trainable = False
    decoder.trainable = False
    results = []

    for (batch, (inp)) in tqdm(enumerate(test_set)):
      enc_hidden = tf.zeros((1, args.enc_units))
      enc_hidden = [enc_hidden, enc_hidden]
      inp = tf.expand_dims(inp, axis=0)
      enc_output, enc_hidden = encoder(inp, enc_hidden)
      dec_hidden = enc_hidden
      if args.sentencepiece == 'True':
        dec_input = tf.expand_dims([vocab.PieceToId('<start>')], 1)
      else:
        dec_input = tf.expand_dims([vocab.word_index['<start>']], 1)
      result = ''
      ids = []
      for t in range(max_seq_len):
        predictions, dec_hidden, _ = decoder(dec_input,
                                             dec_hidden,
                                             enc_output)
        predicted_id = (int(tf.argmax(predictions[0]).numpy()))
        ids.append(predicted_id)
        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)
        # logger.info(dec_input.shape)

      results.append(vocab.DecodeIds(ids))
    with open(TestResults, 'w+') as f:
      f.write('\n'.join(results))

  for (batch, (inp, tgt)) in tqdm(enumerate(dataset.repeat(-1))):
    if PARAMS['step'] < steps:
      start = time.time()
      PARAMS['step'] += 1

      if args.decay is not None:
        optimizer._lr = learning_rate(tf.cast(PARAMS['step'], dtype=tf.float32))

      enc_hidden = tf.zeros((args.batch_size, args.enc_units))
      enc_hidden = [enc_hidden, enc_hidden]
      batch_loss = train_step(inp, tgt, enc_hidden)
      if batch % 100 == 0:
        print('Step {} Learning Rate {:.4f} Train Loss {:.4f} '.format(PARAMS['step'],
                                                                       optimizer._lr,
                                                                       batch_loss))
        print('Time {} \n'.format(time.time() - start))

      if batch % args.checkpoint == 0:
        print("Saving checkpoint \n")
        ckpt_save_path = ckpt_manager.save()
      with open(log_dir + '/' + args.lang + '_' + args.model + '_params', 'wb+') as fp:
        pickle.dump(PARAMS, fp)
    else:
      break

  test_step()
