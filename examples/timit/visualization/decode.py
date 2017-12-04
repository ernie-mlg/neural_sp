#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""Decode the trained model's outputs (TIMIT corpus)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join, abspath
import sys
import yaml
import argparse

sys.path.append(abspath('../../../'))
from models.pytorch.load_model import load
from examples.timit.data.load_dataset import Dataset
from utils.io.labels.phone import Idx2phone
from utils.io.variable import np2var, var2np

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str,
                    help='path to the model to evaluate')
parser.add_argument('--epoch', type=int, default=-1,
                    help='the epoch to restore')
parser.add_argument('--beam_width', type=int, default=1,
                    help='beam_width (int, optional): beam width for beam search.' +
                    ' 1 disables beam search, which mean greedy decoding.')
parser.add_argument('--eval_batch_size', type=int, default=1,
                    help='the size of mini-batch in evaluation')
parser.add_argument('--max_decode_length', type=int, default=40,
                    help='the length of output sequences to stop prediction when EOS token have not been emitted')


def main():

    args = parser.parse_args()

    # Load config file
    with open(join(args.model_path, 'config.yml'), "r") as f:
        config = yaml.load(f)
        params = config['param']

    # Get voabulary number (excluding blank, <SOS>, <EOS> classes)
    with open('../metrics/vocab_num.yml', "r") as f:
        vocab_num = yaml.load(f)
        params['num_classes'] = vocab_num[params['label_type']]

    # Load model
    model = load(model_type=params['model_type'], params=params)

    # Load dataset
    vocab_file_path = '../metrics/vocab_files/' + params['label_type'] + '.txt'
    test_data = Dataset(
        model_type=params['model_type'],
        data_type='test', label_type=params['label_type'],
        vocab_file_path=vocab_file_path,
        batch_size=args.eval_batch_size, splice=params['splice'],
        num_stack=params['num_stack'], num_skip=params['num_skip'],
        sort_utt=True, reverse=True,
        use_cuda=model.use_cuda, volatile=True,
        save_format=params['save_format'])

    # GPU setting
    model.set_cuda(deterministic=False)

    # Restore the saved model
    checkpoint = model.load_checkpoint(
        save_path=args.model_path, epoch=args.epoch)
    model.load_state_dict(checkpoint['state_dict'])

    # Change to evaluation mode
    model.eval()

    # Visualize
    decode(model=model,
           model_type=params['model_type'],
           dataset=test_data,
           label_type=params['label_type'],
           beam_width=args.beam_width,
           max_decode_length=args.max_decode_length,
           save_path=None)
    # save_path=model.save_path)


def decode(model, model_type, dataset, label_type, beam_width,
           max_decode_length=40, save_path=None):
    """Visualize label outputs.
    Args:
        model: the model to evaluate
        model_type (string): ctc or attention
        dataset: An instance of a `Dataset` class
        label_type (string): phone39 or phone48 or phone61
        beam_width: (int): the size of beam
        max_decode_length (int, optional): the length of output sequences
            to stop prediction when EOS token have not been emitted.
            This is used for seq2seq models.
        save_path (string): path to save decoding results
    """
    idx2phone = Idx2phone('../metrics/vocab_files/' + label_type + '.txt')

    if save_path is not None:
        sys.stdout = open(join(model.model_dir, 'decode.txt'), 'w')

    for data, is_new_epoch in dataset:

        # Create feed dictionary for next mini batch
        inputs, labels, inputs_seq_len, labels_seq_len, input_names = data

        # Decode
        labels_pred, perm_indices = model.decode(
            inputs, inputs_seq_len,
            beam_width=beam_width,
            max_decode_length=max_decode_length)

        for i_batch in range(inputs.size(0)):
            print('----- wav: %s -----' % input_names[i_batch])

            ##############################
            # Reference
            ##############################
            if dataset.is_test:
                str_true = labels[i_batch][0]
                # NOTE: transcript is seperated by space(' ')
            else:
                # Permutate indices
                labels = var2np(labels[perm_indices])
                labels_seq_len = var2np(labels_seq_len[perm_indices])

                # Convert from list of index to string
                if model_type == 'ctc':
                    str_true = idx2phone(
                        labels[i_batch][:labels_seq_len[i_batch]])
                elif model_type == 'attention':
                    str_true = idx2phone(
                        labels[i_batch][1:labels_seq_len[i_batch] - 1])
                    # NOTE: Exclude <SOS> and <EOS>

            ##############################
            # Hypothesis
            ##############################
            # Convert from list of index to string
            str_pred = idx2phone(labels_pred[i_batch])

            if model_type == 'attention':
                str_pred = str_pred.split('>')[0]
                # NOTE: Trancate by the first <EOS>

                # Remove the last space
                if len(str_pred) > 0 and str_pred[-1] == ' ':
                    str_pred = str_pred[:-1]

            print('Ref: %s' % str_true)
            print('Hyp: %s' % str_pred)

        if is_new_epoch:
            break


if __name__ == '__main__':
    main()