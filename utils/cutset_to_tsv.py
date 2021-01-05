#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Make a dataset tsv file."""

import argparse
import codecs
from distutils.util import strtobool
from lhotse import CutSet
import os
import re
import sentencepiece as spm
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--cutset_yaml', type=str, default='', nargs='?',
                    help='path to cutset yaml file')
parser.add_argument('--dict', type=str,
                    help='dictionary file')
parser.add_argument('--unit', type=str,
                    choices=['word', 'wp', 'char', 'phone', 'word_char'],
                    help='token units')
parser.add_argument('--remove_space', type=strtobool, default=False,
                    help='')
parser.add_argument('--unk', type=str, default='<unk>',
                    help='<unk> token')
parser.add_argument('--space', type=str, default='<space>',
                    help='<space> token')
parser.add_argument('--nlsyms', type=str, default='', nargs='?',
                    help='path to non-linguistic symbols, e.g., [noise] etc.')
parser.add_argument('--wp_model', type=str, default=False, nargs='?',
                    help='prefix of the wordpiece model')
parser.add_argument('--wp_nbest', type=int, default=1, nargs='?',
                    help='')
args = parser.parse_args()


def main():    
    
    cutset = CutSet.from_yaml(args.cutset_yaml)
    
    nlsyms = []
    if args.nlsyms:
        with codecs.open(args.nlsyms, 'r', encoding="utf-8") as f:
            for line in f:
                nlsyms.append(line.strip())

    token2idx = {}
    idx2token = {}
    with codecs.open(args.dict, 'r', encoding="utf-8") as f:
        for line in f:
            token, idx = line.strip().split(' ')
            token2idx[token] = str(idx)
            idx2token[str(idx)] = token

    if args.unit == 'wp':
        sp = spm.SentencePieceProcessor()
        sp.Load(args.wp_model + '.model')

    print('utt_id\tspeaker\tfeat_path\txlen\txdim\ttext\ttoken_id\tylen\tydim\tcut_id\tprev_utt')

    xdim = None
    pbar = tqdm(total=len(cutset))
    utt2featpath = {}
    utt2num_frames = {}
    utt2spk = {}
    for cut in cutset:
        # Remove successive spaces
        cut_id = cut.id
        utt_id = cut.supervisions[0].id
        feat_path = os.path.join(cut.features.storage_path, cut.features.storage_key)        
        xlen,xdim = cut.load_features().shape
        speaker = cut.supervisions[0].speaker

        line = cut.supervisions[0].text
        line = re.sub(r'[\s]+', ' ', line.strip())
        words = line.split(' ')
        if '' in words:
            words.remove('')
        text = ' '.join(words)

        # Skip empty line
        if text == '':
            continue

        # Convert strings into the corresponding indices
        token_ids = []
        if args.unit in ['word', 'word_char']:
            for w in words:
                if w in token2idx.keys():
                    token_ids.append(token2idx[w])
                else:
                    # Replace with <unk>
                    if args.unit == 'word_char':
                        for c in list(w):
                            if c in token2idx.keys():
                                token_ids.append(token2idx[c])
                            else:
                                token_ids.append(token2idx[args.unk])
                    else:
                        token_ids.append(token2idx[args.unk])

        elif args.unit == 'wp':
            # Remove space before the first special symbol
            wps = sp.EncodeAsPieces(text)
            if wps[0] == '▁' and wps[1][0] == '<':
                wps = wps[1:]

            for wp in wps:
                if wp in token2idx.keys():
                    token_ids.append(token2idx[wp])
                else:
                    # Replace with <unk>
                    token_ids.append(token2idx[args.unk])

        elif args.unit == 'char':
            for i, w in enumerate(words):
                if w in nlsyms:
                    token_ids.append(token2idx[w])
                else:
                    for c in list(w):
                        if c in token2idx.keys():
                            token_ids.append(token2idx[c])
                        else:
                            # Replace with <unk>
                            token_ids.append(token2idx[args.unk])

                # Insert <space> mark
                if not args.remove_space:
                    if i < len(words) - 1:
                        token_ids.append(token2idx[args.space])

        elif args.unit == 'phone':
            for p in words:
                token_ids.append(token2idx[p])

        else:
            raise ValueError(args.unit)
        token_id = ' '.join(token_ids)
        ylen = len(token_ids)
        ydim = len(token2idx.keys())

        print('%s\t%s\t%s\t%d\t%d\t%s\t%s\t%d\t%d\t%s' %
              (utt_id, speaker, feat_path, xlen, xdim, text, token_id, ylen, ydim, cut_id))

        # data augmentation for wordpiece
        if args.unit == 'wp' and args.wp_nbest > 1:
            raise NotImplementedError

            for wp_i in sp.NBestEncodeAsPieces(text, args.wp_nbest)[1:]:
                if wp_i in token2idx.keys():
                    token_ids = token2idx[wp_i]
                else:
                    # Replace with <unk>
                    token_ids = token2idx[args.unk]

                token_id = ' '.join(token_ids)
                ylen = len(token_ids)

                print('%s\t%s\t%s\t%d\t%d\t%s\t%s\t%d\t%d\t%s' %
                      (utt_id, speaker, feat_path, xlen, xdim, text, token_id, ylen, ydim, cut_id))

        pbar.update(1)


if __name__ == '__main__':
    main()
