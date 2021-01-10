#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Make a dataset tsv file."""

import argparse
import codecs
import os
import re
from tqdm import tqdm
from lhotse import CutSet
import sentencepiece as spm
from distutils.util import strtobool
from functools import partial, reduce
from lhotse import NumpyFilesReader
from lhotse.manipulation import combine
from lhotse.utils import compute_num_frames


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
parser.add_argument('--tsv_location', type=str, default="",
                    help='output location for tsv file')
parser.add_argument('--jobs', type=int, default=10,
                    help='number of jobs to process feature in parallel')
parser.add_argument('--parallel', type=str, default="process",
                    choices=['process', 'thread'],
                    help='choice of parallelize execution of a Python method')
args = parser.parse_args()


def main():    
    
    cutset = CutSet.from_yaml(args.cutset_yaml)
    
    if args.parallel == "process":
        from concurrent.futures import ProcessPoolExecutor
        ex = ProcessPoolExecutor(args.jobs)
    elif args.parallel == "thread":
        from concurrent.futures import ThreadPoolExecutor
        ex = ThreadPoolExecutor(args.jobs)
    
    nlsyms = []
    if args.nlsyms:
        with codecs.open(args.nlsyms, 'r', encoding="utf-8") as f:
            for line in f:
                nlsyms.append(line.strip())

    token2idx = {}
    idx2token = {}
    sp = None
    
    with codecs.open(args.dict, 'r', encoding="utf-8") as f:
        for line in f:
            token, idx = line.strip().split(' ')
            token2idx[token] = str(idx)
            idx2token[str(idx)] = token

    if args.unit == 'wp':
        sp = spm.SentencePieceProcessor()
        sp.Load(args.wp_model + '.model')
        

    print_lines = ['utt_id\tspeaker\tfeat_path\txlen\txdim\ttext\ttoken_id\tylen\tydim\toffset\tprev_utt']
    
    futures = [
        ex.submit(
            Cut2Tsv,
            cs,
            token2idx,
            idx2token,
            nlsyms,
            sp, 
            args
        )
        for i, cs in enumerate(cutset)
    ]
    
    progress = partial(
            tqdm, desc='Converting cutset into tsv data', total=len(futures)
        )
    
    print_lines += combine(progress(f.result() for f in futures))
    
    out = open(args.tsv_location, "w")
    for line in print_lines:
        if line == "":
            continue
        out.write(line + "\n")
    out.close()

def Cut2Tsv(cut, token2idx, idx2token, nlsyms, sp, args):
    cut_id = cut.id
    utt_id = cut.supervisions[0].id
    feat_path = cut.features.storage_path  + "@" + cut.features.storage_key
    xlen,xdim = cut.load_features().shape
    speaker = cut.supervisions[0].speaker
    
    offset = cut.start
    duration = cut.duration
    full_duration= cut.features.duration
    num_frames = cut.features.num_frames
    frame_shift = round(full_duration / num_frames, ndigits=3)
    left_offset_frames = compute_num_frames(offset, frame_shift)
    right_offset_frames = left_offset_frames + compute_num_frames(duration, frame_shift)

    line = cut.supervisions[0].text
    line = re.sub(r'[\s]+', ' ', line.strip())
    words = line.split(' ')
    if '' in words:
        words.remove('')
    text = ' '.join(words)

    # Skip empty line
    if text == '':
        return [""]

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

    print_line = '%s\t%s\t%s\t%d\t%d\t%s\t%s\t%d\t%d\t%d'%(utt_id, speaker, feat_path, xlen, xdim, text, token_id, ylen, ydim, left_offset_frames)

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

            print_line = '%s\t%s\t%s\t%d\t%d\t%s\t%s\t%d\t%d\t%d\t%d'%(utt_id, speaker, feat_path, xlen, xdim, text, token_id, ylen, ydim, left_offset_frames, right_offset_frames)
    return [print_line]

if __name__ == '__main__':
    main()
