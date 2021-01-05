#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""extract feats using lhotse"""

import argparse
from distutils.util import strtobool
from lhotse.kaldi import load_kaldi_data_dir
from lhotse.features.io import LilcomFilesWriter
from lhotse.dataset import SpeechRecognitionDataset
from lhotse import CutSet, LilcomFilesWriter, FeatureSetBuilder, FeatureSet, RecordingSet, SupervisionSet
from lhotse.dataset.speech_recognition import K2DataLoader, K2SpeechRecognitionDataset, \
    K2SpeechRecognitionIterableDataset
from lhotse.dataset.unsupervised import UnsupervisedDataset, UnsupervisedWaveformDataset
from lhotse.dataset.vad import VadDataset
from lhotse.dataset.diarization import DiarizationDataset
from lhotse import Fbank, Mfcc, Spectrogram
from lhotse import features
import subprocess
import torch
import matplotlib.pyplot as plt
from typing import Dict
import numpy as np
import sys
import warnings
def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
    
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str,
                    help='path to data folder')
parser.add_argument('--sample_rate', type=int, default=16000,
                    help='Sample Rate of the audio dataset')
parser.add_argument('--feature_type', type=str, default=None,
                    choices=['fbank', 'spectrogram', 'mfcc'],
                    help='type of extracted feature, only valid when input_type is speech')
parser.add_argument('--cmvn', type=str, default=None,
                    choices=['global', 'perinstance', 'online'],
                    help='type of extracted feature, only valid when input_type is speech')
parser.add_argument('--feature_config', type=str, default=False,
                    help='config file for lhotse feature extraction')
parser.add_argument('--feature_dump_location', type=str, default=False,
                    help='path to dump extracted features')
parser.add_argument('--cutset_yaml', type=str, default=False,
                    help='path to cutset yaml file')
parser.add_argument('--jobs', type=int, default=50,
                    help='number of jobs to process feature in parallel')
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
args = parser.parse_args()

def main():
    
    recording_set, supervision_set= load_kaldi_data_dir(args.data, args.sample_rate)
    print("Done Generating Recording and Supervision manifest from (%s)"%args.data)

    if args.feature_type == None:
        print("Based on the settings, features are not extracted for dataset %s"%dataset)\

    if args.feature_type == 'fbank':
        from lhotse import Fbank as feature_pipeline
    elif args.feature_type == 'spectrogram':
        from lhotse import Spectrogram as feature_pipeline
    elif args.feature_type == 'mfcc':
        from lhotse import Spectrogram as feature_pipeline
    else:
        print("%s is not a valid option for feature. Fbank is picked as feature type.")
        from lhotse import Fbank as feature_pipeline

    if args.feature_config == "":
        feature_extractor = feature_pipeline()
    else:
        feature_extractor = feature_pipeline.from_yaml(args.feature_config)
    
    # Create a feature set builder that uses this extractor and dumps the results in specified location
    with LilcomFilesWriter(args.feature_dump_location) as storage:
        builder = FeatureSetBuilder(feature_extractor=feature_extractor, storage=storage)
        # Extract the features using 8 parallel processes, compress, and store them on in 'features/storage/' directory.
        # Then, return the feature manifest object, which is also compressed and
        # stored in 'features/feature_manifest.json.gz'
        feature_set = builder.process_and_store_recordings(
            recordings=recording_set,
            num_jobs=args.jobs
        )
    print("Done Extracting feature in type (%s)"%(args.feature_type))
    
    cut_set = CutSet.from_manifests(recordings=recording_set, features=feature_set, supervisions=supervision_set)
    cut_set = cut_set.trim_to_supervisions()
    cut_set.to_yaml(args.cutset_yaml)

    print("Done Generating Cut manifests save at (%s)"%args.cutset_yaml)

if __name__ == '__main__':
    main()
