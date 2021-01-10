#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""extract feats using lhotse"""

import pickle
import lilcom
import argparse
import warnings
from pathlib import Path
from tqdm.auto import tqdm
from functools import partial, reduce
from lhotse.manipulation import combine
from lhotse.kaldi import load_kaldi_data_dir
from lhotse import CutSet, features, LilcomFilesWriter, FeatureSetBuilder

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
parser.add_argument('--storage_type', type=str, default="Lilcom",
                    choices=['Lilcom', 'Numpy'],
                    help='feature storage storge')
parser.add_argument('--cmvn', type=str, default=None,
                    choices=['Save_Apply', 'Load_Apply'],
                    help='Calculate/Save or Load cmvn stats and apply')
parser.add_argument('--cmvn_location', type=str, default=None,
                    help='Location to load or save cmvn stats.')
parser.add_argument('--parallel', type=str, default="process",
                    choices=['process', 'thread'],
                    help='choice of parallelize execution of a Python method')
parser.add_argument('--speed_perturb', action='store_true',
                    help='data will be augmented with speed perturbation 0.9 abd 1.1')
parser.add_argument('--feature_config', type=str, default=False,
                    help='config file for lhotse feature extraction')
parser.add_argument('--feature_dump_location', type=str, default=False,
                    help='path to dump extracted features')
parser.add_argument('--cutset_yaml', type=str, default=False,
                    help='path to cutset yaml file')
parser.add_argument('--jobs', type=int, default=10,
                    help='number of jobs to process feature in parallel')

parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
args = parser.parse_args()

def main():
    
    recording_set, supervision_set= load_kaldi_data_dir(args.data, args.sample_rate)    
    print("Done Generating Recording and Supervision manifest from (%s)"%args.data)
    
    cut_set = CutSet.from_manifests(recordings=recording_set, supervisions=supervision_set)
    if (args.speed_perturb):
        cut_set = cut_set + cut_set.perturb_speed(0.9) + cut_set.perturb_speed(1.1)

    jobs = min(len(cut_set), args.jobs)

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

    if args.parallel == "process":
        from concurrent.futures import ProcessPoolExecutor
        ex = ProcessPoolExecutor(jobs)
    elif args.parallel == "thread":
        from concurrent.futures import ThreadPoolExecutor
        ex = ThreadPoolExecutor(jobs)
        
    if args.storage_type == "Lilcom":
        from lhotse.features.io import LilcomFilesWriter as storage_type
        import lilcom
    elif args.storage_type == "Numpy":
        from lhotse.features.io import NumpyFilesWriter as storage_type
        import numpy as np

        
    if args.feature_config == "":
        feature_extractor = feature_pipeline()
    else:
        feature_extractor = feature_pipeline.from_yaml(args.feature_config)

    
        
    cut_set = cut_set.compute_and_store_features(
                extractor=feature_extractor,
                storage_path=args.feature_dump_location,
                storage_type=storage_type,
                executor=ex,
                num_jobs=jobs
            )
    print("Done Extracting feature in type (%s)"%(args.feature_type))
    
    if ("Save" in args.cmvn):
        stats = cut_set.compute_global_feature_stats(storage_path = args.cmvn_location)
        print("Done Computing CMVN stats from this feature set")
    elif ("Load" in args.cmvn):
        with open(args.cmvn_location, 'rb') as f:
             stats=pickle.load(f)
    
    if ("Apply" in args.cmvn):
        futures = [
            ex.submit(
                Apply_cmvn_on_cut,
                cs,
                stats,
                args.storage_type
            )
            for i, cs in enumerate(cut_set)
        ]
        
        progress = partial(
                tqdm, desc='Applying CMVN on extacted features (chunks progress)', total=len(futures)
            )
        
        cut_set = combine(progress(f.result() for f in futures))

    if args.parallel == "process":
        ex = ProcessPoolExecutor(args.jobs)
    elif args.parallel == "thread":
        ex = ThreadPoolExecutor(args.jobs)

    cut_set = cut_set.trim_to_supervisions()
    cut_set.to_yaml(args.cutset_yaml)

    print("Done Generating Cut manifests save at (%s)"%args.cutset_yaml)

def Apply_cmvn_on_cut(cut, stats, storage_type):
    feature = cut.load_features()
    feature_cmvn = (feature - stats["norm_means"])/stats["norm_stds"]
    storage_key = cut.features.storage_key[4:]
    new_storage_path_ = Path(cut.features.storage_path+"_cmvn")
    new_storage_path_.mkdir(parents=True, exist_ok=True)
    subdir = new_storage_path_ / storage_key[:3]
    subdir.mkdir(exist_ok=True)
    if storage_type == "Lilcom":
        output_features_path = (subdir / storage_key).with_suffix('.llc')
        serialized_feats = lilcom.compress(feature_cmvn, tick_power=-5)
        with open(output_features_path, 'wb') as f:
            f.write(serialized_feats)
    elif storage_type == "Numpy":
        output_features_path = (subdir / storage_key).with_suffix('.npy')
        np.save(output_features_path, feature_cmvn, allow_pickle=False)
    cut.features.storage_path = cut.features.storage_path+"_cmvn"
    return CutSet.from_cuts([cut])
    
    
if __name__ == '__main__':
    main()
