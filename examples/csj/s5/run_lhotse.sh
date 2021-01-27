#!/bin/bash

# Copyright 2018 Kyoto University (Hirofumi Inaguma)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

echo ============================================================================
echo "                                   CSJ                                    "
echo ============================================================================

stage=0
stop_stage=5
gpu=3
benchmark=true
speed_perturb=true
stdout=false

### vocabulary
unit=wp      # word/wp/char/word_char/phone
vocab=10000
wp_type=bpe  # bpe/unigram (for wordpiece)

#########################
# feature configuration
#########################
sample_rate=16000
feature_type=fbank
feature_conf=conf/fbank_lhotse.yaml
jobs=50

#########################
# ASR configuration
#########################
conf=conf/asr/blstm_las.yaml
conf2=
asr_init=
external_lm=

#########################
# LM configuration
#########################
remove_space=false
unk="<unk>"
space="<space>"
nlsyms=""
wp_model=""
wp_nbest=1
lm_conf=conf/lm/rnnlm.yaml

### path to save the model
model=./exp

### path to the model directory to resume training
resume=
lm_resume=

### path to save preproecssed data
data=./data

### path to original data
CSJDATATOP=/home/yyu/CSJ_RAW/  ## CSJ database top directory.
CSJVER=usb  ## Set your CSJ format (dvd or usb).
## Usage    :
## Case DVD : We assume CSJ DVDs are copied in this directory with the names dvd1, dvd2,...,dvd17.
##            Neccesary directory is dvd3 - dvd17.
##            e.g. $ ls ${CSJDATATOP}(DVD) => 00README.txt dvd1 dvd2 ... dvd17
##
## Case USB : Neccesary directory is MORPH/SDB and WAV
##            e.g. $ ls ${CSJDATATOP}(USB) => 00README.txt DOC MORPH ... WAV fileList.csv
## Case merl :MERL setup. Neccesary directory is WAV and sdb

### data size
datasize=all
lm_datasize=all  # default is the same data as ASR
# NOTE: aps_other=default using "Academic lecture" and "other" data,
#       aps=using "Academic lecture" data,
#       sps=using "Academic lecture" data,
#       all_except_dialog=using All data except for "dialog" data,
#       all=using All data

. ./cmd.sh
. ./path.sh
. utils/parse_options.sh

set -e
set -u
set -o pipefail

if [ -z ${gpu} ]; then
    echo "Error: set GPU number." 1>&2
    echo "Usage: ./run.sh --gpu 0" 1>&2
    exit 1
fi
n_gpus=$(echo ${gpu} | tr "," "\n" | wc -l)

train_set=train_nodev_${datasize}
dev_set=dev_${datasize}
test_set="eval1 eval2 eval3"
test_single_set="eval1"
if [ ${speed_perturb} = true ]; then
    train_set=train_nodev_sp_${datasize}
    dev_set=dev_sp_${datasize}
fi

if [ ${unit} = char ] || [ ${unit} = phone ]; then
    vocab=
fi
if [ ${unit} != wp ]; then
    wp_type=
fi

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ] && [ ! -e ${data}/.done_stage_0_${datasize} ]; then
    echo ============================================================================
    echo "                       Data Preparation (stage:0)                          "
    echo ============================================================================

    mkdir -p ${data}
    local/csj_make_trans/csj_autorun.sh ${CSJDATATOP} ${data}/csj-data ${CSJVER} || exit 1;
    local/csj_data_prep.sh ${data}/csj-data ${data} ${datasize} || exit 1;
    for x in eval1 eval2 eval3; do
        local/csj_eval_data_prep.sh ${data}/csj-data/eval ${data} ${x} || exit 1;
        utils/data/get_reco2dur.sh ${data}/${x} || exit 1;
    done

    # Remove <sp> and POS tag, and lowercase
    for x in train_${datasize} ${test_set}; do
        local/remove_pos.py ${data}/${x}/text | nkf -Z > ${data}/${x}/text.tmp
        mv ${data}/${x}/text.tmp ${data}/${x}/text
    done
    
    # Use the first 4k sentences from training data as dev set. (39 speakers.)
    utils/subset_data_dir.sh --first ${data}/train_${datasize} 4000 ${data}/${dev_set} || exit 1;  # 6hr 31min
    n=$[$(cat ${data}/train_${datasize}/segments | wc -l) - 4000]
    utils/subset_data_dir.sh --last ${data}/train_${datasize} ${n} ${data}/${train_set}.tmp || exit 1;

    # Finally, the full training set:
    utils/data/remove_dup_utts.sh 300 ${data}/${train_set}.tmp ${data}/${train_set} || exit 1;  # 233hr 36min
    rm -rf ${data}/*.tmp
    utils/data/get_reco2dur.sh  ${data}/${train_set} || exit 1;
    utils/data/get_reco2dur.sh  ${data}/${dev_set} || exit 1;

    touch ${data}/.done_stage_0_${datasize} && echo "Finish data preparation (stage: 0)."
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ] && [ ! -e ${data}/.done_stage_1_${datasize}_sp${speed_perturb} ]; then
    echo ============================================================================
    echo "                    Feature extranction (stage:1)                          "
    echo ============================================================================

    if [ ! -e ${data}/.done_stage_1_${datasize}_spfalse ]; then
        x=${train_set}
        python -W ignore ${NEURALSP_ROOT}/utils/extract_feats.py \
            --data ${data}/${x} \
            --sample_rate ${sample_rate} \
            --feature_type ${feature_type} \
            --feature_config ${feature_conf} \
            --feature_dump_location ${data}/${feature_type}/${x} \
            --jobs ${jobs} \
            --cutset_yaml ${data}/${x}/cutset.yaml \
            --storage_type Lilcom \
            --cmvn Save_Apply \
            --cmvn_location ${data}/${feature_type}/${x}/stats.npy \
            --speed_perturb

        for x in ${dev_set} ${test_set}; do
            mkdir -p ${data}/${feature_type}/${x}
            python -W ignore ${NEURALSP_ROOT}/utils/extract_feats.py \
            --data ${data}/${x} \
            --sample_rate ${sample_rate} \
            --feature_type ${feature_type} \
            --feature_config ${feature_conf} \
            --feature_dump_location ${data}/${feature_type}/${x} \
            --jobs ${jobs} \
            --cutset_yaml ${data}/${x}/cutset.yaml \
            --storage_type Lilcom \
            --cmvn Load_Apply \
            --cmvn_location ${data}/${feature_type}/${train_set}/stats.npy
        done
    fi

    touch ${data}/.done_stage_1_${datasize}_sp${speed_perturb} && echo "Finish feature extranction (stage: 1)."
fi

dict=${data}/dict/${train_set}_${unit}${wp_type}${vocab}.txt; mkdir -p ${data}/dict
wp_model=${data}/dict/${train_set}_${wp_type}${vocab}
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ] && [ ! -e ${data}/.done_stage_2_${datasize}_${unit}${wp_type}${vocab}_sp${speed_perturb} ]; then
    echo ============================================================================
    echo "                      Dataset preparation (stage:2)                        "
    echo ============================================================================

    if [ ${unit} = wp ]; then
        make_vocab.sh --unit ${unit} --character_coverage 0.9995 \
            --vocab ${vocab} --wp_type ${wp_type} --wp_model ${wp_model} \
            ${data} ${dict} ${data}/${train_set}/text || exit 1;
    elif [ ${unit} = phone ]; then
        lexicon=${data}/local/train_${datasize}/lexicon.txt
        for x in ${train_set} ${dev_set} ${test_set}; do
            map2phone.py --text ${data}/${x}/text --lexicon ${lexicon} > ${data}/${x}/text.phone
        done
        make_vocab.sh --unit ${unit}\
            ${data} ${dict} ${data}/${train_set}/text.phone || exit 1;
    else
        # character
        make_vocab.sh --unit ${unit} --character_coverage 0.9995 \
            ${data} ${dict} ${data}/${train_set}/text || exit 1;
    fi

    # Compute OOV rate
    if [ ${unit} = word ]; then
        mkdir -p ${data}/dict/word_count ${data}/dict/oov_rate
        echo "OOV rate:" > ${data}/dict/oov_rate/word${vocab}_${datasize}.txt
        for x in ${train_set} ${dev_set} ${test_set}; do
            cut -f 2- -d " " ${data}/${x}/text | tr " " "\n" | sort | uniq -c | sort -n -k1 -r \
                > ${data}/dict/word_count/${x}_${datasize}.txt || exit 1;
            compute_oov_rate.py ${data}/dict/word_count/${x}_${datasize}.txt ${dict} ${x} \
                >> ${data}/dict/oov_rate/word${vocab}_${datasize}.txt || exit 1;
            # NOTE: speed perturbation is not considered
        done
        cat ${data}/dict/oov_rate/word${vocab}_${datasize}.txt
    fi

    echo "Making dataset tsv files for ASR ..."
    mkdir -p ${data}/dataset
    if [ ${unit} = phone ]; then
        text="text.phone"
    else
        text="text"
    fi
    
    for x in ${test_set}; do
        python -W ignore ${NEURALSP_ROOT}/utils/cutset_to_tsv.py \
                        --unit ${unit} \
                        --dict ${dict} \
                        --remove_space ${remove_space} \
                        --unk ${unk} \
                        --space ${space} \
                        --nlsyms ${nlsyms} \
                        --wp_model ${wp_model} \
                        --wp_nbest ${wp_nbest} \
                        --jobs ${jobs} \
                        --cutset_yaml ${data}/${x}/cutset.yaml \
                        --tsv_location ${data}/dataset/${x}_${datasize}_${unit}${wp_type}${vocab}.tsv || exit 1;
    done
    
    for x in ${train_set} ${dev_set}; do
        python -W ignore ${NEURALSP_ROOT}/utils/cutset_to_tsv.py \
                        --unit ${unit} \
                        --dict ${dict} \
                        --remove_space ${remove_space} \
                        --unk ${unk} \
                        --space ${space} \
                        --nlsyms ${nlsyms} \
                        --wp_model ${wp_model} \
                        --wp_nbest ${wp_nbest} \
                        --jobs ${jobs} \
                        --cutset_yaml ${data}/${x}/cutset.yaml \
                        --tsv_location ${data}/dataset/${x}_${unit}${wp_type}${vocab}.tsv || exit 1;
    done
    
    touch ${data}/.done_stage_2_${datasize}_${unit}${wp_type}${vocab}_sp${speed_perturb} && echo "Finish creating dataset for ASR (stage: 2)."
fi

mkdir -p ${model}
if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo ============================================================================
    echo "                        LM Training stage (stage:3)                       "
    echo ============================================================================

    if [ ! -e ${data}/.done_stage_3_${datasize}${lm_datasize}_${unit}${wp_type}${vocab} ]; then
        [ ! -e ${data}/.done_stage_1_${datasize}_sp${speed_perturb} ] && echo "run ./run.sh --datasize ${lm_datasize} first" && exit 1;

        echo "Making dataset tsv files for LM ..."
        mkdir -p ${data}/dataset_lm
        make_dataset.sh --unit ${unit} --wp_model ${wp_model} \
            ${data}/${train_set} ${dict} > ${data}/dataset_lm/train_nodev_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv || exit 1;
        for x in dev ${test_set}; do
            cp ${data}/dataset/${x}*_${lm_datasize}_${unit}${wp_type}${vocab}.tsv \
                ${data}/dataset_lm/${x}_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv || exit 1;
        done

        touch ${data}/.done_stage_3_${datasize}${lm_datasize}_${unit}${wp_type}${vocab} && echo "Finish creating dataset for LM (stage: 3)."
    fi

    lm_test_set="${data}/dataset_lm/eval1_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv \
                 ${data}/dataset_lm/eval2_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv \
                 ${data}/dataset_lm/eval3_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv"

    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/lm/train.py \
        --corpus csj \
        --config ${lm_conf} \
        --n_gpus ${n_gpus} \
        --cudnn_benchmark ${benchmark} \
        --train_set ${data}/dataset_lm/train_nodev_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv \
        --dev_set ${data}/dataset_lm/dev_${lm_datasize}_vocab${datasize}_${unit}${wp_type}${vocab}.tsv \
        --eval_sets ${lm_test_set} \
        --unit ${unit} \
        --dict ${dict} \
        --wp_model ${wp_model}.model \
        --model_save_dir ${model}/lm \
        --stdout ${stdout} \
        --resume ${lm_resume} || exit 1;

    echo "Finish LM training (stage: 3)."
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo ============================================================================
    echo "                       ASR Training stage (stage:4)                        "
    echo ============================================================================

    CUDA_VISIBLE_DEVICES=${gpu} ${NEURALSP_ROOT}/neural_sp/bin/asr/train.py \
        --corpus csj \
        --config ${conf} \
        --config2 ${conf2} \
        --n_gpus ${n_gpus} \
        --cudnn_benchmark ${benchmark} \
        --train_set ${data}/dataset/${train_set}_${unit}${wp_type}${vocab}.tsv \
        --dev_set ${data}/dataset/${dev_set}_${unit}${wp_type}${vocab}.tsv \
        --eval_sets ${data}/dataset/${test_single_set}_${datasize}_${unit}${wp_type}${vocab}.tsv \
        --train_cutset ${data}/${train_set}/cutset.yaml \
        --dev_cutset ${data}/${dev_set}/cutset.yaml \
        --eval_cutsets ${data}/${test_single_set}/cutset.yaml \
        --unit ${unit} \
        --dict ${dict} \
        --wp_model ${wp_model}.model \
        --model_save_dir ${model}/asr \
        --asr_init ${asr_init} \
        --external_lm ${external_lm} \
        --stdout ${stdout} \
        --resume ${resume} \
        --storage_type Lilcom \
        --batch_size 40 \
        --lhotse true|| exit 1;

    echo "Finish ASR model training (stage: 4)."
fi
