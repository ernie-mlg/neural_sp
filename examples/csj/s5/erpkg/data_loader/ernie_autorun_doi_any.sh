#! /bin/bash

# Copyright  2015 Tokyo Institute of Technology (Authors: Takafumi Moriya and Takahiro Shinozaki)
#            2015 Mitsubishi Electric Research Laboratories (Author: Shinji Watanabe)
# Apache 2.0
# Acknowledgement  This work was supported by JSPS KAKENHI Grant Number 26280055.
# modified by Riku Nakano 2019.05.08

if [ $# -ne 3 ]; then
    echo "Usage: "$(basename $0)" <data_sets_id> <transcription-dir> <morph_flag>"
    echo "e.g., "$(basename $0)" 200313103857876339 data/csj-data true"
    echo "See comments in the script for more details"
    exit 1
fi

data_sets_id=$1
outd=$2
morph_flag=$3

set -e # exit on error

#[ ! -e $wavlist ] && echo "Not exist wavlist." && exit 1

bash erpkg/data_loader/data_loader.sh $data_sets_id $outd $morph_flag

## Exclude speech data given by test set speakers.
if [ ! -e $outd/.done_mv_eval_dup ]; then
    (
        echo "Make evaluation set 1, 2, 3. And exclude speech data given by test set speakers."
        mkdir -p $outd/{\eval,excluded}
        mkdir -p $outd/eval/eval{1,2,3}

        # Exclude speaker ID
        A01M0056="S05M0613 R00M0187 D01M0019 D04M0056 D02M0028 D03M0017"

        # Evaluation set ID
        eval1=$(cat erpkg/eval_list/eval1_list)
        eval2=$(cat erpkg/eval_list/eval2_list)
        eval3=$(cat erpkg/eval_list/eval3_list)
        # Speech data given by test set speakers (e.g. eval2 : A01M0056)
        for list in $A01M0056; do
            find . -type d -name $list | xargs -i mv {} $outd/excluded
        done
        wait
        for eval_file in $eval1; do
            mv $outd/core/$eval_file $outd/eval/eval1
        done
        wait

        for eval_file in $eval2; do
            mv $outd/core/$eval_file $outd/eval/eval2

        done
        wait

        for eval_file in $eval3; do
            mv $outd/core/$eval_file $outd/eval/eval3

        done
        wait

        # Evaluation data
        [ 10 -eq $(ls $outd/eval/eval1 | wc -l) ] && echo -n >$outd/eval/.done_eval1
        [ 10 -eq $(ls $outd/eval/eval2 | wc -l) ] && echo -n >$outd/eval/.done_eval2
        [ 10 -eq $(ls $outd/eval/eval3 | wc -l) ] && echo -n >$outd/eval/.done_eval3
        if [ 3 -eq $(ls -a $outd/eval | grep done_eval | wc -l) ]; then
            echo -n >$outd/.done_mv_eval_dup
            echo "Done!"
        else
            echo "Bad processing of making evaluation set part" && exit
        fi
    )
fi

## make lexicon.txt
if [ ! -e $outd/.done_make_lexicon ]; then
    echo "Make lexicon file."
    (
        lexicon=$outd/lexicon
        rm -f $outd/lexicon/lexicon.txt
        mkdir -p $lexicon
        cat $outd/*/*/*.4lex | grep -v "+ー" | grep -v "++" | grep -v "×" >$lexicon/lexicon.txt
        sort -u $lexicon/lexicon.txt >$lexicon/lexicon_htk.txt
        if $morph_flag; then
            local/csj_make_trans/vocab2dic.pl -p local/csj_make_trans/kana2phone -e $lexicon/ERROR_v2d -o $lexicon/lexicon.txt $lexicon/lexicon_htk.txt
            cut -d'+' -f1,3- $lexicon/lexicon.txt >$lexicon/lexicon_htk.txt
            cut -f1,3- $lexicon/lexicon_htk.txt | perl -ape 's:\t: :g' >$lexicon/lexicon.txt
            sort $lexicon/lexicon.txt | uniq >$lexicon/lexicon_tmp.txt
            cp $lexicon/lexicon_tmp.txt $lexicon/lexicon.txt
        else
            erpkg/local/csj_make_trans/vocab2dic.pl -p local/csj_make_trans/kana2phone -e $lexicon/ERROR_v2d -o $lexicon/lexicon.txt $lexicon/lexicon_htk.txt
            sort $lexicon/lexicon.txt | uniq > $lexicon/lexicon_tmp.txt
            cp $lexicon/lexicon_tmp.txt $lexicon/lexicon.txt
            python3 erpkg/morph_option/lex2clean.py $lexicon/lexicon_tmp.txt $lexicon/lexicon.txt
        fi
        
        rm $lexicon/lexicon_tmp.txt

        if [ -s $lexicon/lexicon.txt ]; then
            echo -n >$outd/.done_make_lexicon
            echo "Done!"
        else
            echo "Bad processing of making lexicon file" && exit
        fi
    )
fi

[ ! 3 -le $(ls -a $outd | grep done | wc -l) ] &&
    echo "ERROR : Processing is incorrect." && exit

echo "Finish processing original CSJ data" && echo -n >$outd/.done_make_all
