#! /bin/bash

# Copyright  2015 Tokyo Institute of Technology (Authors: Takafumi Moriya and Takahiro Shinozaki)
#            2015 Mitsubishi Electric Research Laboratories (Author: Shinji Watanabe)
# Apache 2.0
# Acknowledgement  This work was supported by JSPS KAKENHI Grant Number 26280055.
# modified by Riku Nakano 2019.05.08
# modified by Natsuki Hibi 2020.02.25
# original: erpkg/csj_autorun.sh

if [ $# -ne 3 ]; then
    echo "Usage: "$(basename $0)" <data_set> <transcription-dir> <morph_flag>"
    echo "e.g., "$(basename $0)" debug_train_dataset data/csj-data false"
    exit 1
fi

data_set=$1
outd=$2
morph_flag=$3

resource="/disk107/DATA"
SDB=MORPH/SDB
WAV=WAV

#################################
#    we load Ernie own data.    #
#################################
echo "#################################"
echo "#    we load Ernie own data.    #"
echo "#################################"
wav_list_path="/disk107/DATA/ERNIE/learning_wavs/$data_set"
echo "Use WavList: ${wav_list_path}"
wav_list=$(cat $wav_list_path)
vol='core'
declare -A WAV_ID_NAME
eval WAV_ID_NAME=(`cat $resource/ERNIE/wav_id_name.txt`)
if [ ! -e $outd/.done_make_trans_ernie ]; then
    touch ${outd}/Not_exist.list #data/csj-data
    for id in $wav_list; do
        TPATH="$resource/ERNIE/$SDB"
        WPATH="$resource/ERNIE/$WAV"
	
	    if [ -f "${WPATH}/${WAV_ID_NAME[$id]}" ]; then
            sdb_id=$(find ${TPATH}/E*000${id}.sdb -type f | gawk -F/ '{print $NF}' | sed 's/.sdb//g')
            mkdir -p $outd/$vol/${sdb_id}

            if $morph_flag; then
                erpkg/csj2kaldi4m.pl $TPATH/${sdb_id}.sdb  $outd/$vol/${sdb_id}/${sdb_id}.4lex $outd/$vol/${sdb_id}/${sdb_id}.4trn.t || exit 1;
            else
                erpkg/csj2kaldi4m_nomorph.pl $TPATH/${sdb_id}.sdb  $outd/$vol/${sdb_id}/${sdb_id}.4lex $outd/$vol/${sdb_id}/${sdb_id}.4trn.t || exit 1;
            fi

            local/csj_make_trans/csjconnect.pl -50 10 $outd/$vol/${sdb_id}/${sdb_id}.4trn.t ${sdb_id} > $outd/$vol/${sdb_id}/${sdb_id}-trans.text || exit 1;
            find $WPATH -iname ${WAV_ID_NAME[$id]} >$outd/$vol/${sdb_id}/${sdb_id}-wav.list || exit 1;
	    else
	        # Not path_exist
	        echo ${sdb_id} ${WPATH}/${WAV_ID_NAME[$id]} >> ${outd}/Not_exist.list
	    fi
    done

    # check file. if There are 0 byte file, exit.
    if [ -s $outd/$vol/${sdb_id}/${sdb_id}-trans.text ]; then
        echo -n >$outd/$vol/.done_$vol
        echo "Complete processing transcription data in ernie core."
    else
        echo "Bad processing of making transcriptions part. (ERNIE)" && exit;
    fi
    wait

    if [ -e $outd/$vol/.done_$vol ]; then
        echo -n >$outd/.done_make_trans_ernie
        echo "Done!"
    else
        echo "Bad processing of making transcriptions part. (ERNIE)" && exit;
    fi
fi
