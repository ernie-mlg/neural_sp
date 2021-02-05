cpath=$(cd $(dirname $0); pwd)
scp_dir=extractor
scp_path=${cpath}/${scp_dir}

echo "remaking ${scp_path}/conf/online.conf"
echo "--feature-type=mfcc" > $cpath/online.conf
echo "--mfcc-config=${cpath}/mfcc.conf" >> $cpath/online.conf
echo "--ivector-extraction-config=${cpath}/ivector_extractor.conf" >> $cpath/online.conf
echo "--endpoint.silence-phones=1:2:3:4:5:6:7:8:9:10" >> $cpath/online.conf


echo "remaking ${scp_path}/ivector_extractor.conf"
echo "--splice-config=${scp_path}/splice.conf" > $cpath/ivector_extractor.conf
echo "--cmvn-config=${scp_path}/online_cmvn.conf" >> $cpath/ivector_extractor.conf
echo "--lda-matrix=${scp_path}/final.mat" >> $cpath/ivector_extractor.conf
echo "--global-cmvn-stats=${scp_path}/global_cmvn.stats" >> $cpath/ivector_extractor.conf
echo "--diag-ubm=${scp_path}/final.dubm" >> $cpath/ivector_extractor.conf
echo "--ivector-extractor=${scp_path}/final.ie" >> $cpath/ivector_extractor.conf
echo "--num-gselect=5" >> $cpath/ivector_extractor.conf
echo "--min-post=0.025" >> $cpath/ivector_extractor.conf
echo "--posterior-scale=0.1" >> $cpath/ivector_extractor.conf
echo "--max-remembered-frames=1000" >> $cpath/ivector_extractor.conf
echo "--max-count=100" >> $cpath/ivector_extractor.conf
