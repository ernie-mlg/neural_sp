# 生成したcer結果を記録する
# egs., ./erpkg/check_eval_cer.sh

if [ ! -d log ];then
        mkdir log
fi

rm log/cer_summary
for e in 1 2 3
do
        echo "eval$e" >> log/cer_summary
        cat exp/*/*/decode_eval${e}*/scoring_kaldi/best_cer >> log/cer_summary
done

echo "### wer summary ###"
cat log/cer_summary
