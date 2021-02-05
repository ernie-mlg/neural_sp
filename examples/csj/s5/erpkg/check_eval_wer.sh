# 生成したwer結果を記録する
# egs., ./erpkg/check_eval_wer.sh

if [ ! -d log ];then
        mkdir log
fi

rm log/wer_summary
for e in 1 2 3
do
        echo "eval$e" >> log/wer_summary
        cat exp/*/decode_eval${e}_csj/scoring_kaldi/best_wer >> log/wer_summary
        cat exp/*/*/decode_eval${e}*/scoring_kaldi/best_wer >> log/wer_summary
done

echo "### wer summary ###"
cat log/wer_summary
