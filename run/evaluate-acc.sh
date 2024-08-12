for name in 'awry' 'bargains' 'casino' 'cmv' 'donations' 'deals' 'deleted' 'supreme';
do
    python3 -m src.evaluate \
        --find 'gpt-4~trained=0~preds_for_instances='$name'-test@avg=1' \
        --exclude "tempval" "val@" "excess@" \
        --val 'build:'$name \
        --cal 1 \
        --avg 1 \
        --save_frames 0 \
        --clf_stats 1
done