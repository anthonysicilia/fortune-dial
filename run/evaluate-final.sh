for name in 'awry' 'bargains' 'casino' 'cmv' 'donations' 'deals' 'deleted' 'supreme';
do
    python3 -m src.evaluate \
        --find '='$name \
        --exclude "tempval" "val@" "excess@" "easy" "medium" "hard" \
        --val 'build:'$name \
        --cal 0 1 \
        --avg 1 \
        --match_avg_or_exclude_bss 1
done