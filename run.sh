rm results.csv

for m in {0..5}
do
    for s in {0..5}
    do
        python3 get_results.py $s $m
    done
done