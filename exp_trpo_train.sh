CUDA_VISIBLE_DEVICES=0

for f in sensor visual
#for f in visual
#for f in sensor
do

    for t in $(seq 1 3)
    do

        for s in $(seq 1 8)
        do

            for r in $(seq 1 5)
            do
                python3 train_trpo.py --feature_type $f --tcn_run_idx $t --split_idx $s --run_idx $r
            done

        done

    done

done
