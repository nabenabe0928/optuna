for seed in `seq 0 19`
do
    for bench in hpolib jahs hpobench
    do
        cmd="python -m experiments.bench --seed ${seed} --bench ${bench} --ctpe False"
        echo $cmd
        $cmd

        cmd="python -m experiments.bench --seed ${seed} --bench ${bench} --ctpe True --gamma_type linear"
        echo $cmd
        $cmd

        cmd="python -m experiments.bench --seed ${seed} --bench ${bench} --ctpe True --gamma_type sqrt"
        echo $cmd
        $cmd
    done
done