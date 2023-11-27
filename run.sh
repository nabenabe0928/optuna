while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
        --bench)
            bench="$2"
            shift
            shift
            ;;
        *)
            shift
            ;;
    esac
done

for q1 in 0.01 0.1 0.5 0.9 1.0
do
    for q2 in 0.01 0.1 0.5 0.9 1.0
    do
        subcmd="python -m experiments.bench --q1 ${q1} --q2 ${q2} --bench ${bench} --multivariate False"
        cmd="${subcmd} --ctpe False"
        echo $cmd
        $cmd

        cmd="${subcmd} --ctpe True"
        echo $cmd
        $cmd

        cmd="${subcmd} --ctpe True --gamma_type sqrt"
        echo $cmd
        $cmd
    done
done