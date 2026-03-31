# run notebooks via papermill

run_cmd="micromamba -n nedis run "
out_dir="runs"

# parameter ranges
n_repeats=5

seeds=("120")
drop_entities=("0")
# drop_entities=("0" "1" "2" "3" "4" "5" "6" "7" "8" "9" "10")

drop_samples=("0")
# drop_samples=("0" "4" "8" "12" "16" "20" "24" "28" "32" "36" "40")

drop_features=("0")
# drop_features=("0" "10" "20" "30" "40" "50" "60" "70" "80" "90" "100")

bootstrap_iterations=("1000")

bootstrap_thresholds=("0.7")
# bootstrap_thresholds=("-1" "0.7")

# create output directory
mkdir -p "${out_dir}"

# run notebooks
notebook_in="05.07_supplement_exp_regression_stable"
for repeat in $(seq 1 ${n_repeats}); do
    for seed in "${seeds[@]}"; do
        for drop_entity in "${drop_entities[@]}"; do
            for drop_sample in "${drop_samples[@]}"; do
                for drop_feature in "${drop_features[@]}"; do
                    for bootstrap_iteration in "${bootstrap_iterations[@]}"; do
                        for bootstrap_thresh in "${bootstrap_thresholds[@]}"; do
                            notebook_out="${out_dir}/RUN___${notebook_in}___drop-entites_${drop_entity}___drop-samples_${drop_sample}___drop-features_${drop_feature}___bootstrap-iter_${bootstrap_iteration}___bootstrap-thresh_${bootstrap_thresh}___seed_${seed}___repeat_${repeat}"
                            echo $notebook_out
                            ${run_cmd}papermill "${notebook_in}.ipynb" "${notebook_out}.ipynb" \
                                -p random_state "${seed}" \
                                -p postfix "repeat-${repeat}" \
                                -p n_entities_to_drop "${drop_entity}" \
                                -p n_samples_to_drop "${drop_sample}" \
                                -p n_features_to_drop "${drop_feature}" \
                                -p bootstrap_iterations "${bootstrap_iteration}" \
                                -p bootstrap_threshold "${bootstrap_thresh}" \
                                --stdout-file "${notebook_out}.out" \
                                --stderr-file "${notebook_out}.err" \
                                &> /dev/null &
                            sleep 1s
                        done
                    done
                done
            done
        done
    done
done
