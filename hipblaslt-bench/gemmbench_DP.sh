#!/usr/bin/env bash

# Path to the benchmark executable
ROCBLAS_BENCH="/home/xguo/hipBLASLt/build/release/clients/staging/hipblaslt-bench"

# Environment variable(s)
ENV_VARS="HIP_VISIBLE_DEVICES=7"

# Common execution options
EXEC_OPTIONS="-e 38 -l 4"

# Common database insertion options
DB_OPTS="--insert_in_database \
         --db_host executive-dashboard.amd.com \
         --db_port 3307 \
         --db_user xguo \
         --db_arch mi300 \
         --db_branch develop \
	 --db_pass QVHb05pRO]6bQmgh\
         --db_repo https://github.com/ROCm/hipBLASLt"

# Loop over each transpose combination
for T in NN TN NT TT
do
    # YAML file and output file for each transpose type
    YAML_PATH="/home/xguo/StreamK/hipblaslt/scripts/benchmarks/Grid_I8II_${T}.yaml"
    OUTPUT_FILE="/home/xguo/StreamK/hipblaslt/scripts/benchmarks/Grid_I8II_${T}_DP_baseline.out"

    # Customize database label and comments as desired
    DB_LABEL="I8II_${T}_DP_baseline_new"
    DB_COMMENT="DP I8II ${T} Grid_benchmark"

    # Run the supervisor script
    ./rocblas_bench_supervisor.py \
        --rocblas_bench "$ROCBLAS_BENCH" \
        -v "$ENV_VARS" \
        --yaml "$YAML_PATH" \
        -n "$OUTPUT_FILE" \
        $EXEC_OPTIONS \
        $DB_OPTS \
        --db_label "$DB_LABEL" \
        --db_comment "$DB_COMMENT"

done
