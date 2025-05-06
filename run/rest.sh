#!/bin/bash
set -e
cleanup() {
    echo "Cleaning up..."
    kill $tail_pid 2>/dev/null || true
    wait $tail_pid 2>/dev/null || true
}
trap cleanup EXIT INT TERM

job_id=$(sbatch --parsable rest.sbatch)
echo "Submitted job $job_id"
log_path="/data/scratch/apprisco/logs/job_output_${job_id}.log"
echo "Waiting for output log at $log_path..."
while [ ! -f "$log_path" ]; do
    sleep 3
    job_info=$(squeue -j "$job_id" --noheader -o "%T %R")
    echo $job_info
done

echo "Tailing log:"
tail -f "$log_path" -n 100 &
tail_pid=$!
echo "Tailing tail at $tail_pid"
while squeue -j $job_id --noheader > /dev/null; do
    sleep 5
done
echo "Job $job_id completed. Stopping log tail."
kill $tail_pid
