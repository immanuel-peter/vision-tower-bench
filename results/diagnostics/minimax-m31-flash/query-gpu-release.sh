#!/usr/bin/env bash
set -euo pipefail
CUDA_VISIBLE_DEVICES=0 nvidia-smi --id=0 --query-gpu=index,name,uuid,driver_version,memory.used,utilization.gpu --format=csv,noheader,nounits > /tmp/vtb-delivery/issues/21/gpu-release-hardware.csv
CUDA_VISIBLE_DEVICES=0 nvidia-smi --id=0 --query-compute-apps=pid,process_name,gpu_uuid --format=csv,noheader,nounits > /tmp/vtb-delivery/issues/21/gpu-release-processes.csv
