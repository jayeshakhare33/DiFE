#!/bin/bash
apt-get update -y
apt-get install -y python3.10 python3-pip git awscli
pip3 install boto3 pandas pyarrow fastapi uvicorn
# Clone repository if needed (assumes repo is already on the instance)
# Run master orchestrator
python3 /opt/dife/infrastructure/scripts/master_orchestrator.py \
  --s3-bucket ${s3_bucket} \
  --csv-key ${csv_key} \
  --worker-count ${worker_count} \
  --region ${region}
