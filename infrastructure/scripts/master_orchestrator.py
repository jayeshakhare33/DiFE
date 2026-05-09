import argparse
import sys
import time
import threading
import json
import os
import shutil
from typing import List, Dict

import boto3
import requests
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import FileResponse
import uvicorn

# Global state
received_files = []
expected_files = 4 # node features, node mapping, edge features, edge mapping
worker_instance_ids = []

app = FastAPI()

def zip_codebase():
    """Zip the feature_engineering and infrastructure directories for workers to download."""
    # Assuming this script is at /opt/dife/infrastructure/scripts/master_orchestrator.py or similar
    # We will zip the entire repo root except venv and git
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    zip_path = "/tmp/dife_code"
    
    # We use shutil.make_archive to create a zip
    shutil.make_archive(zip_path, 'zip', repo_root)
    return zip_path + ".zip"

@app.get("/download_code")
async def download_code():
    zip_file = zip_codebase()
    return FileResponse(path=zip_file, filename="dife_code.zip", media_type="application/zip")

@app.post("/upload_result")
async def upload_result(worker_id: str = Form(...), extractor_class: str = Form(...), file: UploadFile = File(...)):
    # Save the uploaded file locally
    os.makedirs("/tmp/master_results", exist_ok=True)
    file_path = f"/tmp/master_results/worker_{worker_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    received_files.append(file_path)
    print(f"[MASTER] Received {file.filename} from Worker {worker_id} ({extractor_class})")
    
    return {"status": "ok", "message": f"Saved {file.filename}"}


def start_api_server(port: int = 8080):
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return thread


def launch_workers(classes: List[str], args) -> None:
    """Launch EC2 workers via boto3."""
    ec2 = boto3.client("ec2", region_name=args.region)
    # Obtain the master private IP (self) to inject into worker user_data
    master_ip = args.master_ip
    callback_url = f"http://{master_ip}:8080/upload_result"
    download_url = f"http://{master_ip}:8080/download_code"

    # User data for the worker: downloads code from Master, unzips, pip installs, and runs.
    user_data_template = """#!/bin/bash
apt-get update -y
apt-get install -y python3-pip unzip curl

# Download codebase directly from the Master Node
curl -o /tmp/dife_code.zip {download_url}
mkdir -p /opt/dife
unzip -o /tmp/dife_code.zip -d /opt/dife/

# Install dependencies
cd /opt/dife
pip3 install -r requirements.txt

# RUN WORKER 
python3 infrastructure/scripts/worker_runner.py \\
    --extractor-class '{extractor_class}' \\
    --worker-id {worker_id} \\
    --output-dir /tmp/ \\
    --master-upload-url {callback} \\
    --master-ip {master_ip}
"""
    
    for idx, cls_name in enumerate(classes):
        worker_id = idx + 1
        user_data = user_data_template.format(
            extractor_class=cls_name,
            worker_id=worker_id,
            download_url=download_url,
            callback=callback_url,
            master_ip=master_ip
        )
        try:
            response = ec2.run_instances(
                ImageId=args.ami_id,
                InstanceType=args.instance_type,
                MinCount=1,
                MaxCount=1,
                KeyName=args.key_pair_name,
                SecurityGroupIds=[args.security_group_id],
                SubnetId=args.subnet_id,
                UserData=user_data,
                TagSpecifications=[{
                    "ResourceType": "instance",
                    "Tags": [{"Key": "Name", "Value": f"dife-worker-{worker_id}-{cls_name}"},
                               {"Key": "Project", "Value": args.project_tag}]
                }]
            )
            instance_id = response['Instances'][0]['InstanceId']
            worker_instance_ids.append(instance_id)
            print(f"[MASTER] Launched worker {worker_id} ({instance_id}) for class: {cls_name}")
        except Exception as e:
            print(f"[MASTER] Failed to launch worker {worker_id}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Central Hub Master Orchestrator")
    parser.add_argument("--extractor-classes", default="FeatureExtractor,EdgeFeatureExtractor")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--ami-id", required=True, help="AMI ID for worker instances")
    parser.add_argument("--instance-type", default="t3.micro")
    parser.add_argument("--key-pair-name", default="dife-keypair")
    parser.add_argument("--security-group-id", required=True)
    parser.add_argument("--subnet-id", required=True)
    parser.add_argument("--project-tag", default="DiFE")
    parser.add_argument("--master-ip", default="127.0.0.1", help="Private IP of this master node")
    args = parser.parse_args()

    classes = [c.strip() for c in args.extractor_classes.split(",") if c.strip()]
    print(f"[MASTER] Starting Central Hub. Expecting {len(classes)*2} files from workers.")

    # Start FastAPI server
    api_thread = start_api_server()
    time.sleep(2)  # Give server a moment to start

    # Launch workers
    launch_workers(classes, args)

    # Wait for all files to be received
    print("[MASTER] Waiting for workers to finish and upload Parquet files...")
    while len(received_files) < len(classes) * 2: # 2 files per worker (features + mapping)
        time.sleep(5)
    
    print("[MASTER] All workers have uploaded their Parquet files successfully!")
    print(f"[MASTER] Received files: {received_files}")

    # =========================================================================
    # S3 UPLOAD SECTION (Commented out for you to add your S3 bucket logic later)
    # =========================================================================
    # print("[MASTER] Uploading all received Parquet files to S3...")
    # s3 = boto3.client('s3', region_name=args.region)
    # s3_bucket_name = "YOUR_S3_BUCKET_NAME"
    # for local_file_path in received_files:
    #     file_name = os.path.basename(local_file_path)
    #     s3_key = f"features_output/{file_name}"
    #     print(f"[MASTER] Uploading {local_file_path} to s3://{s3_bucket_name}/{s3_key}")
    #     s3.upload_file(local_file_path, s3_bucket_name, s3_key)
    # print("[MASTER] S3 Upload complete.")
    # =========================================================================

    # Terminate workers
    if worker_instance_ids:
        print(f"[MASTER] Terminating worker instances: {worker_instance_ids}")
        ec2 = boto3.client("ec2", region_name=args.region)
        try:
            ec2.terminate_instances(InstanceIds=worker_instance_ids)
            print("[MASTER] Workers terminated successfully.")
        except Exception as e:
            print(f"[MASTER] Error terminating workers: {e}")

    print("[MASTER] Orchestration complete. Exiting.")
    sys.exit(0)

if __name__ == "__main__":
    main()
