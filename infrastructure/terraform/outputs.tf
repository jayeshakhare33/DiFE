# Terraform outputs for DiFE distributed architecture
output "master_public_ip" {
  description = "Public IP of the master EC2 instance"
  value       = aws_instance.master.public_ip
}

output "master_instance_id" {
  description = "ID of the master EC2 instance"
  value       = aws_instance.master.id
}

output "s3_bucket_name" {
  description = "Name of the S3 bucket used for data exchange"
  value       = aws_s3_bucket.feature_store.id
}

output "private_key_path" {
  description = "Local path to the generated private key for SSH access"
  value       = local_file.private_key.filename
}
