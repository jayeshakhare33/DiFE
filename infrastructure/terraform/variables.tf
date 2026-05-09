# Terraform variables for the distributed EC2 architecture
variable "aws_region" {
  description = "AWS region to deploy resources"
  type        = string
  default     = "us-east-1"
}

variable "instance_type_master" {
  description = "EC2 instance type for the master node"
  type        = string
  default     = "t3.micro"  # Free tier
}

variable "instance_type_worker" {
  description = "EC2 instance type for worker nodes"
  type        = string
  default     = "t3.micro"  # Free tier
}

variable "worker_count" {
  description = "Number of worker EC2 instances"
  type        = number
  default     = 4
}

variable "s3_bucket_name" {
  description = "S3 bucket for CSV input and Parquet output"
  type        = string
  default     = "dife-feature-store"
}

variable "csv_s3_key" {
  description = "S3 key for the input transaction CSV"
  type        = string
  default     = "transactions/input.csv"
}

variable "key_pair_name" {
  description = "Name of the AWS key pair to use/create"
  type        = string
  default     = "dife-keypair"
}

variable "project_tag" {
  description = "Tag to apply to all resources"
  type        = string
  default     = "DiFE"
}
