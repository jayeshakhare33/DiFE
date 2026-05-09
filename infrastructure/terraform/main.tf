// Terraform main configuration for DiFE distributed architecture
terraform {
  required_version = ">= 1.5"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC and Subnet (default VPC reuse for simplicity)
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  tags = {
    Name = "${var.project_tag}-vpc"
  }
}

resource "aws_subnet" "public" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.1.0/24"
  map_public_ip_on_launch = true
  tags = {
    Name = "${var.project_tag}-subnet"
  }
}

# Security Group – allow SSH and master API port 8080
resource "aws_security_group" "allow_ssh_api" {
  name        = "${var.project_tag}-sg"
  description = "Allow SSH and API access"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "Master API"
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    description = "All outbound"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = {
    Name = "${var.project_tag}-sg"
  }
}

# S3 bucket for CSV input & Parquet output
resource "aws_s3_bucket" "feature_store" {
  bucket = var.s3_bucket_name
  acl    = "private"
  tags = {
    Name = "${var.project_tag}-bucket"
  }
}

resource "aws_s3_bucket_versioning" "feature_store_versioning" {
  bucket = aws_s3_bucket.feature_store.id
  versioning_configuration {
    status = "Enabled"
  }
}

# Key pair – generate a local private key and upload public part to AWS
resource "tls_private_key" "dife_key" {
  algorithm = "RSA"
  rsa_bits  = 2048
}

resource "aws_key_pair" "dife_keypair" {
  key_name   = var.key_pair_name
  public_key = tls_private_key.dife_key.public_key_openssh
}

# Save private key locally (outside TF state). Use null_resource with local_file provisioner.
resource "local_file" "private_key" {
  content  = tls_private_key.dife_key.private_key_pem
  filename = "${path.module}/../generated_keys/${var.key_pair_name}.pem"
  file_permission = "0600"
}

# Master EC2 instance
resource "aws_instance" "master" {
  ami                         = data.aws_ami.ubuntu.id
  instance_type               = var.instance_type_master
  subnet_id                   = aws_subnet.public.id
  vpc_security_group_ids      = [aws_security_group.allow_ssh_api.id]
  key_name                    = var.key_pair_name
  associate_public_ip_address = true
  tags = {
    Name = "${var.project_tag}-master"
  }

  user_data = templatefile("${path.module}/modules/master_node/user_data.sh.tpl", {
    s3_bucket = var.s3_bucket_name
    csv_key   = var.csv_s3_key
    region    = var.aws_region
    worker_count = var.worker_count
  })

  provisioner "remote-exec" {
    inline = ["echo Master instance ready"]
    connection {
      type        = "ssh"
      host        = self.public_ip
      user        = "ubuntu"
      private_key = tls_private_key.dife_key.private_key_pem
    }
  }
}

# Data source for latest Ubuntu 22.04 LTS AMI in the region
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*]
  }
}
