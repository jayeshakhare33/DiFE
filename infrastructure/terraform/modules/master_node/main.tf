// master_node/main.tf
resource "aws_instance" "master" {
  ami                    = var.ami_id
  instance_type          = var.instance_type_master
  key_name               = var.key_pair_name
  subnet_id              = aws_subnet.main.id
  vpc_security_group_ids = [aws_security_group.ec2_sg.id]
  associate_public_ip_address = true
  user_data              = templatefile("${path.module}/user_data.sh.tpl", {
    s3_bucket = var.s3_bucket_name
    csv_key   = var.csv_s3_key
    region    = var.aws_region
  })
  tags = {
    Name = "${var.project_tag}-master"
    Project = var.project_tag
  }
}

resource "aws_security_group" "ec2_sg" {
  name   = "${var.project_tag}-sg"
  vpc_id = aws_vpc.main.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    description = "Master API for workers"
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  tags = {
    Name = "${var.project_tag}-vpc"
    Project = var.project_tag
  }
}

resource "aws_subnet" "main" {
  vpc_id            = aws_vpc.main.id
  cidr_block        = "10.0.1.0/24"
  availability_zone = data.aws_availability_zones.available.names[0]
  tags = {
    Name = "${var.project_tag}-subnet"
    Project = var.project_tag
  }
}
