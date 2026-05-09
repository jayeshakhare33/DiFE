// Worker node Terraform module
resource "aws_instance" "worker" {
  count         = var.worker_count
  ami           = var.ami_id
  instance_type = var.instance_type_worker
  key_name      = var.key_pair_name
  subnet_id     = aws_subnet.main.id
  vpc_security_group_ids = [aws_security_group.ec2_sg.id]
  associate_public_ip_address = true
  user_data = templatefile("${path.module}/user_data.sh.tpl", {
    region        = var.aws_region
    s3_bucket     = var.s3_bucket_name
    partition_key = "partitions/worker_${count.index + 1}.csv"
    master_private_ip = aws_instance.master.private_ip
    worker_id    = count.index
  })
  tags = {
    Name = "dife-worker-${count.index + 1}"
    Project = var.project_tag
  }
}
