
dst_path=/home/ubuntu
ec2_ip="52.62.93.171"

scp -r -i  /home/jeffrey/.ssh/krkeypair.pem ./models ubuntu@$ec2_ip:$dst_path

