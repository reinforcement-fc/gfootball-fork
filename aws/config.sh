#!/bin/bash
EC2_ID="i-0c1c896bb70557701"
SSH_KEY_PATH="~/.ssh/g4dn.pem"


EC2_IP=$(aws ec2 describe-instances --instance-ids $EC2_ID --query "Reservations[*].Instances[*].PublicIpAddress" --output text)

echo $EC2_IP


