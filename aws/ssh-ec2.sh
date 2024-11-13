#!/bin/bash
#
source ./config.sh

ssh -i $SSH_KEY_PATH ec2-user@$EC2_IP
