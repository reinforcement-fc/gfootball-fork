#!/bin/bash

source /path/to/config.sh

AWS_PAGER="" aws ec2 start-instances --instance-ids $EC2_ID
