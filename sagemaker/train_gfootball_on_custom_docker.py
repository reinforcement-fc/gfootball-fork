import sagemaker
import os
import sys
import argparse
from sagemaker.estimator import Estimator
from datetime import datetime


def training_job(wait):
    # SageMaker 세션 생성
    sagemaker_session = sagemaker.Session()

    docker_image_version="2.7"

    # Docker 이미지 URI
    docker_image_uri = f"443142193439.dkr.ecr.ap-southeast-2.amazonaws.com/gfootball-fork:{docker_image_version}"

    # 현재 시간으로 고유한 작업 이름 생성
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    training_job_name = f"gfootball-training-job-{current_time}"

    training_job_name = os.getenv("TRAINING_JOB_NAME", training_job_name)

    instance_type = "ml.g4dn.2xlarge"
    instance_type = os.getenv("INSTANCE_TYPE", instance_type)

    print(f"\ttraining_job_name:{training_job_name}")
    print(f"\tinstance_type:{instance_type}")

    # Estimator 설정 (entry_point 생략)
    custom_estimator = Estimator(
        image_uri=docker_image_uri,
        role="arn:aws:iam::443142193439:role/Sagemaker_reinforcementLearning-43008-iac",
        instance_count=1,
        max_run=86400*3,
        instance_type=instance_type,
        sagemaker_session=sagemaker_session,
        source_dir="/home/sagemaker-user/gfootball-fork/gfootball/uts_src",
        output_path=f"s3://sagemaker-ap-southeast-2-443142193439",
        script_mode=True,
        entry_point="train",

    )

    # 훈련 작업 시작
    custom_estimator.fit(job_name=training_job_name, wait=wait)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SageMaker Training Job")
    parser.add_argument("--wait", action="store_true", help="Wait until finishing the job (default: False)")
    args = parser.parse_args()

    # 훈련 작업 모니터링 시작
    training_job(args.wait)
