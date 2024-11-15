import boto3

def log_job(training_job_name):
    log_client = boto3.client("logs")
    log_group_name = "/aws/sagemaker/TrainingJobs"
    log_stream_name = f"{training_job_name}/algo-1"  # Typically, it's "{job_name}/algo-1" for single-instance training jobs

    # Retrieve logs from the log stream
    response = log_client.get_log_events(
        logGroupName=log_group_name,
        logStreamName=log_stream_name,
        startFromHead=True  # Set to False if you want the latest logs only
    )

    for event in response["events"]:
        print(event["message"])

# 훈련 작업 이름을 인자로 받아 실행
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Monitor SageMaker Training Job")
    parser.add_argument("training_job_name", type=str, help="The name of the SageMaker training job to monitor")
    args = parser.parse_args()

    # 훈련 작업 모니터링 시작
    log_job(args.training_job_name)
