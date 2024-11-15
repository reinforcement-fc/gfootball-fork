import boto3
import time

# SageMaker 클라이언트 생성
sagemaker_client = boto3.client('sagemaker', region_name='ap-southeast-2')

def monitor_training_job(training_job_name, poll_interval=60):
    """
    SageMaker 훈련 작업의 상태를 주기적으로 모니터링합니다.

    Args:
        training_job_name (str): 모니터링할 SageMaker 훈련 작업 이름
        poll_interval (int): 상태 체크 간격 (초 단위), 기본 60초
    """
    print(f"Monitoring SageMaker training job: {training_job_name}")
    
    while True:
        # 훈련 작업 상태 조회
        response = sagemaker_client.describe_training_job(TrainingJobName=training_job_name)
        training_status = response['TrainingJobStatus']
        secondary_status = response['SecondaryStatus']
        
        # 현재 상태 출력
        print(f"Status: {training_status}, Secondary Status: {secondary_status}")
        #from pprint import pprint
        #pprint(response)
        
        # 작업이 완료되거나 실패한 경우
        if training_status == 'Completed':
            print("Training job has completed successfully.")
            break
        elif training_status == 'Failed':
            print("Training job has failed.")
            print("Failure reason:", response.get('FailureReason', 'No reason provided'))
            break
        elif training_status == 'Stopped':
            print("Training job was manually stopped.")
            break

        # 설정한 간격 동안 대기
        time.sleep(poll_interval)

# 훈련 작업 이름을 인자로 받아 실행
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Monitor SageMaker Training Job")
    parser.add_argument("training_job_name", type=str, help="The name of the SageMaker training job to monitor")
    args = parser.parse_args()

    # 훈련 작업 모니터링 시작
    monitor_training_job(args.training_job_name)
