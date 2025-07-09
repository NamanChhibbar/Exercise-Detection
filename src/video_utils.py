import os
import tempfile

import cv2
import boto3


def video_capture_from_bytes(video_bytes: bytes) -> cv2.VideoCapture:
  '''
  Creates a cv2.VideoCapture object from raw video bytes.
  Writes the video bytes to a temporary file and returns a VideoCapture object for that file.

  Parameters:
    video_bytes (bytes): Raw bytes of the video file.

  Returns:
    cv2.VideoCapture: A VideoCapture object that can be used to read the video frames
  '''
  with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
    # Write to a temporary file
    tmp.write(video_bytes)
    tmp.flush()
    # Create VideoCapture object
    cap = cv2.VideoCapture(tmp.name)
    # Save the temporary file name to cleanup later
    tmp_name = tmp.name
  return cap, tmp_name


class S3Connection:
  '''
  A class to handle connections to AWS S3 for video file retrieval.

  Attributes:
    s3_client (boto3.client): The S3 client for interacting with AWS S3.
  '''

  def __init__(
    self,
    aws_access_key: str | None = None,
    aws_secret_access_key: str | None = None,
    aws_region_name: str | None = None
  ) -> None:
    '''
    Initializes the S3Connection by creating a boto3 S3 client using credentials from environment variables.
    Must have AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION_NAME set in the environment.

    Parameters:
      aws_access_key (str | None): AWS access key ID. Defaults to environment variable AWS_ACCESS_KEY_ID.
      aws_secret_access_key (str | None): AWS secret access key. Defaults to environment variable AWS_SECRET_ACCESS_KEY.
      aws_region_name (str | None): AWS region name. Defaults to environment variable AWS_REGION_NAME.
    '''
    self.s3_client = boto3.client(
      's3',
      aws_access_key_id=aws_access_key or os.getenv('AWS_ACCESS_KEY_ID'),
      aws_secret_access_key=aws_secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY'),
      region_name=aws_region_name or os.getenv('AWS_REGION_NAME')
    )

  def fetch_video(self, bucket: str, key: str) -> bytes:
    '''
    Fetch a video file from S3.

    Parameters:
      bucket (str): The name of the S3 bucket.
      key (str): The key of the video file in the bucket.

    Returns:
      bytes: Raw bytes of the video file.
    '''
    response = self.s3_client.get_object(Bucket=bucket, Key=key)
    return response['Body'].read()
