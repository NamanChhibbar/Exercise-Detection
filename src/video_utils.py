import os
import io

import numpy as np
import imageio.v3 as iio
import boto3

from .pose_utils import LandmarkExtractor


def landmarks_from_video_bytes(video_bytes: bytes, extractor: LandmarkExtractor) -> np.ndarray:
  '''
  Extracts pose landmarks from video bytes.

  Parameters:
    video_bytes (bytes): Raw bytes of the video file.
    extractor (LandmarkExtractor): Instance of LandmarkExtractor to extract landmarks.

  Returns:
    np.ndarray: Extracted pose landmarks.
  '''
  try:
    # Create video bytes buffer
    buffer = io.BytesIO(video_bytes)
    # Create imageio reader from the buffer
    reader = iio.imopen(buffer, io_mode='r', plugin='pyav')
    # Get frames per second from the video metadata
    fps = reader.metadata()['fps']
    # Get iterator for frames
    frames = reader.iter()
    # Extract landmarks from the video
    landmarks = extractor.extract(frames, fps)
    # Close the reader and buffer
    reader.close()
    buffer.close()
  finally:
    # Ensure the reader and buffer are closed in case of an exception
    if 'reader' in locals():
      reader.close()
    if 'buffer' in locals():
      buffer.close()
  return landmarks


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
