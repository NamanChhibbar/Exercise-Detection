import os

import tensorflow as tf
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from src.video_utils import video_capture_from_bytes, S3Connection
from src.pose_utils import LandmarkExtractor, normalize_landmarks, convert_to_cosine_angles
from src.classifier import SequenceClassifier
from src.repetitions_utils import max_variance_series, count_cycles
from configs import (
  POSE_LANDMARKER_PATH, POSE_LANDMARKER_SAMPLE_RATE, POSE_LANDMARKER_MAX_FRAMES,
  LANDMARK_CLASSIFIER_PATH
)

class DetectExerciseBody(BaseModel):
  '''Request body format for detect-exercise endpoint.'''
  bucket: str
  key: str

# Load environment variables
load_dotenv()
# Initialize FastAPI app
app = FastAPI()

s3_connection = S3Connection()
extractor = LandmarkExtractor(
  model_path=POSE_LANDMARKER_PATH,
  sample_rate=POSE_LANDMARKER_SAMPLE_RATE,
  max_frames=POSE_LANDMARKER_MAX_FRAMES
)
classifier = SequenceClassifier.load_model(model_path=LANDMARK_CLASSIFIER_PATH)

@app.post('/detect-exercise/')
async def detect_exercise(body: DetectExerciseBody):
  bucket = body.bucket
  key = body.key
  # Fetch video bytes from S3
  try:
    video_bytes = s3_connection.fetch_video(bucket, key)
  # Handle S3 errors
  except ClientError as e:
    match e.response['Error']['Code']:
      case 'NoSuchBucket':
        raise HTTPException(status_code=404, detail='Bucket does not exist')
      case 'NoSuchKey':
        raise HTTPException(status_code=404, detail='Video file does not exist')
      case 'AccessDenied':
        raise HTTPException(status_code=403, detail='Access denied to the bucket or key')
      case _:
        raise HTTPException(
          status_code=500,
          detail=f'Error {str(e)} of type {type(e).__module__}.{type(e).__name__} occurred while accessing S3'
        )
  # If no video bytes are returned, raise a 404 error
  if not video_bytes:
    raise HTTPException(status_code=404, detail='No video data found')
  try:
    # Get video capture object and temporary file name
    cap, tmp_name = video_capture_from_bytes(video_bytes)
    # Extract landmarks from the video
    landmarks = extractor.extract(cap)
    # Release the video capture object
    cap.release()
    # Clean up the temporary file
    os.remove(tmp_name)
    # Get cosine angles from landmarks to count repetitions
    angles = convert_to_cosine_angles(landmarks)
    # Normalize and flatten landmarks
    landmarks = normalize_landmarks(landmarks).resize(-1, 96)
    # Create tensor input of shape (batch_size, sequence_length, num_features)
    tensor_input = tf.convert_to_tensor(landmarks, dtype=tf.float32)[tf.newaxis, ...]
    # Predict class using
    predicted_class = classifier.predict(tensor_input)[0]
    # Get max variance series
    series = max_variance_series(angles)
    # Count repetitions in the series
    repetitions = count_cycles(series, frac=0.1)
    return JSONResponse(
      content={'predicted_class': predicted_class, 'repetitions': repetitions},
      status_code=200
    )
  # General error handling
  except Exception as e:
    raise HTTPException(
      status_code=500,
      detail=f'Encountered error {e} of type {type(e).__module__}.{type(e).__name__}'
    )
  finally:
    # Ensure the video capture object is released
    if 'cap' in locals() and cap.isOpened():
      cap.release()
    # Ensure the temporary file is cleaned up
    if 'tmp_name' in locals() and os.path.exists(tmp_name):
      os.remove(tmp_name)
