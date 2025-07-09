import os

import cv2
import tensorflow as tf
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from src.video_utils import video_capture_from_bytes, S3Connection
from src.pose_utils import LandmarkExtractor, convert_to_cosine_angles
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

class DetectExerciseLocalBody(BaseModel):
  file_location: str

# Load environment variables
load_dotenv()
# Initialize FastAPI app
app = FastAPI()

s3_connection = None
# s3_connection = S3Connection()
extractor = LandmarkExtractor(
  model_path=POSE_LANDMARKER_PATH,
  sample_rate=POSE_LANDMARKER_SAMPLE_RATE,
  max_frames=POSE_LANDMARKER_MAX_FRAMES
)
classifier = SequenceClassifier.load_model(model_path=LANDMARK_CLASSIFIER_PATH)

@app.post('/detect-exercise/')
async def detect_exercise(body: DetectExerciseBody):
  try:
    bucket = body.bucket
    key = body.key
    # Fetch video bytes from S3
    video_bytes = s3_connection.fetch_video(bucket, key)
    if not video_bytes:
      raise HTTPException(status_code=404, detail='Video file not found')
    # Get video capture object and temporary file name
    cap, tmp_name = video_capture_from_bytes(video_bytes)
    # Extract landmarks from the video
    landmarks = extractor.extract(cap)
    # Release the video capture object
    cap.release()
    # Clean up the temporary file
    os.remove(tmp_name)
    landmarks = convert_to_cosine_angles(landmarks)
    tensor_input = tf.convert_to_tensor(landmarks, dtype=tf.float32)[tf.newaxis, ...]
    predicted_class = classifier.predict(tensor_input)[0]
    return JSONResponse(content={'predicted_class': predicted_class}, status_code=200)
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
  # General error handling
  except Exception as e:
    raise HTTPException(
      status_code=500,
      detail=f'Error {str(e)} of type {type(e).__module__}.{type(e).__name__} occurred'
    )

@app.post('/detect-exercise-local/')
async def detect_exercise(body: DetectExerciseLocalBody):
  try:
    file_location = body.file_location
    # Process video and make prediction
    cap = cv2.VideoCapture(file_location)
    landmarks = extractor.extract(cap)
    angles = convert_to_cosine_angles(landmarks)
    tensor_input = tf.convert_to_tensor(angles, dtype=tf.float32)[tf.newaxis, ...]
    predicted_class = classifier.predict(tensor_input)[0]
    # Get max variance series and count repetitions
    series = max_variance_series(angles)
    repetitions = count_cycles(series, frac=0.1)
    return JSONResponse(
      content={'predicted_class': predicted_class, 'repetitions': repetitions},
      status_code=200
    )
  except KeyError as e:
    raise HTTPException(status_code=400, detail=f'Missing key: {e}')
  except Exception as e:
    return JSONResponse(
      content={'error': str(e), 'type': f'{type(e).__module__}.{type(e).__name__}'},
      status_code=500
    )
