import json

from dotenv import load_dotenv
import tensorflow as tf
from botocore.exceptions import ClientError
from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from src.pose_utils import LandmarkExtractor, LandmarkClassifier
from src.s3_utils import video_capture_from_bytes, S3Connection
from configs import (
  POSE_LANDMARKER_PATH, POSE_LANDMARKER_SAMPLE_RATE, POSE_LANDMARKER_MAX_FRAMES,
  LANDMARK_CLASSIFIER_PATH
)


load_dotenv()
s3_connection = S3Connection()
extractor = LandmarkExtractor(
  model_path=POSE_LANDMARKER_PATH,
  sample_rate=POSE_LANDMARKER_SAMPLE_RATE,
  max_frames=POSE_LANDMARKER_MAX_FRAMES
)
classifier = LandmarkClassifier.load_model(model_path=LANDMARK_CLASSIFIER_PATH)
class_names = classifier.get_config()['class_names']


@csrf_exempt
def detect_exercise(request: HttpRequest) -> JsonResponse:
  '''
  Detects exercise from a video file uploaded in the request.

  Parameters:
    request (HttpRequest): The HTTP request object containing the video file.

  Returns:
    JsonResponse: A JSON response with the detected exercise and landmarks.
  '''
  # Accept only POST requests
  if request.method != 'POST':
    return JsonResponse({'error': 'Invalid request method'}, status=405)
  try:
    # Get video file info from the request body
    data = json.loads(request.body)
    if type(data) is not dict:
      return JsonResponse({'error': 'Invalid JSON'}, status=400)
    bucket = data['bucket']
    key = data['key']
    # Get video bytes from S3
    video_bytes = s3_connection.fetch_video(bucket, key)
    if not video_bytes:
      return JsonResponse({'error': 'Video file not found'}, status=404)
    # Create a VideoCapture object from the video bytes
    cap = video_capture_from_bytes(video_bytes)
    # Extract landmarks from the video
    landmarks = extractor(cap)
    # Create tensor input for the classifier
    tensor_input = tf.convert_to_tensor(landmarks, dtype=tf.float32)[tf.newaxis, ...]
    # Predict the exercise using the classifier
    class_index = classifier(tensor_input)[0]
    # Get class name
    predicted_class = class_names[class_index]
    return JsonResponse({'predicted_class': predicted_class}, status=200)
  except json.JSONDecodeError:
    return JsonResponse({'error': 'Invalid JSON'}, status=400)
  except KeyError as e:
    return JsonResponse({'error': f'Missing key: {e}'}, status=400)
  except ClientError as e:
    # Handle S3 specific errors
    match e.response['Error']['Code']:
      case 'NoSuchBucket':
        return JsonResponse({'error': 'Bucket does not exist'}, status=404)
      case 'NoSuchKey':
        return JsonResponse({'error': 'Video file does not exist'}, status=404)
      case 'AccessDenied':
        return JsonResponse({'error': 'Access denied to the bucket or key'}, status=403)
      case _:
        return JsonResponse(
          {'error': str(e), 'type': f'{type(e).__module__}.{type(e).__name__}'},
          status=500
        )
  except Exception as e:
    return JsonResponse(
      {'error': str(e), 'type': f'{type(e).__module__}.{type(e).__name__}'},
      status=500
    )
