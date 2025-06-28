import numpy as np
import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode


def normalize_flatten_landmarks(landmarks: np.ndarray) -> np.ndarray:
  '''
  Normalizes landmarks about the nose and flattens them to make them suitable for input to a classifier.

  Parameters:
    landmarks (np.ndarray): Array of shape (num_frames, 33, 3) containing 33 pairs (including nose landmarks) of (x, y, z) coordinates of the landmarks.

  Returns:
    np.ndarray: Normalized landmarks of shape (num_frames, 32, 3).
  '''
  # Normalize landmarks about the nose and remove the nose landmark
  normalized = landmarks[:, 1:] - landmarks[:, 0][:, np.newaxis]
  return normalized

def draw_landmarks(rgb_image: np.ndarray, detection_result) -> np.ndarray:
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)
  # Loop through the detected poses to visualize.
  for pose_landmarks in pose_landmarks_list:
    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
      for landmark in pose_landmarks
    ])
    mp.solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      mp.solutions.pose.POSE_CONNECTIONS,
      mp.solutions.drawing_styles.get_default_pose_landmarks_style()
    )
  return annotated_image


class LandmarkExtractor:
  '''
  Extracts pose landmarks from a video file at the specified rate.

  Attributes:
    sample_rate (int): Rate at which to sample frames to extract landmarks.
    max_frames (int | None): Maximum number of frames to extract landmarks from. If None, all frames will be processed.
    options (PoseLandmarkerOptions): Options for the pose landmarker.
  '''

  def __init__(
    self,
    model_path: str,
    sample_rate: int = 3,
    max_frames: int | None = None
  ) -> None:
    '''
    Initializes the LandmarkExtractor.

    Parameters:
      model_path (str): Path to the mediapipe pose landmarker model.
      sample_rate (int): Rate at which to sample frames to extract landmarks.
      max_frames (int | None): Maximum number of frames to extract landmarks from. If None, all frames will be processed.
    '''
    self.max_frames = max_frames
    self.sample_rate = sample_rate
    self.options = PoseLandmarkerOptions(
      base_options=BaseOptions(model_asset_path=model_path),
      running_mode=RunningMode.VIDEO
    )

  def extract(self, cap: cv2.VideoCapture) -> np.ndarray:
    '''
    Extracts pose landmarks from a video file using OpenCV VideoCapture.

    Parameters:
      cap (cv2.VideoCapture): OpenCV VideoCapture object for the video file.

    Returns:
      np.ndarray: A 3D array of shape (num_frames, 33, 3) containing x, y, z coordinates of the landmarks.
    '''
    # Check if the VideoCapture object is opened
    if not cap.isOpened():
      raise ValueError('VideoCapture object is closed.')
    # Get frame duration in microseconds
    frame_duration = int(1_000_000 / cap.get(cv2.CAP_PROP_FPS))
    frame_timestamp = 0
    frame_count = 0
    frames_extracted = 0
    landmarks_list = []
    try:
      while cap.isOpened():
        # Read a frame
        success, frame = cap.read()
        # Break if no frame is read or max frames are extracted
        if not success or frames_extracted == self.max_frames:
          break
        frame_count += 1
        # Extract landmarks at specified frame rate
        if frame_count % self.sample_rate == 0:
          # Convert the frame to RGB format
          mp_frame = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          )
          # Extract landmarks
          with PoseLandmarker.create_from_options(self.options) as landmarker:
            results = landmarker.detect_for_video(mp_frame, frame_timestamp)
          if results.pose_landmarks:
            # Flatten the landmarks
            landmarks = np.array([
              [landmark.x, landmark.y, landmark.z]
              for landmark in results.pose_landmarks[0]
            ])
            landmarks_list.append(landmarks)
            frames_extracted += 1
        frame_timestamp += frame_duration
    finally:
      # Release the video capture object
      cap.release()
    return np.array(landmarks_list)
