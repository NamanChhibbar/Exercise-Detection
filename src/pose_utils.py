from typing import Iterable

import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
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

def get_cosine_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
  '''
  Computes the cosine of the angle between the vectors (p2 - p1) and (p3 - p1), where p1, p2, and p3 are points in a space.

  Parameters:
    p1 (np.ndarray): First point.
    p2 (np.ndarray): Second point.
    p3 (np.ndarray): Third point.

  Returns:
    float: Cosine of the angle between the vectors (p2 - p1) and (p3 - p1).
  '''
  v1 = p2 - p1
  v2 = p3 - p1
  cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
  return cos_angle

def convert_to_cosine_angles(landmarks: np.ndarray) -> np.ndarray:
  '''
  Converts the landmarks to cosine of angles made by the left and right shoulders, elbows, hips, and knees.

  Parameters:
    landmarks (np.ndarray): Landmarks of shape (num_frames, 33, 3).

  Returns:
    np.ndarray: Cosine of angles of the shape (num_frames, 8).
  '''
  angles = []
  for frame_landmarks in landmarks:
    frame_angles = []
    # Left shoulder angle
    frame_angles.append(get_cosine_angle(frame_landmarks[11], frame_landmarks[13], frame_landmarks[23]))
    # Right shoulder angle
    frame_angles.append(get_cosine_angle(frame_landmarks[12], frame_landmarks[14], frame_landmarks[24]))
    # Left elbow angle
    frame_angles.append(get_cosine_angle(frame_landmarks[13], frame_landmarks[11], frame_landmarks[15]))
    # Right elbow angle
    frame_angles.append(get_cosine_angle(frame_landmarks[14], frame_landmarks[12], frame_landmarks[16]))
    # Left hip angle
    frame_angles.append(get_cosine_angle(frame_landmarks[23], frame_landmarks[11], frame_landmarks[25]))
    # Right hip angle
    frame_angles.append(get_cosine_angle(frame_landmarks[24], frame_landmarks[12], frame_landmarks[26]))
    # Left knee angle
    frame_angles.append(get_cosine_angle(frame_landmarks[25], frame_landmarks[23], frame_landmarks[27]))
    # Right knee angle
    frame_angles.append(get_cosine_angle(frame_landmarks[26], frame_landmarks[24], frame_landmarks[28]))
    angles.append(frame_angles)
  return np.array(angles)


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

  def extract(self, frames: Iterable[np.ndarray], fps: float) -> np.ndarray:
    '''
    Extracts pose landmarks from the given frames.

    Parameters:
      frames (Iterable[np.ndarray]): Iterable of frames from the video file.
      fps (float): Frames per second of the video file.

    Returns:
      np.ndarray: A 3D array of shape (num_frames, 33, 3) containing x, y, z coordinates of the landmarks.
    '''
    # Get frame duration in microseconds
    frame_duration = int(1_000_000 / fps)
    frame_timestamp = 0
    frame_count = 0
    frames_extracted = 0
    landmarks_list = []
    for frame in frames:
      # Break if no frame is read or max frames are extracted
      if frames_extracted == self.max_frames:
        break
      frame_count += 1
      # Skip frames based on the sample rate
      if frame_count % self.sample_rate > 0:
        continue
      # Convert the frame to RGB format
      mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
      # Extract landmarks
      with PoseLandmarker.create_from_options(self.options) as landmarker:
        results = landmarker.detect_for_video(mp_frame, frame_timestamp)
      if results.pose_landmarks:
        # Extract landmarks in the form of a 3D array
        landmarks = np.array([
          [landmark.x, landmark.y, landmark.z]
          for landmark in results.pose_landmarks[0]
        ])
        landmarks_list.append(landmarks)
        frames_extracted += 1
      frame_timestamp += frame_duration
    return np.array(landmarks_list)
