import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode


class LandmarkExtractor:
  '''
  Extracts pose landmarks from a video file at specified intervals.

  Attributes:
    max_frames (int): Maximum number of frames to extract landmarks from.
    frame_rate (int): Interval at which to extract landmarks (every nth frame).
    options (PoseLandmarkerOptions): Options for the pose landmarker.
  '''

  def __init__(self, model_path: str, frame_rate: int = 3, max_frames: int | None = None):
    '''
    Initializes the LandmarkExtractor.

    Parameters:
      model_path (str): Path to the mediapipe pose landmarker model.
      max_frames (int | None): Maximum number of frames to extract landmarks from. If None, all frames will be processed.
      frame_rate (int): Interval at which to extract landmarks (every nth frame).
    '''
    self.max_frames = max_frames
    self.frame_rate = frame_rate
    self.options = PoseLandmarkerOptions(
      base_options=BaseOptions(model_asset_path=model_path),
      running_mode=RunningMode.VIDEO
    )
  
  def __call__(self, video_path: str) -> np.ndarray:
    '''
    Extracts landmarks from a video at specified intervals.

    Parameters:
      video_path (str): Path to the input video file.
    
    Returns:
      np.ndarray: A 2D array of shape (num_frames, 99) containing the x, y, z coordinates of the landmarks.
    '''
    # Open the video
    cap = cv2.VideoCapture(video_path)
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
        if frame_count % self.frame_rate == 0:
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
            ]).flatten()
            landmarks_list.append(landmarks)
            frames_extracted += 1
        frame_timestamp += frame_duration
    finally:
      # Release the video capture object
      cap.release()
    return np.array(landmarks_list)
