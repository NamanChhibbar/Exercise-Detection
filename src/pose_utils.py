import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
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


class LandmarkClassifier(keras.Model):
  '''
  An Energy-Based Model (EBM) for classifying sequences of pose landmarks.
  Uses an LSTM layer followed by a feedforward network (FFN) to classify sequences of pose landmarks.
  Uses energy thresholding to determine if an input is out-of-distribution (OOD).
  The model outputs the class index for in-distribution inputs and -1 for OOD inputs.
  '''

  def __init__(
    self,
    lstm_units: int,
    ffn_layer_sizes: list[int],
    num_classes: int,
    activation: str = 'relu',
    temperature: float = 1.0,
    threshold: float = 0.,
    training: bool = False,
    **kwargs
  ):
    '''
    Parameters:
      lstm_units: Number of units in the LSTM layer.
      ffn_layer_sizes: List of integers, each the size of a dense layer in the FFN.
      num_classes: Number of output classes.
      activation: Activation function to use in feedforward layers.
      temperature: Temperature parameter for energy calculation.
      threshold: Threshold for energy to determine if an input is out-of-distribution is valid.
      training: If True, the model returns softmax probabilities instead of class indices for training, otherwise the model returns class indices or -1 for OOD inputs.
    '''
    super().__init__(**kwargs)
    self.temperature = temperature
    self.threshold = threshold
    self.training = training
    # LSTM layer
    self.lstm = keras.layers.LSTM(units=lstm_units)
    # Feedforward network (list of Dense layers)
    self.ffn = keras.Sequential([
      keras.layers.Dense(units=size, activation=activation)
      for size in ffn_layer_sizes
    ])
    # Output layer
    self.output_layer = keras.layers.Dense(units=num_classes)
    # Build the layers with shape of landmark vectors
    self.build(input_shape=(None, None, 99))

  def call(self, x: tf.Tensor):
    x = self.lstm(x)
    x = self.ffn(x)
    logits = self.output_layer(x)
    if self.training:
      return tf.nn.softmax(logits, axis=-1)
    energy = self.energy(logits)
    classes = tf.argmax(logits, axis=-1, output_type=tf.int32)
    # Set classes to -1 if energy exceeds threshold
    return tf.where(energy > self.threshold, tf.fill(classes.shape, -1), classes)

  def build(self, input_shape):
    super().build(input_shape)
    # Build the LSTM layer
    self.lstm.build(input_shape)
    # Build the feedforward network
    self.ffn.build((None, self.lstm.units))
    # Build the output layer
    self.output_layer.build(self.ffn.output_shape)
  
  def energy(self, logits: tf.Tensor) -> tf.Tensor:
    '''
    Calculates energy of given logits.
    '''
    return -self.temperature * tf.reduce_logsumexp(logits/self.temperature, axis=-1)
