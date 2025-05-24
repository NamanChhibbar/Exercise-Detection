import base64

import cv2

def extract_str_frames(
  video_path: str,
  max_frames: int=10,
  frame_rate: int=3
) -> list[str]:
  '''
  Extracts frames from a video in base64 formatted strings.
  '''
  str_frames = []
  frame_count = 0
  frames_extracted = 0
  # Open the video file
  cap = cv2.VideoCapture(video_path)
  while True:
    # Read a frame from the video
    success, frame = cap.read()
    # Break if no frame is read or max frames are extracted
    if not success or frames_extracted == max_frames:
      break
    frame_count += 1
    if frame_count % frame_rate == 0:
      # Encode the frame as a JPEG image
      img_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
      # Convert the image bytes to a base64 string
      str_frame = base64.b64encode(img_bytes).decode('utf-8')
      str_frames.append(str_frame)
      frames_extracted += 1
  # Release the video capture object
  cap.release()
  return str_frames
