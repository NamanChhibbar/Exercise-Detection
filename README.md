# Exercise Detection API

This FastAPI project implements an API endpoint that:

- Fetches a video from an AWS S3 bucket
- Detects the type of exercise being performed
- Counts the number of repetitions

The project uses the MediaPipe Pose Landmarker for pose analysis and stacked LSTMs with a feed-forward network for landmark classification.

## Features

- Secure credential handling via `.env` file
- Fetches video bytes (no need to download) directly from AWS S3
- Interactive API guide at `/docs` endpoint

## Server Setup

Make sure you have `Python>=3.11` installed on your system.

### Virtual Environment Setup

Create a python virtual environment.

```sh
python -m venv .venv
```

Activate the virtual environment.

```sh
source .venv/bin/activate
```

Install necessary dependencies.

```sh
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the root directory following the template given in [.example.env](.example.env).

```sh
cp .example.env .env
```

Make sure you have the following environment variables set in your `.env` file:

```env
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_REGION_NAME=
```

### Required Models

Make sure you download a [Pose Landmarker model](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker) and have a sequence classifier model of extension `.keras` saved which accepts an input of dimension `(batch_size=1, sequence_len, 96)`.
A classifier class is provided in [src/classifier.py](src/classifier.py) that can be used to train and save the model.
Update the model paths in [configs.py](configs.py) accordingly.

### Run Server

Run your server using Uvicorn on the specified port number.

```sh
uvicorn server:app --port {port_number}
```

## Endpoints

### `POST` /detect-exercise/

Detect the exercise performed and count repetitions from a video stored in an S3 bucket.

#### Example cURL Request

```sh
curl -X POST http://localhost:8000/detect-exercise/ \
  -H "Content-Type: application/json" \
  -d '{"bucket": "my-bucket", "key": "videos/pushups.mp4"}'
```

#### Example Response

```json
{
  "predicted_class": "pushup",
  "repetitions": 10
}
