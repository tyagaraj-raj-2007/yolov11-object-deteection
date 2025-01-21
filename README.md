Install required packages:
    pip install opencv-python onnxruntime-gpu

In the video_object_detection.py file, locate the video_path variable and update it with 
the correct path to your video file. For example
    video_path = 'path/to/your/video.mp4'

for video file inference, run the following command:
   python video_object_detection.py

For live webcam inference, run:
    python webcam_object_detection.py
