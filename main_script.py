'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import cv2
import time

from src.input_feeder import InputFeeder
from src.face_detection import FaceDetection
from src.head_pose_estimation import HeadPoseEstimation
from src.facial_landmark_detection import FacialLandmarkDetection
from src.gaze_estimation import GazeEstimation


### OpenVino Modules
from openvino.inference_engine import IENetwork, IECore

def facedetection_main(args):
    model = args.model
    device = args.device
    file_type = args.input_type
    video_file = args.video

    output_path = args.output_path

    start_model_load_time = time.time()
    fc_detection = FaceDetection(model, device)
    fc_detection.load_model()

    feed = InputFeeder(input_type = file_type, input_file = video_file)
    feed.load_data()
    
    initial_w = int(feed.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(feed.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(feed.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(feed.cap.get(cv2.CAP_PROP_FPS))

    out_video = cv2.VideoWriter(os.path.join(output_path, 'facedetection_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)

    for flag, frame in feed.next_batch():    
        if not flag:
            break

        coords, image = fc_detection.predict(frame)

        out_video.write(image)

    feed.close()

def headdetection_main(args):
    model = args.model
    device = args.device
    file_type = args.input_type
    video_file = args.video

    output_path = args.output_path

    start_model_load_time = time.time()

    ### loading model required
    hpe_detection = HeadPoseEstimation(model, device)
    hpe_detection.load_model()

    fc_detection = FaceDetection(model, device)
    fc_detection.load_model()



    feed = InputFeeder(input_type = file_type, input_file = video_file)
    feed.load_data()
    

    initial_w = int(feed.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(feed.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(feed.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(feed.cap.get(cv2.CAP_PROP_FPS))

    # out_video = cv2.VideoWriter(os.path.join(output_path, 'headposeestimation_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)

    for flag, frame in feed.next_batch():    
        if not flag:
            break
        coords, image = fc_detection.predict(frame)

        result = hpe_detection.predict(frame, coords)

        # out_video.write(image)

    feed.close()



if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--fc_model', required = True,  description = "Face detection model folder")
    parser.add_argument('--hp_model', required = True, description = "Head pose estimation model folder")

    parser.add_argument('--device', default = 'CPU')
    parser.add_argument('--input_type', default = 'video')
    parser.add_argument('--video', default = './bin/demo.mp4')
    parser.add_argument('--output_path', default = './result/')


    # parser.add_argument('--queue_param', default = None)
    # parser.add_argument('--output_path', default = '/results')
    # parser.add_argument('--max_people', default = 2)
    # parser.add_argument('--threshold', default = 0.60)

    args = parser.parse_args()

    ### test each model on my local computer
    # facedetection_main(args)
    headdetection_main(args)
    # faciallandmarkdetection_main(args)

    ### run main script



