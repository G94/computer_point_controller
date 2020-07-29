'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import cv2
import time
import sys
from src.input_feeder import InputFeeder
from src.face_detection import FaceDetection
from src.head_pose_estimation import HeadPoseEstimation
from src.facial_landmark_detection import FacialLandmarkDetection
from src.gaze_estimation import GazeEstimation
from src.mouse_controller import MouseController



### OpenVino Modules
from openvino.inference_engine import IENetwork, IECore

def facedetection_main(args):
    model = args.fc_model
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
    fc_model = args.fc_model
    hp_model = args.hp_model
    device = args.device
    file_type = args.input_type
    video_file = args.video

    output_path = args.output_path

    start_model_load_time = time.time()

    ### loading model required
    hpe_detection = HeadPoseEstimation(hp_model, device)
    hpe_detection.load_model()

    fc_detection = FaceDetection(fc_model, device)
    fc_detection.load_model()



    feed = InputFeeder(input_type = file_type, input_file = video_file)
    feed.load_data()
    

    initial_w = int(feed.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(feed.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(feed.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(feed.cap.get(cv2.CAP_PROP_FPS))

    out_video = cv2.VideoWriter(os.path.join(output_path, 'headposeestimation_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)

    for flag, frame in feed.next_batch():    
        if not flag:
            break

        image, coords = fc_detection.predict(frame)
        result = hpe_detection.predict(frame, coords)

        out_video.write(result)

    feed.close()

def faciallandmarkdetection_main(args):
    fc_model = args.fc_model
    fld_model = args.fl_model
    device = args.device
    file_type = args.input_type
    video_file = args.video

    output_path = args.output_path

    start_model_load_time = time.time()

    fl_detection = FacialLandmarkDetection(fld_model, device)
    fl_detection.load_model()

    fc_detection = FaceDetection(fc_model, device)
    fc_detection.load_model()


    feed = InputFeeder(input_type = file_type, input_file = video_file)
    feed.load_data()
    
    initial_w = int(feed.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(feed.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(feed.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(feed.cap.get(cv2.CAP_PROP_FPS))

    out_video = cv2.VideoWriter(os.path.join(output_path, 'faciallandmark_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)

    for flag, frame in feed.next_batch():    
        if not flag:
            break

        image, coords, face_image = fc_detection.predict(frame)
        # cv2.imshow("cropped", face_image)
        # cv2.waitKey(0)

        output_image = fl_detection.predict(face_image, coords)
        out_video.write(output_image)

    feed.close()


def gazedetection_main(args):
    ## models arguments
    fc_model = args.fc_model
    hp_model = args.hp_model
    fld_model = args.fl_model
    ge_model = args.ge_model

    device = args.device
    file_type = args.input_type
    video_file = args.video
    output_path = args.output_path

    start_model_load_time = time.time()

    ### Instantiate models
    fc_detection = FaceDetection(fc_model, device)
    fc_detection.load_model()

    hpe_detection = HeadPoseEstimation(hp_model, device)
    hpe_detection.load_model()

    fl_detection = FacialLandmarkDetection(fld_model, device)
    fl_detection.load_model()

    ge_detection = GazeEstimation(ge_model, device)
    ge_detection.load_model()


    ### Image Feeder
    feed = InputFeeder(input_type = file_type, input_file = video_file)
    feed.load_data()
    
    initial_w = int(feed.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(feed.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(feed.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = int(feed.cap.get(cv2.CAP_PROP_FPS))

    # out_video = cv2.VideoWriter(os.path.join(output_path, 'gazeestimation_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)

    for flag, frame in feed.next_batch():    
        if not flag:
            break

        image, coords, face_image = fc_detection.predict(frame)

        ### the cropped face iamge pass to the other models
        hp_image, list_angles = hpe_detection.predict(face_image, coords)
        fl_image, l_eye_img, r_eye_img, eye_coords = fl_detection.predict(face_image, coords)

        cv2.imshow("r_eye_img", r_eye_img)
        cv2.imshow("l_eye_img", l_eye_img)
        cv2.waitKey(2000)

        # mouse_coordinate, gaze_vector = ge_detection.predict(l_eye_img, r_eye_img, list_angles)
        # mouse_controller_object.move(mouse_coordinate[0], mouse_coordinate[1])
        # out_video.write(output_image)

    feed.close()


if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--fc_model',  required = True,  help = "Face Detection model folder")
    parser.add_argument('--hp_model',  required = True, help = "Head Pose Estimation model folder")
    parser.add_argument('--fl_model', required = True, help = "Facial Landmark Detection model folder")
    parser.add_argument('--ge_model', required = True, help = "Gaze Estimation model folder")

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
    # headdetection_main(args)
    # faciallandmarkdetection_main(args)
    gazedetection_main(args)

    ### run main script



