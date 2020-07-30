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

from util.cast_argument import cast_to_int, cast_to_bool
import logging as log
import pandas as pd

### OpenVino Modules
from openvino.inference_engine import IENetwork, IECore


def gazedetection_main(args):
    logger = log.getLogger('computer_point_controler_application')
    logger.setLevel(log.DEBUG)
    
    ## models arguments
    fc_model = args.fc_model
    hp_model = args.hp_model
    fld_model = args.fl_model
    ge_model = args.ge_model

    device = args.device
    file_type = args.input_type
    input_file = args.input_file
    output_path = args.output_path

    ### Flag for Intermediate results
    fc_output = args.fc_output
    hp_output = args.hp_output
    fl_output = args.fl_output
    ge_output = args.ge_output

    ### Show image
    show_image = args.show_image

    start_model_load_time = time.time()

    ### Instantiate models
    fc_detection = FaceDetection(fc_model, device)
    fc_detection.load_model()

    hpe_detection = HeadPoseEstimation(hp_model, device)
    hpe_detection.load_model()

    fl_detection = FacialLandmarkDetection(fld_model, device)
    fl_detection.load_model()

  
    fc_output = cast_to_bool(fc_output)
    hp_output = cast_to_bool(hp_output)
    fl_output = cast_to_bool(fl_output)
    ge_output = cast_to_bool(ge_output)
    show_image = cast_to_bool(show_image)

    ge_detection = GazeEstimation(ge_model, device, None,  fc_output, hp_output, fl_output, ge_output, show_image )
    ge_detection.load_model()

    model_loading_time = time.time() - start_model_load_time
    logger.info("Models Loading : {}".format(model_loading_time))


    ### Image Feeder
    feed = InputFeeder(input_type = file_type, input_file = input_file)
    feed.load_data()
    
    ### Mouse Controller
    mouse_controller = MouseController('medium', 'fast')

    if not file_type == 'image':
        initial_w = int(feed.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        initial_h = int(feed.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_len = int(feed.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fps = int(feed.cap.get(cv2.CAP_PROP_FPS))

        out_video = cv2.VideoWriter(os.path.join(output_path, 'gazeestimation_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)

    fd = []
    hp = []
    fl = []
    ge = []

    overall = []
    for flag, frame in feed.next_batch(): 
        overall_inference = time.time()   
        if not flag:
            break
        original_fr = frame.copy()

        start_inference = time.time() 
        image, coords, face_image = fc_detection.predict(frame)
        fc_inference_time = time.time() - start_inference
        fd.append(fc_inference_time)
        logger.info("FaceDetection Inference Time : {}".format(fc_inference_time))


        ### the cropped face image pass to the other models
        start_inference = time.time() 
        hp_image, list_angles = hpe_detection.predict(face_image, coords)
        hp_inference_time = time.time() - start_inference
        hp.append(hp_inference_time)
        logger.info("Head Pose Estimation Inference Time : {}".format(hp_inference_time))

        start_inference = time.time() 
        fl_image, l_eye_img, r_eye_img, eye_coords, landmarks_output = fl_detection.predict(face_image, coords)
        fl_inference_time = time.time() - start_inference
        fl.append(fl_inference_time)
        logger.info("Facial LandMark Estimation Inference Time : {}".format(fl_inference_time))
        print("Facial LandMark Estimation Inference Time : {}".format(fl_inference_time))



        start_inference = time.time()
        mouse_coord, coord_mouse, output_image = ge_detection.predict(l_eye_img, r_eye_img,  coords, list_angles, landmarks_output, original_fr, face_image)
        ge_inference_time = time.time() - start_inference
        ge.append(ge_inference_time)
        logger.info("Gaze Estimation Inference Time : {}".format(ge_inference_time))
        print("Gaze Estimation Inference Time : {}".format(ge_inference_time))



        mouse_controller.move(mouse_coord[0], mouse_coord[1])
        if show_image:
            cv2.imshow("Gaze Estimation", output_image)
            if cv2.waitKey(1) == 27: 
                break


        if not file_type == 'image':
            out_video.write(output_image)

        else:
            cv2.imwrite('./result/image_infer.jpg', output_image)

            break

    feed.close()

    df = pd.DataFrame({'facedection':fd,'headpose':hp,'landmark':fl, 'gaze':ge})
    df.to_excel("inference_time_gpu.xlsx", index = False)

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--fc_model',  required = True,  help = "Face Detection model folder")
    parser.add_argument('--fc_output', default = 0,   required = False,  help = "Generate output of Face Detection ")

    parser.add_argument('--hp_model',  required = True, help = "Head Pose Estimation model folder")
    parser.add_argument('--hp_output', default = 0,    required = False, help = "Generate output of Head Pose Estimation ")

    parser.add_argument('--fl_model', required = True, help = "Facial Landmark Detection model folder")
    parser.add_argument('--fl_output', default = 0,  required = False, help = "Generate output of Facial Landmark Detection ")

    parser.add_argument('--ge_model', required = True, help = "Gaze Estimation model folder")
    parser.add_argument('--ge_output', default = 0,  required = False, help = "Generate output of Gaze Estimation")

    parser.add_argument('--device', default = 'CPU', choices = ['CPU', 'GPU', 'MYRIAD', "FPGA"], help = "Specify the device to run the inference CPU, GPU, FPGA, MYRIAD")
    parser.add_argument('--input_type', default = 'video', choices = ['video', 'cam', 'image'],  help = "type of input file could be video file or streaming from a camera")

    parser.add_argument('--input_file', default = './bin/demo.mp4', help = "path of the video file")
    parser.add_argument('--output_path', default = './result/',help = "path to store the results")
    
    parser.add_argument('--show_image', default = 0,  required = False, help = "show the frames on the screen")

    args = parser.parse_args()

    gazedetection_main(args)



