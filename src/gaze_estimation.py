'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import cv2
import time
import math
import numpy as np
from math import cos, sin, pi
from openvino.inference_engine import IENetwork, IECore
import sys

ANGLE_CONVERSION = 180.0


class GazeEstimation:
    '''
    Class for the GazeEstimation.
    '''

    def __init__(self, model_name, device='CPU', extensions = None, flag_fe=True, flag_he=True, flag_fl=True, flag_gz=True, show_image = False):
        '''
        :param core: 
        :param model:
        :param net:

        :param input_name:
        :param input_shape:
        :param output_shape:
        '''

        self.model_weights = model_name+'.bin'
        self.model_structure = model_name + '.xml'
        self.device = device

        self.core = None
        self.model = None
        self.net = None

        self.input_name = None
        self.input_shape = None
        self.output_name = None
        self.output_shape = None 

        ### Image Metadata
        self.img_width = None
        self.img_height = None    
        self.flag = False

        ### Image Metadata
        self.show_face_detection = flag_fe
        self.show_he_estimation = flag_he    
        self.show_fl_detection= flag_fl
        self.show_gz_estimation = flag_gz
        self.show_image =  show_image


    def load_model(self):
        '''
        This function load the model and their metada.
        Specify the image input metadata.
        Specify the output metadata.
        '''

        self.core = IECore()
        self.model = IENetwork(self.model_structure, self.model_weights)   
        self.net = self.core.load_network(network = self.model, device_name = self.device, num_requests = 1)

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs['left_eye_image'].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape 

        print("input_name:", self.input_name)
        print("input_shape:", self.input_shape)
        print("output_name:", self.output_name)
        print("output_shape:", self.output_shape)


    def check_model(self):
        raise NotImplementedError

    def set_image_metadata(self, image):
        """
        Specify input image metadata
        """
        if self.flag == False:
            self.img_width = image.shape[1]
            self.img_height = image.shape[0]
            self.flag = True

    def preprocess_input(self, image):
        '''
        :param image:
        '''
        self.set_image_metadata(image)

        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame


    def preprocess_output(self, gaze_vector_inference, list_angles):
        '''
        :param gaze_vector_inference: vector of inference from GazeEstimator.
        :param list_angles: list of angles from Head pose estimation model.
        '''

        roll = list_angles[2]
        gaze_vector_denormalized = gaze_vector_inference / cv2.norm(gaze_vector_inference)

        cos_val = math.cos(roll * math.pi / 180.0)
        sin_val = math.sin(roll * math.pi / 180.0)

        x_mouse_coord = gaze_vector_denormalized[0] * cos_val * gaze_vector_denormalized[1] * sin_val
        y_mouse_coord = gaze_vector_denormalized[0] * sin_val * gaze_vector_denormalized[1] * cos_val

        return (x_mouse_coord, y_mouse_coord)


    def draw_outputs(self, gaze_vector, coords, list_angles, landmarks_output,  image, face_image):
        '''
        Draw the elements detected by the model.
        :param coords: coordinates of the box
        :param image: image where to draw the box
        '''
        p_frame = face_image.copy()
        box = []


        if self.show_face_detection:
            for ob in coords:
                    left_facet = (int(ob[0]), int(ob[1]))
                    right_facet = (int(ob[2]), int(ob[3]))    
                    cv2.rectangle(image, left_facet, right_facet, (0, 55, 255), 1)
                      

        if self.show_he_estimation:
            x_facet_distance =  coords[0][2] - coords[0][0] 
            y_facet_distance =  coords[0][3] - coords[0][1] 

            yaw = list_angles[0]
            pitch = list_angles[1]
            roll = list_angles[2]

            sin_y = sin(yaw * pi / ANGLE_CONVERSION)
            sin_p = sin(pitch * pi / ANGLE_CONVERSION)
            sin_r = sin(roll * pi / ANGLE_CONVERSION)

            cos_y = cos(yaw * pi / ANGLE_CONVERSION)
            cos_p = cos(pitch * pi / ANGLE_CONVERSION)
            cos_r = cos(roll * pi / ANGLE_CONVERSION)
            
            scale = 0.5 * coords[0][0]
            
            x_center = int(coords[0][0] + x_facet_distance / 2)
            y_center = int(coords[0][1] + y_facet_distance / 2)

            cv2.line(image, (x_center, y_center), 
                            (((x_center) + int (scale * (cos_r * cos_y + sin_y * sin_p * sin_r))),
                            ((y_center) + int (scale * cos_p * sin_r))),
                            (0, 0, 255), thickness = 2)

            cv2.line(image, (x_center, y_center), 
                            (((x_center) + int (scale * (cos_r * sin_y * sin_p + cos_y * sin_r))),
                            ((y_center) - int (scale * cos_p * cos_r))),
                            (0, 255, 0), thickness = 2)

            cv2.line(image, (x_center, y_center), 
                            (((x_center) + int (scale * sin_y * cos_p)),
                            ((y_center) + int (scale * sin_p))),
                            (255, 0, 0), thickness = 2)       


        

        pairs_coords = [out.reshape((-1, 2)).tolist() for out in landmarks_output]
        eyes_pairs = []
        for item  in pairs_coords[0][0:5]:
            x_, y_ = int(item[0] * p_frame.shape[1] + coords[0][0]), int(item[1] * p_frame.shape[0] + coords[0][1])
            center = np.array((x_, y_))
            eyes_pairs.append(center)

            if self.show_fl_detection:
                cv2.circle(image, tuple(center.astype(int)), 2, (255, 255, 0), 4)


        if self.show_gz_estimation:
            scale = 0.4 * p_frame.shape[1]
            gaze_x_estimation = int((gaze_vector[0]) * scale)
            gaze_y_estimation = int(-(gaze_vector[1]) * scale)


            cv2.arrowedLine(image,  (eyes_pairs[0][0], eyes_pairs[0][1]), ((eyes_pairs[0][0] + gaze_x_estimation),  eyes_pairs[0][1] + (gaze_y_estimation)),(0, 0, 100), 3)
            cv2.arrowedLine(image,  (eyes_pairs[1][0], eyes_pairs[1][1]), ((eyes_pairs[1][0] + gaze_x_estimation),  eyes_pairs[1][1] + (gaze_y_estimation)),(0, 0, 100), 3)  

        # print(self.show_image)

        # sys.exit(0)
        # if self.show_image:
        #     print(self.show_image)
        #     cv2.imshow("Gaze Estimation", image)
        #     # cv2.waitKey(100000)
        #     # sys.exit(0)
        #     if cv2.waitKey(1) == 27: 
        #         break

        return image
  


    def predict(self, left_eye_image, right_eye_image, coords, list_angles, landmarks_output, image, face_image):
        '''
        :param left_eye_image: left eye image with [1x3x60x60] shape.
        :param right_eye_image: rigth eye image with [1x3x60x60] shape.
        :param list_angles: estimates angles (yaw, pitch and roll).
        :param coords: bounding box.
        :param image: complete frame.
        '''

        left_eye_image = self.preprocess_input(left_eye_image)
        right_eye_image = self.preprocess_input(right_eye_image)

        gaze_vector = self.net.infer(inputs = {'left_eye_image': left_eye_image , 
                                                'right_eye_image': right_eye_image, 
                                                'head_pose_angles': list_angles})

        output_image = self.draw_outputs(gaze_vector[self.output_name][0],coords, list_angles, landmarks_output,  image, face_image)
        cords = self.preprocess_output(gaze_vector[self.output_name][0], list_angles)
        return  gaze_vector[self.output_name][0], cords, output_image

