'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import cv2
import time
import math
import numpy as np

from openvino.inference_engine import IENetwork, IECore


class GazeEstimation:
    '''
    Class for the GazeEstimation.
    '''

    def __init__(self, model_name, device='CPU', extensions=None):
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

        pairs_coords = [out.reshape((-1, 2)).tolist() for out in landmarks_output]

        eyes_pairs = []

        for item  in pairs_coords[0][0:5]:
            x_, y_ = int(item[0] * p_frame.shape[1] + coords[0][0]), int(item[1] * p_frame.shape[0] + coords[0][1])
            center = np.array((x_, y_))
            eyes_pairs.append(center)
            cv2.circle(image, tuple(center.astype(int)), 2, (255, 255, 0), 4)


        scale = 0.4 * p_frame.shape[1]
        gaze_x_estimation = int((gaze_vector[0]) * scale)
        gaze_y_estimation = int(-(gaze_vector[1]) * scale)


        cv2.arrowedLine(image,  (eyes_pairs[0][0], eyes_pairs[0][1]), ((eyes_pairs[0][0] + gaze_x_estimation),  eyes_pairs[0][1] + (gaze_y_estimation)),(0, 0, 100), 3)
        cv2.arrowedLine(image,  (eyes_pairs[1][0], eyes_pairs[1][1]), ((eyes_pairs[1][0] + gaze_x_estimation),  eyes_pairs[1][1] + (gaze_y_estimation)),(0, 0, 100), 3)  
        
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

