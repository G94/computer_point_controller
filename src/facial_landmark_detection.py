'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import cv2
import time
import sys

import numpy as np
from openvino.inference_engine import IENetwork, IECore


class FacialLandmarkDetection:
    '''
    Class for the FacialLandmarkDetection.
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
        self.input_shape = self.model.inputs[self.input_name].shape
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


    def preprocess_output(self, outputs, image, coords):
        '''
        :param outputs:
        :param image:
        :param coords:
        '''
        box = []
        pairs_coords = [out.reshape((-1, 2)).tolist() for out in outputs]

        x_facet_distance =  coords[0][2] - coords[0][0] 
        y_facet_distance =  coords[0][3] - coords[0][1] 

        for item  in pairs_coords[0]:
            center = np.array((coords[0][0], coords[0][1])) +  np.array((x_facet_distance, y_facet_distance)) * np.array((item[0], item[1]))
            
            cv2.circle(image,tuple(center.astype(int)), 2, (255, 255, 0), 4) 

        return None

    def draw_outputs(self, outputs, image, coords):
        '''
        Draw the elements detected by the model.
        :param outputs: Outputs are the coordinates for each landmark
        :param image: the face image
        '''

        p_frame = image.copy()
        box = []
        pairs_coords = [out.reshape((-1, 2)).tolist() for out in outputs]

        x_facet_distance =  coords[0][2] - coords[0][0] 
        y_facet_distance =  coords[0][3] - coords[0][1] 


      
        ## loop over each ladnmark to rescale to the scale of the face image
        eyes_pairs = []
        for item  in pairs_coords[0][0:2]:

            x_, y_ = int(item[0] * p_frame.shape[1]), int(item[1] * p_frame.shape[0])
            center = np.array((x_, y_))
            eyes_pairs.append(center)
            # cv2.circle(p_frame, tuple(center.astype(int)), 2, (255, 255, 0), 4)

        return p_frame, eyes_pairs

    def cropped_eye(self, eyes_pair_list, face_image):
        leye_x_min = eyes_pair_list[0][1] - 10
        leye_x_max = eyes_pair_list[0][1] + 10
        leye_y_min = eyes_pair_list[0][0] - 10
        leye_y_max = eyes_pair_list[0][0] + 10


        reye_x_min = eyes_pair_list[1][1] - 10
        reye_x_max = eyes_pair_list[1][1] + 10
        reye_y_min = eyes_pair_list[1][0] - 10
        reye_y_max = eyes_pair_list[1][0] + 10

        eye_coords = [[leye_x_min, leye_y_min, leye_x_max, leye_y_max],
                          [reye_x_min, reye_y_min, reye_x_max, reye_y_max]]

        ### cropping image
        l_eye_img = face_image[leye_x_min:leye_x_max, leye_y_min:leye_y_max]
        # cv2.imshow("l_eye_img", l_eye_img)
        # cv2.waitKey(2000)
 
        r_eye_img = face_image[reye_x_min:reye_x_max, reye_y_min:reye_y_max]       

        return l_eye_img, r_eye_img, eye_coords

    def predict(self, image, coords):
        '''
        :param image: Cropped face Image
        :param coords: Bounding box of the face
        :return:    image with landmarks, 
                    two cropped images for each eye (left and right) 
                    and the coordinates of each eye in an array.
        '''

        resize_frame = self.preprocess_input(image)
        landmarks_output = self.net.infer({self.input_name: resize_frame})
        draw_image, eyes_pairs = self.draw_outputs(landmarks_output[self.output_name], image, coords)
        l_eye_img, r_eye_img, eye_coords = self.cropped_eye(eyes_pairs, image)

        return draw_image, l_eye_img, r_eye_img, eye_coords, landmarks_output[self.output_name]
