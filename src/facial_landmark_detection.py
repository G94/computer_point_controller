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
    Class for the Face Detection Model.
    '''

    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
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
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
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
        if self.flag == False:
            self.img_width = image.shape[1]
            self.img_height = image.shape[0]
            self.flag = True

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        self.set_image_metadata(image)

        p_frame = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        return p_frame


    def preprocess_output(self, outputs, image, eye_surrounding_area = 10):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''

        leye_x = outputs[0][0].tolist()[0][0]
        leye_y = outputs[0][1].tolist()[0][0]
        reye_x = outputs[0][2].tolist()[0][0]
        reye_y = outputs[0][3].tolist()[0][0]

        box = (leye_x, leye_y, reye_x, reye_y)

        h, w = image.shape[0:2]
        # w = image.shape[1]
        box = box * np.array([w, h, w, h])
        box = box.astype(np.int32)

        (lefteye_x, lefteye_y, righteye_x, righteye_y) = box
        # cv2.rectangle(image,(lefteye_x,lefteye_y),(righteye_x,righteye_y),(255,0,0))

        le_xmin = lefteye_x - eye_surrounding_area
        le_ymin = lefteye_y - eye_surrounding_area
        le_xmax = lefteye_x + eye_surrounding_area
        le_ymax = lefteye_y + eye_surrounding_area

        re_xmin = righteye_x - eye_surrounding_area
        re_ymin = righteye_y - eye_surrounding_area
        re_xmax = righteye_x + eye_surrounding_area
        re_ymax = righteye_y + eye_surrounding_area

        left_eye = image[le_ymin:le_ymax, le_xmin:le_xmax]
        right_eye = image[re_ymin:re_ymax, re_xmin:re_xmax]
        eye_coords = [[le_xmin, le_ymin, le_xmax, le_ymax], [re_xmin, re_ymin, re_xmax, re_ymax]]
        ## 
        return (lefteye_x, lefteye_y), (righteye_x, righteye_y), eye_coords, left_eye, right_eye

    def draw_outputs(self, outputs, image, coords):
        '''
        :param coords: coordinates of the box
        :param image: image where to draw the box
        '''
        
        box = []
        
        # keypoints = [landmarks.left_eye,
        #         landmarks.right_eye,
        #         landmarks.nose_tip,
        #         landmarks.left_lip_corner,
        #         landmarks.right_lip_corner]
        
        # out[self.output_blob].reshape((-1, 2))) \
        #               for out in outputs

        ##3 join them in pairs
        pairs_coords = [out.reshape((-1, 2)).tolist() for out in outputs]
        # print(pairs_coords)
        # print(pairs_coords[0][1])
        # print("---")
        # print(pairs_coords[0][2])
        # print(pairs_coords[1])
        # sys.exit(0)
        # left_eye_x = (landmarks.left_eye[0] * faceBoundingBoxWidth + roi[0].position[0])
        # left_eye_y = (landmarks.left_eye[1] * faceBoundingBoxHeight + roi[0].position[1])
        
        # right_eye_x = (landmarks.right_eye[0] * faceBoundingBoxWidth + roi[0].position[0])
        # right_eye_y = (landmarks.right_eye[1] * faceBoundingBoxHeight + roi[0].position[1])
        
        # nose_tip_x = (landmarks.nose_tip[0] * faceBoundingBoxWidth + roi[0].position[0])
        # nose_tip_y = (landmarks.nose_tip[1] * faceBoundingBoxHeight + roi[0].position[1])
        
        # left_lip_corner_x = (landmarks.left_lip_corner[0] * faceBoundingBoxWidth + roi[0].position[0])
        # left_lip_corner_y = (landmarks.left_lip_corner[1] * faceBoundingBoxHeight + roi[0].position[1])



        for item  in pairs_coords[0]:
            print(item)
            center = np.array((coords[0][0], coords[0][1])) +  np.array((coords[0][2], coords[0][3])) * np.array((item[0], item[1]))
            print(center)
            cv2.circle(image,tuple(center.astype(int)), 10, (255, 255, 0), -1)     
        
        return image


    def predict(self, image, coords):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        
        # try:
        resize_frame = self.preprocess_input(image)

        outputs = self.net.infer({self.input_name: resize_frame})
        # print("-----")
        # print(outputs[self.output_name].shape)
        # print("-----")
        # print([self.output_name])
        # sys.exit(0)
        ## preprocess_output(self, outputs, image, eye_surrounding_area = 10):
        #    (lefteye_x, lefteye_y), (righteye_x, righteye_y), eye_coords, left_eye, right_eye = self.preprocess_output(outputs[self.output_name], image, 10)
        write_image = self.draw_outputs(outputs[self.output_name], image, coords)

        return write_image
