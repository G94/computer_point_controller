'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import cv2
import time
import sys
from math import cos, sin, pi

from openvino.inference_engine import IENetwork, IECore


ANGLE_CONVERSION = 180.0


class HeadPoseEstimation:
    '''
    Class for the HeadPoseEstimation.
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
        try:
            self.model = IENetwork(self.model_structure, self.model_weights)   
            self.net = self.core.load_network(network = self.model, device_name = self.device, num_requests = 1)
        
        except FileNotFoundError as fnf_error:
            log.error("The file was not found : {}".format(model_loading_time))
            sys.exit(0)
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


    def preprocess_output(self, outputs):
        '''
        :param outputs:

        '''
        labels = ['angle_y_fc','angle_p_fc','angle_r_fc']

        list_angles = []
        # print(outputs)
        # print(type(outputs['angle_y_fc']))
 
        for label in labels:
            list_angles.append(outputs[label].tolist()[0][0])

        return list_angles
        

    def draw_outputs(self, list_angles, image, coords):
        '''
        Draw the elements detected by the model.
        :param coords: Face coordinates
        :param list_angles: yaw, pitch and roll from Head pose estimation
        :return image: Image with vectors
        '''
     
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
                        (0, 0, 255), thickness=2)

        cv2.line(image, (x_center, y_center), 
                        (((x_center) + int (scale * (cos_r * sin_y * sin_p + cos_y * sin_r))),
                        ((y_center) - int (scale * cos_p * cos_r))),
                        (0, 255, 0), thickness=2)

        cv2.line(image, (x_center, y_center), 
                        (((x_center) + int (scale * sin_y * cos_p)),
                        ((y_center) + int (scale * sin_p))),
                        (255, 0, 0), thickness=2)       

        return image


    def predict(self, image, coords):
        '''
        :param image: face image
        :return: list of angles
        '''

        resize_frame = self.preprocess_input(image)

        outputs = self.net.infer({self.input_name: resize_frame})
        list_angles = self.preprocess_output(outputs)

        image = self.draw_outputs(list_angles, image, coords)
        return image, list_angles

