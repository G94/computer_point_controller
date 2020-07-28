'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import cv2
import time
from math import cos, sin, pi

from openvino.inference_engine import IENetwork, IECore




class HeadPoseEstimation:
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


    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.

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
        :param coords: coordinates of the box
        :param image: image where to draw the box
        '''

        faceBoundingBoxWidth = coords[0].size[0]
        faceBoundingBoxHeight = coords[0].size[1]
        
        yaw = list_angles[0]
        pitch = list_angles[1]
        roll = list_angles[2]

        sinY = sin(yaw * pi / 180.0)
        sinP = sin(pitch * pi / 180.0)
        sinR = sin(roll * pi / 180.0)

        cosY = cos(yaw * pi / 180.0)
        cosP = cos(pitch * pi / 180.0)
        cosR = cos(roll * pi / 180.0)
        
        axisLength = 0.4 * faceBoundingBoxWidth
        xCenter = int(image[0].position[0] + faceBoundingBoxWidth / 2)
        yCenter = int(image[0].position[1] + faceBoundingBoxHeight / 2)

        if self.hp_out:   
            #center to right
            cv2.line(frame, (xCenter, yCenter), 
                            (((xCenter) + int (axisLength * (cosR * cosY + sinY * sinP * sinR))),
                            ((yCenter) + int (axisLength * cosP * sinR))),
                            (0, 0, 255), thickness=2)
            #center to top
            cv2.line(frame, (xCenter, yCenter), 
                            (((xCenter) + int (axisLength * (cosR * sinY * sinP + cosY * sinR))),
                            ((yCenter) - int (axisLength * cosP * cosR))),
                            (0, 255, 0), thickness=2)
            
            #Center to forward
            cv2.line(frame, (xCenter, yCenter), 
                            (((xCenter) + int (axisLength * sinY * cosP)),
                            ((yCenter) + int (axisLength * sinP))),
                            (255, 0, 0), thickness=2)       

        




        return box, image


    def predict(self, image, coords):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.

        :return: list of angles
        '''

        # try:
        resize_frame = self.preprocess_input(image)
        # print("InputFeeder sucessfully completed")

        print(self.input_name)
        outputs = self.net.infer({self.input_name: resize_frame})
        # print("InputFeeder sucessfully completed")
        # print([self.output_name])
        # print(outputs)

        list_angles = self.preprocess_output(outputs)

        # print("InputFeeder sucessfully completed")
        post_image, post_coord = self.draw_outputs(list_angles, image, coords)
        return list_angles

        # except Exception as e:
        #     print("Error in function self.predict:", e)
