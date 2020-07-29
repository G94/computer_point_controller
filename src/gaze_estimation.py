'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
import os
import cv2
import time
# os.chdir("C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\python\python3.6")


from openvino.inference_engine import IENetwork, IECore


class GazeEstimation:
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


    def preprocess_output(self, outputs, head_position):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.

        '''
        roll = head_position[2]
        gaze_vector = output / cv2.norm(output)

        cosValue = math.cos(roll * math.pi / 180.0)
        sinValue = math.sin(roll * math.pi / 180.0)

        x = gaze_vector[0] * cosValue * gaze_vector[1] * sinValue
        y = gaze_vector[0] * sinValue * gaze_vector[1] * cosValue


        return None

    def draw_outputs(self, coords, image):
        '''
        :param coords: coordinates of the box
        :param image: image where to draw the box
        '''
        
        return None


    def predict(self, left_eye_image, right_eye_image, list_angles, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''

        resize_frame = self.preprocess_input(image)
        gaze_vector = self.net.infer({  'left_eye_image': 1 , 
                                        'right_eye_image': 2, 
                                        'head_pose_angles': list_angles})

        print(gaze_vector)
        # coords = self.preprocess_output(outputs[self.output_name])


        # post_image, post_coord = self.draw_outputs(coords, image)
        return  gaze_vector

