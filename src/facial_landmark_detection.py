'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''

import os
import cv2
import time
# os.chdir("C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\python\python3.6")


from openvino.inference_engine import IENetwork, IECore
from input_feeder import InputFeeder

class FacialLandmarkDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        raise NotImplementedError

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        raise NotImplementedError

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        raise NotImplementedError

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
    '''
    Before feeding the data into the model for inference,
    you might have to preprocess it. This function is where you can do that.
    '''
        raise NotImplementedError

    def preprocess_output(self, outputs):
    '''
    Before feeding the output of this model to the next model,
    you might have to preprocess the output. This function is where you can do that.
    '''
        raise NotImplementedError



def main(args):

    model = args.model
    device = args.device
    file_type = args.input_type
    video_file = args.video

    # max_people = args.max_people
    # threshold = args.threshold
    # output_path = args.output_path

    start_model_load_time = time.time()
    fc_detection = FacialLandmarkDetection(model, device)

    fc_detection.load_model()
    feed = InputFeeder(input_type = file_type, input_file = video_file)
    feed.load_data()

    for batch in feed.next_batch():
        # print(type(batch))
        result = fc_detection.predict(batch)
        print(result)
        
    feed.close()


if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required = True)
    parser.add_argument('--device', default = 'CPU')
    parser.add_argument('--input_type', default = 'video')
    parser.add_argument('--video', default = './bin/demo.mp4')

    # parser.add_argument('--queue_param', default = None)
    # parser.add_argument('--output_path', default = '/results')
    # parser.add_argument('--max_people', default = 2)
    # parser.add_argument('--threshold', default = 0.60)

    args = parser.parse_args()

    main(args)