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


print(IENetwork)

class FaceDetection:
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


    def preprocess_output(self, outputs, prob_threshold = 0.6):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.

        '''
        coords = []

        outs = outputs[0][0]

        for out in outs:
            conf = out[2]

            if conf > prob_threshold:
                x_min = out[3]
                y_min = out[4]
                x_max = out[5]
                y_max = out[6]
                coords.append([x_min,y_min,x_max,y_max])

        return coords

    def draw_outputs(self, coords, image):
        '''
        :param coords: coordinates of the box
        :param image: image where to draw the box
        '''
        
        box = []
        
        for ob in coords:
                x_facet = (int(ob[0] * self.img_width), int(ob[1] * self.img_height))
                y_facet = (int(ob[2] * self.img_width), int(ob[2] * self.img_height))    
                
                cv2.rectangle(image, x_facet, y_facet, (0, 55, 255), 1)
                box.append([x_facet[0], x_facet[1], y_facet[0], y_facet[1]])       
        
        return box, image


    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''

        # try:
        resize_frame = self.preprocess_input(image)
        # print("InputFeeder sucessfully completed")
        outputs = self.net.infer({self.input_name: resize_frame})
        # print("InputFeeder sucessfully completed")
        # print([self.output_name])
        # print(outputs)

        coords = self.preprocess_output(outputs[self.output_name])

        # print("InputFeeder sucessfully completed")
        post_image, post_coord = self.draw_outputs(coords, image)
        return post_image, post_coord

        # except Exception as e:
        #     print("Error in function self.predict:", e)


def main(args):
    

    model = args.model
    device = args.device
    file_type = args.input_type
    video_file = args.video

    # max_people = args.max_people
    # threshold = args.threshold
    output_path = args.output_path

    start_model_load_time = time.time()
    fc_detection = FaceDetection(model, device)

    fc_detection.load_model()
    print("load model sucessfully load")

    feed = InputFeeder(input_type = file_type, input_file = video_file)
    feed.load_data()

    initial_w = int(feed.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    initial_h = int(feed.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_len = int(feed.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(feed.cap.get(cv2.CAP_PROP_FPS))

    out_video = cv2.VideoWriter(os.path.join(output_path, 'facedetection_video.mp4'), cv2.VideoWriter_fourcc(*'avc1'), fps, (initial_w, initial_h), True)
    
    print("InputFeeder sucessfully completed")

    for flag, frame in feed.next_batch():
        # print(type(batch))
    
        if not flag:
            break
        # try:
            # cv2.imshow('video', cv2.resize(frame,(500,500)))
        coords, image = fc_detection.predict(frame)
        print("video write")
        out_video.write(image)

        # except Exception as e:
        #     print(str(e))
        
    feed.close()


if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required = True)
    parser.add_argument('--device', default = 'CPU')
    parser.add_argument('--input_type', default = 'video')
    parser.add_argument('--video', default = './bin/demo.mp4')
    parser.add_argument('--output_path', default = './result/')
    # parser.add_argument('--queue_param', default = None)
    # parser.add_argument('--output_path', default = '/results')
    # parser.add_argument('--max_people', default = 2)
    # parser.add_argument('--threshold', default = 0.60)

    args = parser.parse_args()

    main(args)