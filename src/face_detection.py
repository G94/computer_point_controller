'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''


class FaceDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.plugin = None
        self.network = None
        self.input_blob = None
        self.output_blob = None
        self.exec_network = None
        self.infer_request = None


    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        model_xml = model
        model_bin = os.path.splitext(model_xml)[0] + ".bin"     
        self.plugin = IECore()

        self.network = IENetwork(model = model_xml, weights = model_bin)

        supported = self.plugin.query_network(network = self.network, device_name = device)
        unsupported_layers = [l for l in self.network.layers.keys() if l not in supported]
        if    len(unsupported_layers)!=0:
            print("Unsupported format ", unsupported_layers)
    
        if cpu_extension and "CPU" in device:
            self.plugin.add_extension(cpu_extension, device)
        
        self.plugin.load_network(network=self.network, device_name=device) 
        
        self.input_blob = next(iter(self.network.inputs))
        self.output_blob = next(iter(self.network.outputs))   

  

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        try:
            coords, image = self.net.infer({self.input_name: image})
            return   coords, image
        except:
            print("-----predict _ error")


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
