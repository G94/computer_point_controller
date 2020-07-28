# Computer Pointer Controller

*TODO:* Write a short introduction to your project

## Project Set Up and Installation
*TODO:* Explain the setup procedures to run your project. For instance, this can include your project directory structure, the models you need to download and where to place them etc. Also include details about how to install the dependencies your project requires.


Installation of openvino


### Download Models
you can download all the models using the model_downloader.


#### Landmark detection



#### Face detection
(openvino_env) C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\deployment_tools\open_model_zoo\tools\downloader>python3 downloader.py --name face-detection-adas-binary-0001 --precisions FP16 -o C:\Users\VoxivaAI\Desktop\workspace_gustavo\github\computer_point_controller\models\intel

#### Head pose estimation


#### Gaze estimation
(openvino_env) C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\deployment_tools\open_model_zoo\tools\downloader>python3 downloader.py --name gaze-estimation-adas-0002 --precisions FP16 -o C:\Users\VoxivaAI\Desktop\workspace_gustavo\github\computer_point_controller\models


## Demo
*TODO:* Explain how to run a basic demo of your model.


Activate openvino variables
On the terminal run

cd C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\bin
 setupvars.bat


### Run main_script
python main_scripts.py --model models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 --device CPU --input_type video  --output_path result/

python main_scripts.py --model models/intel/head-pose-estimation-adas-0001/head-pose-estimation-adas-0001 --device CPU --input_type video  --output_path result/





## Documentation
*TODO:* Include any documentation that users might need to better understand your project code. For instance, this is a good place to explain the command line arguments that your project supports.


## Benchmarks
*TODO:* Include the benchmark results of running your model on multiple hardwares and multiple model precisions. Your benchmarks can include: model loading time, input/output processing time, model inference time etc.


## Results
*TODO:* Discuss the benchmark results and explain why you are getting the results you are getting. For instance, explain why there is difference in inference time for FP32, FP16 and INT8 models.


## Stand Out Suggestions
This is where you can provide information about the stand out suggestions that you have attempted.


### Async Inference
If you have used Async Inference in your code, benchmark the results and explain its effects on power and performance of your project.


### Edge Cases
There will be certain situations that will break your inference flow. For instance, lighting changes or multiple people in the frame. Explain some of the edge cases you encountered in your project and how you solved them to make your project more robust.
