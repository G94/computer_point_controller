# Computer Pointer Controller

Computer Pointer Controller is a computer vision application that runs Gaze estimation to manage the movement of the pointer. 


## Project Set Up and Installation

### Project Structure
```bash
.
├── _bin
|   ├── demo.mp4
├── _models
|   ├── all the requires models
├── _result
|   ├── videos of the executions
├── _src
|   ├── input_feeder.py
|   ├── mouse_controller.py
|   ├── face_detection.py
|   ├── facial_landmark_detection.py
|   ├── head_pose_estimation.py
|   ├── gaze_estimation.py
├── _main_script.py
├── _README.md
├── requirements.txt
```


### Installation of openvino

- Install openvino on [Windows](https://docs.openvinotoolkit.org/latest/openvino_docs_install_guides_installing_openvino_windows.html)


#### Set up a environment
- Create and environment with Conda:

```console
conda create --name openvino_env

activate openvino_env
```


- install requirements:
Go into the main directory and run, the requirements file
```console
pip3 install -r requirements.txt
```


#### Run setupvars.bat
```console
> cd C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\bin

> setupvars.bat
```


### Download Models
You can download all the models using the [Model Downloader](https://docs.openvinotoolkit.org/latest/omz_tools_downloader_README.html).

#### Arguments:
- name: name of the model to download, for more [information](https://docs.openvinotoolkit.org/latest/usergroup4.html) 
- precisions: you can download the model in different precisions.
- o: The folder output where the model should be stored.


#### Landmark detection
```console
(openvino_env) C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\deployment_tools\open_model_zoo\tools\downloader>python3 downloader.py --name face-detection-adas-binary-0001 --precisions FP16 -o C:\Users\VoxivaAI\Desktop\workspace_gustavo\github\computer_point_controller\models\intel
```

#### Face detection
```console
(openvino_env) C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\deployment_tools\open_model_zoo\tools\downloader>python3 downloader.py --name face-detection-adas-binary-0001 --precisions FP16 -o C:\Users\VoxivaAI\Desktop\workspace_gustavo\github\computer_point_controller\models\intel
```

#### Head pose estimation
```console
(openvino_env) C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\deployment_tools\open_model_zoo\tools\downloader>python3 downloader.py --name head-pose-estimation-adas-0001 --precisions FP16 -o C:\Users\VoxivaAI\Desktop\workspace_gustavo\github\computer_point_controller\models
```

#### Gaze estimation
```console
(openvino_env) C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\deployment_tools\open_model_zoo\tools\downloader>python3 downloader.py --name gaze-estimation-adas-0002 --precisions FP16 -o C:\Users\VoxivaAI\Desktop\workspace_gustavo\github\computer_point_controller\models
```


## Demo

Activate openvino variables
On the terminal run activate your virtual env:

```console
C:\> activate openvino_env
```

Run setupvars, which is located in the following path for Windows:
```console
cd C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\bin

(openvino_env) C:\Program Files (x86)\IntelSWTools\openvino_2020.4.287\bin>setupvars.bat
```


### Run main_script

#### Run on CPU
python main_scripts.py --model models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001 --device CPU --input_type video  --output_path result/


python main_script.py  --fc_model models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001   --hp_model models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001 --device CPU --input_type video  --output_path result/


gaze estimation
(openvino_env) c:\Users\VoxivaAI\Desktop\workspace_gustavo\github\computer_point_controller>python main_script.py  --fc_model models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001   --hp_model models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001  --fl_model models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009 --ge_model models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002  --device CPU --input_type video  --output_path result/






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
