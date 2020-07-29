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

### Language setting
| Details            |              |
|-----------------------|---------------|
| Programming Language: |  Python 3.5 or 3.6 |

### Hardware

* 6th to 10th generation Intel® Core™ processor with Iris® Pro graphics or Intel® HD Graphics.




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
Now that openvino environment and setupvars are configure, we can run our main script.

#### Run on CPU
```console
(openvino_env) c:\Users\VoxivaAI\Desktop\workspace_gustavo\github\computer_point_controller> python main_script.py  --fc_model models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001   --hp_model models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001  --fl_model models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009 --ge_model models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002  --device CPU --input_type video  --output_path result/
```


## Documentation
Arguments for main_script:

* --fc_model: path to the face detection model
* --hp_model: path to the head pose estimation model
* --fl_model: path to the facial landmark detection model
* --ge_model: path to the gaze estimation model
* --device: name of the device
* --input_type: type of file will be feed
* --video:  path to the video file
* --output_path: path to save the results


## Benchmarks
Sum of inference time for each model.
![Inference Time](img/inference_time.PNG)

Average inference time for each model.
![Mean Inference Time](img/mean_inference_time.PNG)


### Model size
The size of the model pre- and post-conversion was...
| |face-detection-adas-binary-0001|head-pose-estimation-adas-0001|landmarks-regression-retail-0009|gaze-estimation-adas-0002|
|-|-|-|-|-|
|BIN|1.7 MB|3.7 MB|373 KB|3.6 MB|
|XML|114 KB|50 KB|42 KB|65 KB|
|PRECISION|FP32|FP16|FP16|FP16|


## Results and next steps
* The Face detection takes most of the time to infer it might be the bottleneck.
* Although, Gaze detection have several process and inputs it only takes 0.001 seconds in average.
* It's necessary optimize the bottleneck using VTune.


## Edge Cases
In the near future the project will be tested on other devices like the NCS2.
