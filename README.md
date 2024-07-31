# arcface-pytorch
pytorch implement of arcface 

## Installation
```bash
conda env create -f environment.yml
conda activate arcface
```

## Face Recognition with OpenCV

This application uses OpenCV to capture images from your webcam. When you press the `1` key, the current frame is saved if a face is detected. The application then starts recognizing faces in subsequent frames. Release the saved face by press the `2` key.
1. Navigate to the project root directory
2. Run the script
```bash
  python recog_cam.py
```

## Face Capture with OpenCV
This application uses OpenCV to capture images from your webcam. It saves the captured face image and the embedded vector from the model in a local dataframe. The script then uses t-SNE for 2D visualization plot.

1. Navigate to the project root directory
2. Run the script The dataframe will be saved in `{root}/df/`
```bash
  python capture.py
```
3. Run the notebook
- Navigate to `{root}/notebook/tsne.ipynb` to reduce dimensions.
4. Run the plot script
```bash
  python plot.py
```

# References
https://github.com/deepinsight/insightface

https://github.com/auroua/InsightFace_TF

https://github.com/MuggleWang/CosFace_pytorch

# pretrained model and lfw test dataset
the pretrained model and the lfw test dataset can be download here. link: https://pan.baidu.com/s/1tFEX0yjUq3srop378Z1WMA pwd: b2ec
the pretrained model use resnet-18 without se. Please modify the path of the lfw dataset in config.py before you run test.py.
