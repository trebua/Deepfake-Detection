# Deepfake-Detection
Project in the module "CS4225/CS5425 Big Data Systems for Data Science" at the National University of Singapore.

## Prerequisites
* OpenCV: ```python -m pip install opencv-python```
* CVLib: ```python -m pip install cvlib```

## Dataset
The dataset should be stored in a folder named *data*, using the dataset gathered with these instructions:
https://github.com/ondyari/FaceForensics/blob/master/dataset/README.md

The preprocessing is now done with the assumption that the following commands are run:

```python download-FaceForensics.py path/to/project/data -d DeepFakeDetection -c c40 -t videos```

and

```python download-FaceForensics.py path/to/project/data -d DeepFakeDetection_original -c c40 -t videos```


