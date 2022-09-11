# geoguessr_final
This is the DL workshop final project.

Usage instructions:
1. In order to run an image through the classification model, use:
```python3 demo_classifier.py image_path -type -k```, where:
image_path is the path to the image to infer
-type (optional) use this to get the models trained in the first experiment.
Options: -type={resnet,wideresnet,vggbn,inception}. Not using this parameter will choose the ResNet classifiers from experiment 2.
-k (optional) use the model which was trained using top-k accuracy. Options: -k={1,3,5}
2. In order to run an image through the regression model, use:
```python3 demo_regression.py image_path -type -k -restrict```, where:
image_path is the path to the image to infer
-type (optional) use this to get the models trained in the first experiment (of the classification experiments).
Options: -type={resnet,wideresnet,vggbn,inception}. Not using this parameter will choose the ResNet classifiers from experiment 2.
-k (optional) use the model which was trained using top-k accuracy. Options: -k={1,3,5}
-restrict (optional) restrict the regression model to run only on the top n predictions (according to the classification model). Options: -restrict={1,2,...,15}
