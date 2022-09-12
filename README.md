# geoguessr_final
This is the DL workshop final project.

NOTE: In order to use the interactive demo, you have to download the trained models, which are not present in this GitHub.

They are available at https://drive.google.com/drive/mobile/folders/1jRTMVh1IncvQfJ7vhRjxo3AS_Jz6TJl6?utm_source=en&sort=13&direction=a

Download the directories "1k_best", "3k_best", "5k_best", "100k_best" and "reg_best", 
and place them in the appropriate directories. This may take some time.
If you're not interested in the regression part, download only the first 4 directories.

Usage instructions:

In order to run an image through the classification model, use:

```python3 demo_classifier.py image_path -type -k```, where:

image_path is the path to the image to infer

-type (optional) use this to get the models trained in the first experiment.
Options: -type={resnet,wideresnet,vggbn,inception}. Not using this parameter will choose the ResNet classifiers from experiment 2.

-k (optional) use the model which was trained using top-k accuracy. Options: -k={1,3,5}

In order to run an image through the regression model, use:

```python3 demo_regression.py image_path -type -k -restrict```, where:

image_path is the path to the image to infer

-type (optional) use this to get the models trained in the first experiment (of the classification experiments).
Options: -type={resnet,wideresnet,vggbn,inception}. Not using this parameter will choose the ResNet classifiers from experiment 2.

-k (optional) use the model which was trained using top-k accuracy. Options: -k={1,3,5}

-restrict (optional) restrict the regression model to run only on the top n predictions (according to the classification model). Options: -restrict={1,2,...,15}
