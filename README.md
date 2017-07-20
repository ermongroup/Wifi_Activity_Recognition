# Wifi_Activity_Recognition using LSTM

Latest dataset & Tensorflow code for IEEE Communication Magazine.  
Title: A Survey on Behaviour Recognition Using WiFi Channel State Information

Work by Siamak Yousefi, Hirokazu Narui, Sankalp Dayal, [Stefano Ermon](http://cs.stanford.edu/~ermon), Shahrokh Valaee

<br/>

## Prerequisite

Tensorflow >= 1.0  
numpy  
pandas  
matplotlib  
scikit-learn  

<br/>

## How to run
1. git clone this repository  
 -> **Notice: The dataset size is ~17GB**
 
2. Run the cross_vali_data_convert_merge.py  
 -> This script makes csv files(input features & label) of each activity in "input_files" folder.　　

3. Run the cross_vali_data_convert.py  
 -> This script makes learning curve images & confusion matrix in a new folder.　　
