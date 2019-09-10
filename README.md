# Wifi_Activity_Recognition using LSTM

Latest dataset & Tensorflow code for IEEE Communication Magazine.  
Title: [A Survey on Behaviour Recognition Using WiFi Channel State Information](http://ieeexplore.ieee.org/document/8067693/)

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
0. Download dataset from [here](https://drive.google.com/file/d/19uH0_z1MBLtmMLh8L4BlNA0w-XAFKipM/view?usp=sharing)  
 -> **Notice: Dataset size is ~4GB**  

1. "git clone" this repository.  
 
2. Run the cross_vali_data_convert_merge.py  
 -> This script makes csv files(input features & label) of each activity in "input_files" folder.　　

3. Run the cross_vali_recurrent_network_wifi_activity.py 
 -> This script makes learning curve images & confusion matrix in a new folder.　　

## Dataset
We collect dataset using [Linux 802.11n CSI Tool](https://dhalperi.github.io/linux-80211n-csitool/).  

The files with "input_" prefix are WiFi Channel State Information data.  
 -> 1st column shows timestamp.  
 -> 2nd - 91st column shows (30 subcarrier * 3 antenna) amplitude.  
 -> 92nd - 181st column shows (30 subcarrier * 3 antenna) phase.
 
The files with "annotation_" prefix are annotation data.

## Jupyter notebook
[PCA_STFT](https://github.com/ermongroup/Wifi_Activity_Recognition/blob/master/PCA_STFT_visualize.ipynb) file visualize the data from .csv file. This code refers to [CARM](https://www.cse.msu.edu/~alexliu/publications/WeiCARM/WeiCARM_MOBICOM15.pdf).
