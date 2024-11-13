# PNMoG-WSPN
This repository contains the description and the codes for the paper "Individualized Statistical Modeling of Lesions in Fundus Images for Anomaly Detection"

![The PNMoG-WSPN model](https://github.com/yuchendu/PNMoG-WSPN/blob/main/flowchart.jpg)

## Abstract
Anomaly detection in fundus images remains challenging due to the fact that fundus images often contain diverse types of lesions with various properties in locations, sizes, shapes, and colors. Current methods achieve anomaly detection mainly through reconstructing or separating the fundus image background from a fundus image under the guidance of a set of normal fundus images. The reconstruction methods, however, ignore the constraint from lesions. The separation methods primarily model the diverse lesions with pixel-based independent and identical distributed (i.i.d.) properties, neglecting the individualized variations of different types of lesions and their structural properties. And hence, these methods may have difficulty to well distinguish lesions from fundus image backgrounds especially with the normal personalized variations (NPV). To address these challenges, we propose a patch-based non-i.i.d. mixture of Gaussian (MoG) to model diverse lesions for adapting to their statistical distribution variations in different fundus images and their patch-like structural properties. Further, we particularly introduce the weighted Schatten p-norm as the metric of low-rank decomposition for enhancing the accuracy of the learned fundus image backgrounds and reducing false-positives caused by NPV. With the individualized modeling of the diverse lesions and the background learning, fundus image backgrounds and NPV are finely learned and subsequently distinguished from diverse lesions, to ultimately improve the anomaly detection. The proposed method is evaluated on two real-world databases and one artificial database, outperforming the state-of-the-art methods. 
## The Mathematical Model
The overall mathematical model of the proposed method is listed as below:

![The mathematical model](https://github.com/yuchendu/PNMoG-WSPN/blob/main/equation.png)

Where the healthy fundus image backgrounds are modeled by their low-rank property with NPV, which is constrained by Schatten p-norm shown as the first item in the above model. The diverse lesions are regularized by both sparsity property and non-i.i.d. property, shown in the second item (L-1 norm) and the third item (patch-based multi-MoG) in the above model, respectively.
## How to use
### Environment and necessary packages
1. The codes are implemented under MATLAB 2018b platform. A MATLAB software must be required to run the code. 
2. The users should change the folders directions to their local directions.
3. Some tensor calculation related third-party packages are required, such as tenmat.m
### Running time consuming and memory consuming
