# PNMoG-WSPN
This repository contains the description and the codes for the paper "Individualized Statistical Modeling of Lesions in Fundus Images for Anomaly Detection"

![The PNMoG-WSPN model](https://github.com/yuchendu/PNMoG-WSPN/blob/main/flowchart.jpg)

## Abstract
Anomaly detection in fundus images remains challenging due to the fact that fundus images often contain diverse types of lesions with various properties in locations, sizes, shapes, and colors. Current methods achieve anomaly detection mainly through reconstructing or separating the fundus image background from a fundus image under the guidance of a set of normal fundus images. The reconstruction methods, however, ignore the constraint from lesions. The separation methods primarily model the diverse lesions with pixel-based independent and identical distributed (i.i.d.) properties, neglecting the individualized variations of different types of lesions and their structural properties. And hence, these methods may have difficulty to well distinguish lesions from fundus image backgrounds especially with the normal personalized variations (NPV). To address these challenges, we propose a patch-based non-i.i.d. mixture of Gaussian (MoG) to model diverse lesions for adapting to their statistical distribution variations in different fundus images and their patch-like structural properties. Further, we particularly introduce the weighted Schatten p-norm as the metric of low-rank decomposition for enhancing the accuracy of the learned fundus image backgrounds and reducing false-positives caused by NPV. With the individualized modeling of the diverse lesions and the background learning, fundus image backgrounds and NPV are finely learned and subsequently distinguished from diverse lesions, to ultimately improve the anomaly detection. The proposed method is evaluated on two real-world databases and one artificial database, outperforming the state-of-the-art methods. 
## The Mathematical Model
The overall mathematical model of the proposed method is shown below:

![The mathematical model](https://github.com/yuchendu/PNMoG-WSPN/blob/main/equation.png)

In this model, the healthy fundus image backgrounds are characterized by their low-rank property with Normal Personalized Variations (NPV), constrained by the Schatten p-norm as indicated in the first term. The diverse lesions are regularized by their sparsity and non-i.i.d. properties, represented by the L1 norm and the patch-based multi-MoG terms in the second and third terms, respectively.
## How to Use
### Environment and Necessary Packages
1. The code is implemented on the MATLAB 2018b platform. MATLAB is required to run the code.
2. Users should adjust the folder paths to match their local directories.
3. Some third-party packages related to tensor calculations, such as tenmat.m, are required.
### Running Time and Memory Consumption
1. The algorithm's running time depends on the computer's hardware, the number of images, and the parameter settings. Generally, a more powerful computer, fewer images, and optimal parameter settings result in faster convergence.
2. Parameters such as nv_init and rho affect convergence speed. In our tests, setting nv_init to 7/SingularMax yields the best indicators but requires more iterations to converge. Setting it to 700/SingularMax speeds up convergence with a slight decline in detection accuracy.
3. Memory consumption depends on the data type. Integer data types require less memory but slightly degrade performance. Single and double-precision floating-point data types require more memory but provide higher performance.
### For large patch sizes
As the patch size increases, the dimension of the Gaussian calculations increases dramatically, leading to instability. Our algorithm down-samples patches larger than 2x2 to 2x2 for patch-based Gaussian calculations. To test larger patch sizes, please call the function PNMoG_WSPN_patch.m. 
## Citations
If you use this algorithm, please cite:

@ARTICLE{9966440,
  
  author={Du, Yuchen and Wang, Lisheng and Meng, Deyu and Chen, Benzhi and An, Chengyang and Liu, Hao and Liu, Weiping and Xu, Yupeng and Fan, Ying and Feng, Dagan and Wang, Xiuying and Xu, Xun},
  
  journal={IEEE Transactions on Medical Imaging}, 
  
  title={Individualized Statistical Modeling of Lesions in Fundus Images for Anomaly Detection}, 
  
  year={2023},
  
  volume={42},
  
  number={4},
  
  pages={1185-1196},
  
  keywords={Lesions;Image reconstruction;Anomaly detection;Adaptation models;Measurement;Solid modeling;Sociology;Anomaly detection;non-independent and identical distribution;mixture of Gaussian;weighted Schatten p-Norm;normal personalized variations},
  
  doi={10.1109/TMI.2022.3225422}}
## Contact
If you have any questions, please feel free to contact yuchendu@rocketmail.com
