# PNMoG-WSPN
This repository contains the description and the codes for the paper "Individualized Statistical Modeling of Lesions in Fundus Images for Anomaly Detection"
# Abstract
Anomaly detection in fundus images remains challenging due to the fact that fundus images often contain diverse types of lesions with various properties in locations, sizes, shapes, and colors. Current methods achieve anomaly detection mainly through reconstructing or separating the fundus image background from a fundus image under the guidance of a set of normal fundus images. The reconstruction methods, however, ignore the constraint from lesions. The separation methods primarily model the diverse lesions with pixel-based independent and identical distributed (i.i.d.) properties, neglecting the individualized variations of different types of lesions and their structural properties. And hence, these methods may have difficulty to well distinguish lesions from fundus image backgrounds especially with the normal personalized variations (NPV). To address these challenges, we propose a patch-based non-i.i.d. mixture of Gaussian (MoG) to model diverse lesions for adapting to their statistical distribution variations in different fundus images and their patch-like structural properties. Further, we particularly introduce the weighted Schatten p-norm as the metric of low-rank decomposition for enhancing the accuracy of the learned fundus image backgrounds and reducing false-positives caused by NPV. With the individualized modeling of the diverse lesions and the background learning, fundus image backgrounds and NPV are finely learned and subsequently distinguished from diverse lesions, to ultimately improve the anomaly detection. The proposed method is evaluated on two real-world databases and one artificial database, outperforming the state-of-the-art methods. 
# The Mathematical Model
\begin{equation}\label{equ:Model}
\begin{aligned}
\min\|\mathcal{B}_{(3)}\|_{\omega,S_{p}}^{p}
+&\beta\|\mathcal{F}\|_{1}\\
-\lambda\sum_{l}&\sum_{n_{p}}log\sum_{k}\pi_{kl}\mathcal{N}\left(f\left(\mathcal{F}\right)_{n_{p}l}|\mu_{kl},\Sigma_{kl}\right)\\
s.t.\quad \mathcal{X}=\mathcal{B}+&\mathcal{F}
\end{aligned}
\end{equation}
