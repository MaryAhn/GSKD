# GSKD
**Global Structural Similarity Knowledge Distillation**(Transaction of Image Processing Submitted)
Keon-hee Ahn, Hyejin Park, Hyesong Choi, and Dongbo Min

## Abstract
Knowledge distillation (KD) has emerged as an appealing solution for effectively compressing deep neural networks for semantic segmentation tasks. While fine-grained structural similarity is crucial in many KD approaches and enhances performance in semantic segmentation, it often overlooks more abstract semantic contextual information. In this paper, we introduce a novel method for transferring global structural knowledge. This method aims to complement existing fine-grained structural knowledge by extracting more abstract attributes from the feature maps of both teacher and student models. Initially, our method performs class-balanced sampling on these feature maps to gather a representative set of samples, we then construct a pair-wise similarity map, capturing the global structural similarity. Subsequently, this map is shuffled row-wise and grouped column-wise to form sub-image descriptors, which are conceptualized as semi-global structural knowledge to be transferred to the student model. By repeating this process for all sub-image descriptors, we ensure a comprehensive transfer of global structural knowledge. Experimental results demonstrate that our proposed approach, when integrated with various state-of-the-art KD methods, significantly enhances performance. 

## Introduction
1. Deploying deep neural networks (DNN) for semantic segmentation in the real world is often problematic due to their high complexity even if DNN models have shown remarkable performance [1,2].
2. Due to the above reason, model compression approaches have been proposed, especially, knowledge distillation (KD) has been considered an efficient and practical approach [3,4,5], in which a cumbersome (teacher) model supervises a compact and lightweight (student) model by transferring knowledge learned from the teacher to the student model.
3. In this paper, we propose a novel method to leverage global structural knowledge on the KD of semantic segmentation, intending to complement the existing fine-grained structural relationships [3,5,6], by effectively extracting more abstract attributes in the feature maps of the teacher and student models in a plug-in-play manner.
4. Our method produces a ___semi-global descriptor___ from the teacher model and the student model respectively, through a row-wise shuffled and column-wise grouped similarity map from the class-balanced sampled points, then minimizes the difference between these descriptors.
5. Extensive experiments proved the effectiveness of our method, especially, DIST [7] with our framework (mIoU 78.17) outperforms the teacher model (mIoU 78.01).

## Overview
![Figure2](https://github.com/MaryAhn/GSKD/assets/43198379/6f75c4f3-1605-4871-bdf4-3be5ef4e7c15)
Overview of the proposed Global Structural Similarity Knowledge Distillation (GSKD). The GSKD starts with sampling $\mathcal{B}$ pixels from the feature map of the teacher (or student) model in a class-balanced manner and computes the global self-similarity map $V_t$ (or $V_s$) with a size of $\mathcal{B} \times \mathcal{B}$. The global self-similarity map is divided into sub-image descriptors by row-wise shuffling and column-wise grouping. The sub-image descriptors are used as the knowledge to distill the global structural similarity of the teacher to the student.

## Paper
[Paper link](https://drive.google.com/file/d/1BDWb4muBBCQKPDQDr56n8pRqc9RggCSa/view?usp=drive_link)

## Acknowledgement
The code is mostly based on the code in [CIRKD](https://github.com/winycg/CIRKD).

## Reference

[1] Chen, L.-C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L. (2017). Deeplab: Semantic image segmentation with deep convolutional nets, atrous convolution, and fully connected crfs. IEEE Transactions on Pattern Analysis and Machine Intelligence, 40(4), 834–848.

[2] Zhao, H., Shi, J., Qi, X., Wang, X., & Jia, J. (2017). Pyramid scene parsing network. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2881–2890.

[3] Liu, Y., Chen, K., Liu, C., Qin, Z., Luo, Z., & Wang, J. (2019). Structured knowledge distillation for semantic segmentation. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2604–2613.

[4] Wang, Y., Zhou, W., Jiang, T., Bai, X., & Xu, Y. (2020). Intra-class Feature Variation Distillation for Semantic Segmentation. Proceedings of the European Conference on Computer Vision.

[5] Xie, J., Shuai, B., Hu, J.-F., Lin, J., & Zheng, W.-S. (2018). Improving fast segmentation with teacher-student learning. ArXiv Preprint ArXiv:1810.08476.

[6] Yang, C., Zhou, H., An, Z., Jiang, X., Xu, Y., & Zhang, Q. (2022). Cross-image relational knowledge distillation for semantic segmentation. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 12319–12328.

[7] Huang, T., You, S., Wang, F., Qian, C., & Xu, C. (2022). Knowledge distillation from a stronger teacher. Advances in Neural Information Processing Systems, 35, 33716–33727.
