# GSKD
Official implementation of **Global Structural Similarity Knowledge Distillation**(Transaction of Image Processing Submitted)

## Introduction
1. Deploying deep neural networks (DNN) for semantic segmentation in the real world is often problematic due to their high complexity even if DNN models have shown remarkable performance\cite{chen2017deeplab,zhao2017pyramid}.
2. Due to the above reason, model compression approaches have been proposed, especially, knowledge distillation (KD) has been considered an efficient and practical approach, in which a cumbersome (teacher) model supervises a compact and lightweight (student) model by transferring knowledge learned from the teacher to the student model.
3. In this paper, we propose a novel method to leverage global structural knowledge on the KD of semantic segmentation, intending to complement the existing fine-grained structural relationships\cite{skd,xie2018improving,yang2022cross}, by effectively extracting more abstract attributes in the feature maps of the teacher and student models in a plug-in-play manner.
4. Our method produces a ___semi-global descriptor___ from the teacher model and the student model respectively, through a row-wise shuffled and column-wise grouped similarity map from the class-balanced sampled points, then minimizes the difference between these descriptors.
5. Extensive experiments proved the effectiveness of our method, especially, DIST with our framework (mIoU 78.17) outperforms the teacher model (mIoU 78.01).

## Overview
![Figure2](https://github.com/MaryAhn/GSKD/assets/43198379/6f75c4f3-1605-4871-bdf4-3be5ef4e7c15)
Overview of the proposed Global Structural Similarity Knowledge Distillation (GSKD). The GSKD starts with sampling $\mathcal{B}$ pixels from the feature map of the teacher (or student) model in a class-balanced manner and computes the global self-similarity map $V_t$ (or $V_s$) with a size of $\mathcal{B} \times \mathcal{B}$. The global self-similarity map is divided into sub-image descriptors by row-wise shuffling and column-wise grouping. The sub-image descriptors are used as the knowledge to distill the global structural similarity of the teacher to the student.

## Paper
[Paper link](https://drive.google.com/file/d/1BDWb4muBBCQKPDQDr56n8pRqc9RggCSa/view?usp=drive_link)
