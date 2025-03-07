# Skelite: Compact Neural Networks for Efficient Iterative Skeletonization

Code repository for _Skelite: Compact Neural Networks for Efficient Iterative Skeletonization_. Training data is being added soon.


## Abstract

Skeletonization extracts thin representations from images that compactly encode their geometry and topology. These representations have become an important topological prior for preserving connectivity in curvilinear structures, aiding medical tasks like vessel segmentation. Existing compatible skeletonization algorithms face significant trade-offs: morphology-based approaches are computationally efficient but prone to frequent breakages, while topology-preserving methods require substantial computational resources. 

We propose a novel framework for training iterative skeletonization algorithms with a learnable component. The framework leverages synthetic data, task-specific augmentation, and a model distillation strategy to learn compact neural networks that produce thin, connected skeletons with a fully differentiable iterative algorithm. 

Our method demonstrates a $100\times$ speedup over topology-constrained algorithms while maintaining high accuracy and generalizing effectively to new domains without fine-tuning. Benchmarking and downstream validation in 2D and 3D tasks demonstrate its computational efficiency and real-world applicability.

