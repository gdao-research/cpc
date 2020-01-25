# Contrastive Predictive Coding

Reproduce of [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/pdf/1807.03748.pdf) paper in image recognition task.

I trained CPC to encode image into a representation vector which significantly has lower dimension.
The linear model applied directly on pixel image only reached 54% accuracy.
However, after transform into representation vector, the image recognition task achieved 90% accuracy.
This show that good representation has been learned through the unsupervised learning process.

## How to Use
- Build Docker image from Dockerfile
- Learn representation transformation model with ```python main.py```
- Run benchmark model on raw pixel with ```python linear_benchmark.py```
- Run linear CPC to compare with ```python linear_cpc.py```
