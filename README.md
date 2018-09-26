# DiscoGAN-TensorFlow
This repository is a Tensorflow implementation of [DiscoGAN](https://arxiv.org/abs/1703.05192), ICML2017.

<p align='center'>
  <img src="https://user-images.githubusercontent.com/37034031/46002379-e020ed00-c0e8-11e8-81d1-3ee153c6850f.png" width=600)
</p>  
  
* *All samples in README.md are genearted by neural network except the first image for each row.*
  
## Requirements
- tensorflow 1.10.0
- python 3.5.3
- numpy 1.14.2
- opencv 3.2.0
- matplotlib 2.2.2
- scipy 0.19.1
- pillow 5.0.0

## Applied GAN Structure
1. **Generator**
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/46003429-7c4bf380-c0eb-11e8-9892-c4e42eaf31e4.png" width=400>
</p>

2. **Discriminator**
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/46003548-c2a15280-c0eb-11e8-8b58-078f20aec279.png" width=450>
</p>

## Toy Dataset
Results from 2-dimensional Gaussian Mixture Models. [Ipython Notebook](https://github.com/ChengBinJin/DiscoGAN-TensorFlow/tree/master/src/jupyter).

<p align = 'center'>
  <a>
    <img src = 'https://user-images.githubusercontent.com/37034031/46060867-93e4b400-c19f-11e8-841c-33eae9878b6f.gif' width=800>
  </a>
</p>
