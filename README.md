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

## Generated Images
### 1. Toy Dataset
Results from 2-dimensional Gaussian Mixture Models. [Ipython Notebook](https://github.com/ChengBinJin/DiscoGAN-TensorFlow/tree/master/src/jupyter).  
(A) Original GAN  
(B) GAN with Reconstruction Loss  
(C) Domain A to B of DiscoGAN  
(D) Domain B to A of DiscoGAN  

<p align = 'center'>
  <a>
    <img src = 'https://user-images.githubusercontent.com/37034031/46060867-93e4b400-c19f-11e8-841c-33eae9878b6f.gif' width=1000>
  </a>
</p>

### 2. Handbags2Shoes Dataset
- handbag -> shoe -> handbag
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/46061569-17070980-c1a2-11e8-99b5-e4b94a2c5714.png" width=900>
</p>

- shoe -> handbag -> shoe
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/46061802-ce038500-c1a2-11e8-9c5e-94ef78f41a27.png" width=900>
</p>

### 3. edges2shoes
- edge -> shoe -> edge 
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/46062145-de682f80-c1a3-11e8-8bc2-0d19ccceae76.png" width=900>
</p>

- shoe -> edge -> shoe  
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/46062180-f50e8680-c1a3-11e8-9023-28e192511c56.png" width=900>
</p>

### 4. edges2handbags
- edge -> handbag -> edg  
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/46062314-60585880-c1a4-11e8-840c-0aee2d8af55e.png" width=900>
</p>

- handbag -> edge -> handbag
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/46062489-eb395300-c1a4-11e8-8340-389d71b06934.png" width=900>
</p>

### 5. cityscapes
- RGB image -> segmentation label -> RGB image  
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/46062778-d1e4d680-c1a5-11e8-86d9-57092b51e23e.png" width=900>
</p>

- segmentation label -> RGB image -> segmentation label  
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/46062804-e923c400-c1a5-11e8-8afb-27d9860fa307.png" width=900>
</p>

### 6. facades
- RGB image -> segmentation label -> RGB image  
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/46063019-85e66180-c1a6-11e8-87a9-dbbeb9419db1.png" width=900>
</p>

- segmentation label -> RGB image -> segmentation label  
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/46063068-b0d0b580-c1a6-11e8-9717-c09680b7c53b.png" width=900>
</p>

### 7. maps
- RGB image -> segmentation label -> RGB image  
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/46063278-4b30f900-c1a7-11e8-83c5-559d069301ae.png" width=900>
</p>

- segmentation label -> RGB image -> segmentation label  
<p align='center'>
   <img src="https://user-images.githubusercontent.com/37034031/46063306-63a11380-c1a7-11e8-8707-154f59819b9a.png" width=900>
</p>

## Documentation
### Download Dataset
Download `edges2shoes`, `edges2handbags`,  `cityscapes`, `facades`, and `maps` datasets from [pix2pix](https://github.com/phillipi/pix2pix) first. Use the following command to download datasets and copy the datasets on the corresponding file as introduced in **Directory Hierarchy** information.
```
python download.py
```

### Directory Hierarchy
``` 
.
│   DiscoGAN
│   ├── src
│   │   ├── dataset.py
│   │   ├── discogan.py
│   │   ├── download.py
│   │   ├── main.py
│   │   ├── reader.py
│   │   ├── solver.py
│   │   ├── tensorflow_utils.py
│   │   └── utils.py
│   Data
│   ├── cityscapes
│   ├── edges2handbags
│   ├── edge2shoes
│   ├── facades
│   └── maps
```  
**src**: source codes of the WGAN

### Implementation Details
Implementation uses TensorFlow to train the DiscoGAN. Same generator and critic networks are used as described in [DiscoGAN paper](https://arxiv.org/abs/1703.05192). We applied learning rate control that started at 2e-4 for the first 1e5 iterations, and decayed linearly to zero as [cycleGAN](https://github.com/junyanz/CycleGAN). It's helpful to overcome mode collapse problem.  

To respect the original discoGAN paper we set the balance between GAN loss and reconstruction loss are 1:1. Therefore, discoGAN is not good at `A -> B -> A`. However, in the [cycleGAN](https://github.com/junyanz/CycleGAN) the ratio is 1:10. So the reconstructed image is still very similar to input image.  

The official code of [DiscoGAN](https://github.com/SKTBrain/DiscoGAN) implemented by pytorch that used weigt decay. Unfortunately, tensorflow is not support weight deacy as I know. I used regularization term instead of weight decay. So the performance maybe a little different with original one.   

### Training DiscoGAN
Use `main.py` to train a DiscoGAN network. Example usage:

```
python main.py
```
 - `gpu_index`: gpu index, default: `0`
 - `batch_size`: batch size for one feed forward, default: `200`
 - `dataset`: dataset name from [edges2handbags, edges2shoes, handbags2shoes, maps, cityscapes, facades], default: `facades`
 - `is_train`: training or inference mode, default: `True`
 
 - `learning_rate`: initial learning rate for Adam, default: `0.0002`
 - `beta1`: beta1 momentum term of Adam, default: `0.5`
 - `beta2`: beta2 momentum term of Adam, default: `0.999`
 - `weight_decay`: hyper-parameter for regularization term, default: `1e-4`

 - `iters`: number of interations, default: `100000`
 - `print_freq`: print frequency for loss, default: `100`
 - `save_freq`: save frequency for model, default: `10000`
 - `sample_freq`: sample frequency for saving image, default: `500`
 - `sample_batch`: number of sampling images for check generator quality, default: `200`
 - `load_model`: folder of save model that you wish to test, (e.g. 20180907-1739). default: `None` 

### Test DiscoGAN
Use `main.py` to test a DiscoGAN network. Example usage:

```
python main.py --is_train=false --load_model=folder/you/wish/to/test/e.g./20180926-1739
```
Please refer to the above arguments.

### Citation
```
  @misc{chengbinjin2018discogan,
    author = {Cheng-Bin Jin},
    title = {DiscoGAN-tensorflow},
    year = {2018},
    howpublished = {\url{https://github.com/ChengBinJin/DiscoGAN-TensorFlow}},
    note = {commit xxxxxxx}
  }
```
 
### Attributions/Thanks
- This project refered some code from [carpedm20](https://github.com/carpedm20/DiscoGAN-pytorch) and [GunhoChoi](https://github.com/GunhoChoi/DiscoGAN-TF).  
- Some readme formatting was borrowed from [Logan Engstrom](https://github.com/lengstrom/fast-style-transfer)

## License
Copyright (c) 2018 Cheng-Bin Jin. Contact me for commercial use (or rather any use that is not academic research) (email: sbkim0407@gmail.com). Free for research use, as long as proper attribution is given and this copyright notice is retained.

## Related Projects
- [Vanilla GAN](https://github.com/ChengBinJin/VanillaGAN-TensorFlow)
- [DCGAN](https://github.com/ChengBinJin/DCGAN-TensorFlow)
- [WGAN](https://github.com/ChengBinJin/WGAN-TensorFlow/blob/master/README.md)
- [pix2pix](https://github.com/ChengBinJin/pix2pix-tensorflow)

