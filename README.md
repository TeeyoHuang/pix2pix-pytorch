# pix2pix-pytorch
the pytorch version of pix2pix 

## Requirments 
- CUDA 8.0+  
- pytorch 0.3.1    
- torchvision  

## Datasets 
- Download a pix2pix dataset (e.g.facades):  
```bash
bash ./datasets/download_pix2pix_dataset.sh facades
``` 
## Train a model:  
```
python pix2pix_train.py --data_root 'your data directory' --which_direction "BtoA"  
```

## Result examples  
### epoch-0  
![image](https://github.com/TeeyoHuang/pix2pix-pytorch/blob/master/result/0.png)   
### epoch-99   
![image](https://github.com/TeeyoHuang/pix2pix-pytorch/blob/master/result/99.png)   
### epoch-199   
![image](https://github.com/TeeyoHuang/pix2pix-pytorch/blob/master/result/199.png)   


## Reference  
[1][Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1703.10593.pdf)
```
@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}

```

## Personal-Blog  
[teeyohuang](https://blog.csdn.net/Teeyohuang/article/details/82699781)
