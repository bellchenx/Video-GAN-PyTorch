# Video GAN Experiment Platform
## Introduction
Video GAN Experiment Platform was developed by Bell Chen in Feb 2018 for testing neural network structure of video generation. This platform is powered by python, pytorch, ffmpeg, and other dependences.  

It is integrated with muiltiple functions that are needed to do experiment, like spliting videos, generating datasets, training different networks, and generating a video with comparison of original data.  

All the experiment details are in my blog.
https://bellchen.me/3d-gan-for-video-generation/

## Network Structure
There are 2 main neural networks integrated in this platform - one generator and one discriminator. The loss of two network can be set to DC-GAN loss and mixed losses. I designed fully-convolutional network that doubles the number of channels for every layer when forward-propagating the network.  

In generator, there are several type of networks that can be mannually selected using argument '-net'. In this version, it has UNet-based generator and ResNet-based generator. Because of the copyright, I will not release ResNet-based generator V3-5's codes until formal paper is published.

## Dataset
I designed a frame traverse slider which can generate arbitrary sequence of frames as network’s input. You can set length of time dimension of 3D convolutional network whatever you want. Please remenber that large 3D Conv-Net will use large amount of memory of your video card.  

During experiments, due to the slow computational speed of python object class, I optimized the allocation of GPU and CPU by reducing algorithm complexity and increasing batch-size so that the network-trainer can occupy full GPU with least interval caused by slow “for” loop computed by CPU.  

And also, it will load all of data to your CPU memory to speed up training because the speed of copying several images from  hard disk to CPU memory will limit the speed of copying data to GPU memory. So make sure you have a large swap memory space to contain all the data.

According to test result, the old version of frame slider occupy 96% of GPU usage. In comparison, the new slider can use full GPU core. On average, the training speed on laptop with GTX960M rises 30%.

## Installation
This network was initially developed on PyTorch 0.3. But after tests, it can successfully run on PyTorch 0.4, so we do not care the version of dependencies, just upgrade to the newest version.

1. Anaconda (recommend):
  https://conda.io/docs/user-guide/install/index.html
2. PyTorch (GPU is strongly recommended):
  https://pytorch.org/
3. ffmpeg (needed for some functions):  
  macOS terminal: 
  ```
  brew install ffmpeg
  ```
  Ubuntu terminal:
  ```
  sudo apt-get install ffmpeg
  ```

## Usage
Clone this code to your computer and locate into the main folder. Just run 'main.py' in python.  
```
  python main.py [arguments]
```
Detailed usage of optinal argments can be found using:
```
  python main.py -h
```
Or editing 'main.py' to see the function of optional argments.
              
## Copyright
This code is owned by Bell Chen. You can use it and edit it anyway you want only for non-commercial education and personal research. If you are plan to publish a non-commercial research paper using this code, please note the original author of this code and let me know.  

If you are using this platform for any commercial use including research and development, please contact me and ask permission.  

Contact Email: chenbell [at] live.com  
'''
The copyright of the code is inviolable. If you want any commercial use, please contact the author.  
Los derechos de autor del código son inviolables. Si existe algún uso comercial, comuníquese con el autor.  
Le droit d'auteur du code est inviolable. S'il y a une utilisation commerciale, veuillez contacter l'auteur.  
代码版权不容侵犯，如有任何商业使用，请联系作者。  
代碼版權不容侵犯，如有任何商業使用，請聯繫作者。  
コードの著作権は侵害されません。商業的に使用される場合は、著者に連絡してください。  
이 코드의 저작권은 불가침합니다. 상업적 용도가있는 경우 저자에게 문의하십시오.  
कोड का कॉपीराइट अचूक है। यदि कोई वाणिज्यिक उपयोग है, तो कृपया लेखक से संपर्क करें।  
'''
