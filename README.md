# Video GAN Experiment Platform
Video GAN Experiment Platform was developed by Bell Chen in Feb 2018 for testing neural network structure of video generation. This platform is powered by python, pytorch, ffmpeg, and other dependences. It is integrated with muiltiple functions that are needed to do experiment, like spliting videos, generating datasets, training different networks, and generating a video with comparison of original data. I designed a frame traverse slider which can generate arbitrary sequence of frames as network’s input.

**All the experiment details are in my blog.
https://bellchen.me/3d-gan-for-video-generation/

# Frame Slider
During experiments, due to the slow computational speed of python object class, I optimized the allocation of GPU and CPU by reducing algorithm complexity and increasing batch-size so that the network-trainer can occupy full GPU with least interval caused by slow “for” loop computed by CPU. According to test result, the old version of frame slider occupy 96% of GPU usage. In comparison, the new slider can use full GPU core. On average, the training speed on laptop with GTX960M rises 30%.

# Network Structure
There are 2 main neural networks integrated in this platform - one generator and one discriminator. The loss of two network can be set to DC-GAN loss and mixed losses. I designed fully-convolutional network that doubles the number of channels for every layer when forward-propagating the network. In generator, there are several type of networks that can be mannually selected using argument '-net'. In this version, it has UNet-based generator and ResNet-based generator. Because of the copyright, I will not release ResNet-based generator V3-5's codes until formal paper is published.

# Installation
This network was initially developed on PyTorch 0.3. But after tests, it can successfully run on PyTorch 0.4, so we do not care the version of dependencies, just upgrade to the newest version.

1. Anaconda (strongly recommend):
  https://conda.io/docs/user-guide/install/index.html
2. PyTorch (GPU is strongly recommended):
  https://pytorch.org/
3. ffmpeg: 
  macOS terminal: 
  ```
  brew install ffmpeg
  ```
  Ubuntu terminal:
  ```
  sudo apt-get install ffmpeg
  ```

# Usage
Clone this code to your computer and locate into the main folder. Just run 'main.py' in python.
terminal:
```
  python main.py [arguments]
```
optional arguments:
  -h, --help            Show this help message and exit  
  --train, -t           Start a new training session.  
  --resume, -r          Resume training from checkpoint.  
  --split_video, -s     Split video for training images.  
  --learning_rate LEARNING_RATE, -lr  
                        Learning rate of network.  
  --batch_size BATCH_SIZE, -batch  
                        Batch size in training step.  
  --lr_decay_epoch LR_DECAY_EPOCH, -decay  
                        Epoch number where learning rate starts to decay.  
  --log_frequency LOG_FREQUENCY, -log  
                        Logging frequency in training steps.  
  --sample_frequency SAMPLE_FREQUENCY, -sample  
                        Sampling frequency in training epochs.  
  --max_epoch MAX_EPOCH, -epoch  
                        Max epochs in training.
  --dataset_a_dir DATASET_A_DIR, -a  
                        Training-set frames location.  
  --dataset_b_dir DATASET_B_DIR, -b  
                        Teacher-set frames location.  
  --sample_dir SAMPLE_DIR, -test  
                        Sample-set frames location.  
  --generate_video, -g  Generate video result in sampling.  
  --result_dir RESULT_DIR, -result  
                        Sampling result location.  
  --num_workers NUM_WORKERS, -worker  
                        The number of dataset workers  
  --video_a_input VIDEO_A_INPUT, -va  
                        File name of input video for training-set.  
  --video_b_input VIDEO_B_INPUT, -vb  
                        File name of input video for teacher-set.  
  --video_output_dir VIDEO_OUTPUT_DIR, -vo  
                        File name of input video for teacher-set.  
  --video_sample_input VIDEO_SAMPLE_INPUT, -vs  
                        File name of input video for sampling.  
  --fps FPS, -fps FPS   FPS of input and generated video.  
  --network NETWORK, -net  
                        Training-set frames location.  
  --image_size IMAGE_SIZE, -img  
                        The number of downsampling layer in network.  
  --channel CHANNEL, -c  
                        The number of output channels in translators first  
                        downsampling layer.  
  --num_block NUM_BLOCK, -block  
                        The number of resnet blocks in network.  
  --num_downsample NUM_DOWNSAMPLE, -downsample  
                        The number of downsampling layer in network.  
  --u_depth U_DEPTH, -ud  
                        The depth of U-Net.  
  --u_channel U_CHANNEL, -uc  
                        The number of channels in the first layer of U-Net.  
  --pool_size POOL_SIZE, -pool  
                        The length of frame sequence as network input.  
  --train_step TRAIN_STEP, -t_step  
                        The step of frame sequence.  
  --sample_step SAMPLE_STEP, -s_step  
                        The length of frame sequence as network input.  
              
# Copyright
This code is owned by Bell Chen. You can use it and edit it anyway you want only for non-commercial education and personal research. If you are plan to publish a non-commercial research paper using this code, please note the original author of this code and let me know. If you are using this platform for any commercial use including research and development, please contact me and ask permission.

The copyright of the code is inviolable. If there is any commercial use, please contact the author.  
Los derechos de autor del código son inviolables. Si existe algún uso comercial, comuníquese con el autor.  
Le droit d'auteur du code est inviolable.S'il y a une utilisation commerciale, veuillez contacter l'auteur.  
代码版权不容侵犯，如有任何商业使用，请联系作者  
代碼版權不容侵犯，如有任何商業使用，請聯繫作者  
コードの著作権は侵害されません。商業的に使用される場合は、著者に連絡してください。  
이 코드의 저작권은 불가침합니다. 상업적 용도가있는 경우 저자에게 문의하십시오.  
कोड का कॉपीराइट अचूक है। यदि कोई वाणिज्यिक उपयोग है, तो कृपया लेखक से संपर्क करें।  

Contact Email: chenbell [at] live.com
