# [Pytorch] PixelRL


Implementation of PixelRL model in **Fully Convolutional Network with Multi-Step Reinforcement Learning for Image Processing** paper with Pytorch.

I currently implemented 3 denoise models in the paper, I'm going to update other models soon.


## Train
You run this command to begin the training:
```
python train.py --sigma=25             \
                --episodes=2000        \
                --batch-size=64        \
                --save-every=200       \
                --ckpt-dir='checkpoint'
```

**NOTE**: if you want to re-train a new model, you should delete all files in sub-directories in **checkpoint** directory. Your checkpoint will be saved when above command finishs and can be used for the next times, so you can train a model on Google Colab without taking care of GPU time limit.

## Demo 
After Training, you can test models with this command, the results are in **results** directory.
```
python test.py --sigma=15              \
               --save-images=1         \
               --model-path="checkpoint/15/model-15.pt"
```

