# Deep Dream

参考 [DeepDream的代码实现](https://sherlockliao.github.io/2017/07/23/deep_dream_code/)

use vgg16 to dream.
you can add your own network to train.(in `model.py`)

## install

need `pytorch` 

## How to Use

usage: main.py [-h] [--layerNum LAYERNUM] [--octave_n OCTAVE_N]
               [--octave_scale OCTAVE_SCALE] [--savename SAVENAME]
               [--savedir SAVEDIR] [--guide GUIDE]
               [--learning_rate LEARNING_RATE]
               [--num_iterations NUM_ITERATIONS]
               imgpath

### examples

```bash
python main.py -h # show help
```

```bash
python main.py ./imgs/sky.jpg # save in tmpDirPath['./result']
```

```bash
python main.py ./imgs/sky.jpg --guide ./imgs/flower.jpg # use flower guide dream
```
