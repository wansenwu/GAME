# GAME：Grounding Any Modalities via LLM

### Installation

1. Clone this repository 

   ```
   git clone https://github.com/wansenwu/GAME.git
   ```
   
2. Install Package

   ```
   pip install -r requirements.txt
   ```

### Dataset

Visual Grounding, Video Grounding and Visual+Video finetune data

Baidu Disk Link: https://pan.baidu.com/s/1KtQeYWptxiakBhkUtVP9bw?pwd=xit8  

#### Visual Grounding Raw Images

Flickr30k dataset：https://uofi.app.box.com/s/1cpolrtkckn4hxr1zhmfg0ln9veo6jpl
MSCOCO dataset：http://images.cocodataset.org/zips/train2014.zip
referit dataset：https://drive.google.com/drive/folders/1-faf4GiPBTwzEItdphhjlIZlP30GIPoc?usp=sharing

#### Note

It is necessary to download all the raw images, and the paths in the finetune data need to match the paths of the raw images in order to run properly.

### Training and Evaluation

#### Train 

```
bash ./LLaVA/scripts/v1_5/finetune.sh
```

#### Eval

For image grounding 

```
bash ./LLaVA/scripts/v1_5/eval/vg.sh
```

For video grounding 

```
bash ./LLaVA/scripts/v1_5/eval/video.sh
```

