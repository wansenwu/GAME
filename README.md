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

### Dataset and weights

Visual Grounding, Video Grounding and Visual+Video finetune data

Baidu Disk Link: https://pan.baidu.com/s/1KtQeYWptxiakBhkUtVP9bw?pwd=xit8  

#### Visual Grounding Raw Images

Flickr30k dataset：https://uofi.app.box.com/s/1cpolrtkckn4hxr1zhmfg0ln9veo6jpl

MSCOCO dataset：http://images.cocodataset.org/zips/train2014.zip

referit dataset：https://drive.google.com/drive/folders/1-faf4GiPBTwzEItdphhjlIZlP30GIPoc?usp=sharing

#### Note

It is necessary to download all the raw images, and the paths in the finetune data need to match the paths of the raw images in order to run properly.

#### LLM weights

The [llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) and [13b](https://huggingface.co/liuhaotian/llava-v1.5-13b) weights are used. 

#### Projector weights

[Vicuna-13B-v1.5](https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-13b-v1.5)

[Vicuna-7B-v1.5](https://huggingface.co/liuhaotian/llava-v1.5-mlp2x-336px-pretrain-vicuna-7b-v1.5)

### Training and Evaluation

### Train

**For single image grounding:**

param settings in **finetune.sh**

```
--deepspeed ./scripts/zero3.json
--data_path  path/to/vg_ft_cfr_new.json
```

Then, run the script:

```
bash ./LLaVA/scripts/v1_5/finetune.sh
```

**For single video grounding:**

param settings in **finetune.sh**

```
--deepspeed ./scripts/zero3.json
--data_path path/to/video_act.json
```

Then, run the script:

```
bash ./LLaVA/scripts/v1_5/finetune.sh
```

**For joint image+video grounding:**

param settings in **finetune.sh**

```
--deepspeed ./scripts/zero1.json
--data_path path/to/ivg.json
```

Then, run the script:

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

