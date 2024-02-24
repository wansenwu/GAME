# GAME：Grounding Any Modalities via LLM

### Installation

1. Clone this repository and navigate to folder

   ```
   git clone https://github.com/haotian-liu/LLaVA.git
   cd LLaVA
   ```

2. Install Package

   ```
   pip install -r requirements.txt
   ```

### Dataset

Visual Grounding, Video Grounding and Visual+Video finetune data

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

