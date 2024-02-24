import json
import numpy as np
from llava.eval.model_video_g import video_evaluate
datasets = ['TACoS', 'Charades']


for dataset in datasets:
    root_dir = '/ai/test/code/LLaVA/playground/data/eval/video_g/answers_1219_video_image/{}/merge.jsonl'.format(dataset)
    pred_windows = []
    gt_windows = []
    with open(root_dir, 'r') as file:
        # 逐行读取文件内容
        for k, line in enumerate(file):
            # 解析 JSON 对象
            try:
                json_object = json.loads(line)
                if 'question_id' in json_object:
                    pred_windows.append(json_object['text'])
                    gt_windows.append(json_object['gt_box'])
            except json.decoder.JSONDecodeError:
                pass
    print('total_length:', len(pred_windows))
    miou_at_one, iou_thd2recall_at_one = video_evaluate(np.array(pred_windows), np.array(gt_windows))
    print(dataset, "miou:", miou_at_one, iou_thd2recall_at_one)
