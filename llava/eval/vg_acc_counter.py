import json

datasets = [  "unc/unc_val",
              "unc/unc_testA",
              "unc/unc_testB",
              "unc+/unc+_val",
              "unc+/unc+_testA",
              "unc+/unc+_testB",
              "gref/gref_val",
              "gref_umd/gref_umd_val",
              "gref_umd/gref_umd_test",
              "flickr/flickr_test",
              "referit/referit_test"]

for dataset in datasets:
    i = 0
    # 打开 JSONL 文件
    path_new = '/ai/test/code/LLaVA/playground/data/eval/vg'
    with open(path_new + '/answers_reg/{}/merge.jsonl'.format(dataset), 'r') as file:
        # 逐行读取文件内容
        for k, line in enumerate(file):
            # 解析 JSON 对象
            json_object = json.loads(line)
            if json_object['iou'] >= 0.5:
                i = i + 1
    acc = float(i/k)
    print(dataset, acc)