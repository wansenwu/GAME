import json
import os
import re
import torch
import os.path as osp
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from tqdm import tqdm
import copy
import csv
import re
from typing import List, Dict


contractions = {
    "aint": "ain't",
    "arent": "aren't",
    "cant": "can't",
    "couldve": "could've",
    "couldnt": "couldn't",
    "couldn'tve": "couldn't've",
    "couldnt've": "couldn't've",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "dont": "don't",
    "hadnt": "hadn't",
    "hadnt've": "hadn't've",
    "hadn'tve": "hadn't've",
    "hasnt": "hasn't",
    "havent": "haven't",
    "hed": "he'd",
    "hed've": "he'd've",
    "he'dve": "he'd've",
    "hes": "he's",
    "howd": "how'd",
    "howll": "how'll",
    "hows": "how's",
    "Id've": "I'd've",
    "I'dve": "I'd've",
    "Im": "I'm",
    "Ive": "I've",
    "isnt": "isn't",
    "itd": "it'd",
    "itd've": "it'd've",
    "it'dve": "it'd've",
    "itll": "it'll",
    "let's": "let's",
    "maam": "ma'am",
    "mightnt": "mightn't",
    "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've",
    "mightve": "might've",
    "mustnt": "mustn't",
    "mustve": "must've",
    "neednt": "needn't",
    "notve": "not've",
    "oclock": "o'clock",
    "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at",
    "'ows'at": "'ow's'at",
    "'ow'sat": "'ow's'at",
    "shant": "shan't",
    "shed've": "she'd've",
    "she'dve": "she'd've",
    "she's": "she's",
    "shouldve": "should've",
    "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've",
    "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd",
    "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've",
    "somebodyll": "somebody'll",
    "somebodys": "somebody's",
    "someoned": "someone'd",
    "someoned've": "someone'd've",
    "someone'dve": "someone'd've",
    "someonell": "someone'll",
    "someones": "someone's",
    "somethingd": "something'd",
    "somethingd've": "something'd've",
    "something'dve": "something'd've",
    "somethingll": "something'll",
    "thats": "that's",
    "thered": "there'd",
    "thered've": "there'd've",
    "there'dve": "there'd've",
    "therere": "there're",
    "theres": "there's",
    "theyd": "they'd",
    "theyd've": "they'd've",
    "they'dve": "they'd've",
    "theyll": "they'll",
    "theyre": "they're",
    "theyve": "they've",
    "twas": "'twas",
    "wasnt": "wasn't",
    "wed've": "we'd've",
    "we'dve": "we'd've",
    "weve": "we've",
    "werent": "weren't",
    "whatll": "what'll",
    "whatre": "what're",
    "whats": "what's",
    "whatve": "what've",
    "whens": "when's",
    "whered": "where'd",
    "wheres": "where's",
    "whereve": "where've",
    "whod": "who'd",
    "whod've": "who'd've",
    "who'dve": "who'd've",
    "wholl": "who'll",
    "whos": "who's",
    "whove": "who've",
    "whyll": "why'll",
    "whyre": "why're",
    "whys": "why's",
    "wont": "won't",
    "wouldve": "would've",
    "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've",
    "wouldn'tve": "wouldn't've",
    "yall": "y'all",
    "yall'll": "y'all'll",
    "y'allll": "y'all'll",
    "yall'd've": "y'all'd've",
    "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've",
    "youd": "you'd",
    "youd've": "you'd've",
    "you'dve": "you'd've",
    "youll": "you'll",
    "youre": "you're",
    "youve": "you've",
}

manual_map = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
articles = ["a", "an", "the"]
period_strip = re.compile("(?!<=\d)(\.)(?!\d)")
comma_strip = re.compile("(\d)(\,)(\d)")
punct = [
    ";",
    r"/",
    "[",
    "]",
    '"',
    "{",
    "}",
    "(",
    ")",
    "=",
    "+",
    "\\",
    "_",
    "-",
    ">",
    "<",
    "@",
    "`",
    ",",
    "?",
    "!",
]


def normalize_word(token):
    """Borrowed directly from `https://github.com/dandelin/ViLT/`
    (along with related global vars)."""
    _token = token
    for p in punct:
        if (p + " " in token or " " + p in token) or (
            re.search(comma_strip, token) != None
        ):
            _token = _token.replace(p, "")
        else:
            _token = _token.replace(p, " ")
    token = period_strip.sub("", _token, re.UNICODE)

    _token = []
    temp = token.lower().split()
    for word in temp:
        word = manual_map.setdefault(word, word)
        if word not in articles:
            _token.append(word)
    for i, word in enumerate(_token):
        if word in contractions:
            _token[i] = contractions[word]
    token = " ".join(_token)
    token = token.replace(",", "")
    return token


def is_value_present(dict_list, value):
    for i, dictionary in enumerate(dict_list):
        if value in dictionary.values():
            return i, True
    return 0, False


def gen_tw_data():
    label_mapping = {
        '0': "negative",
        '1': "neutral",
        '2': "positive"
    }

    datasets = ['17']
    # datasets = ['15', '17']

    ft_data = []

    for dataset in datasets:
        # load msa data
        root_dir = '/ai/test/data/'
        tw_data = []
        with open('/ai/test/data/MSA/IJCAI2019_data/twitter20{}/train.tsv'.format(dataset)) as f:
            tsvreader = csv.reader(f, delimiter='\t')
            for line in tsvreader:
                tw_data.append(line)

        data_dir = 'MSA/IJCAI2019_data/twitter20{}_images/'.format(dataset)

        for item in tqdm(tw_data[1:]):
            index, label, image_id, query, target = copy.deepcopy(item)

            new_query = query.replace('$T$', target)
            new_query = normalize_word(new_query)

            label = label_mapping[label]

            image_path = root_dir + data_dir + image_id
            assert os.path.exists(image_path)
            # image = Image.open(image_path).convert('RGB')

            index, exist = is_value_present(ft_data, image_id)
            if not exist:
                ft_data.append({
                    "id": image_id,
                    "image": data_dir + image_id,
                    "conversations": [
                        {'from': 'human',
                         'value': f"<image>\nGiven the image and sentence {new_query}, what is the sentiment expressed?"},
                        {'from': 'gpt', 'value': "{}".format(label)},
                    ],
                })
            else:
                ft_data[index]['conversations'].append(
                    {'from': 'human',
                     'value': f"Given the image and sentence {new_query}, what is the sentiment expressed?"}
                )

                ft_data[index]['conversations'].append(
                    {'from': 'gpt', 'value': "{}".format(label)}
                )
        print(dataset + f' Number of samples: {len(ft_data)}')

    with open("tw{}_new.json".format(dataset), "w") as f:
        json.dump(ft_data, f, indent=2)
    return ft_data

def gen_mvsa_data(ft_data):

    # load msa data
    root_dir = '/ai/test/data/'

    datasets = ['single', 'multiple']

    for dataset in datasets:
        m_m_path = '/ai/test/data/MSA/msa_dataset_all/{}/filter_cap.json'.format(dataset)
        mm_data = json.load(open(m_m_path))
        if dataset == 'multiple':
            data_dir = 'MSA/MVSA/data/'
        else:
            data_dir = 'MSA/MVSA_Single/data/'

        for item in tqdm(mm_data['annotations']):
            image_id = copy.deepcopy(item['image_id'])
            label = copy.deepcopy(item['caption']).split('##answer## ')[1]
            cpation = copy.deepcopy(item['caption'])

            new_query = cpation.split('the text')[1].split(', the sentiment')[0]

            image_path = root_dir + data_dir + image_id + '.jpg'
            assert os.path.exists(image_path)
            try:
                image = Image.open(image_path).convert('RGB')
                index, exist = is_value_present(ft_data, image_id)
                if not exist:
                    ft_data.append({
                        "id": image_id,
                        "image": data_dir + image_id + '.jpg',
                        "conversations": [
                            {'from': 'human',
                             'value': f"<image>\nGiven the image and sentence {new_query}, what is the sentiment expressed?"},
                            {'from': 'gpt', 'value': "{}".format(label)},
                        ],
                    })
                else:
                    ft_data[index]['conversations'].append(
                        {'from': 'human',
                         'value': f"Given the image and sentence {new_query}, what is the sentiment expressed?"}
                    )

                    ft_data[index]['conversations'].append(
                        {'from': 'gpt', 'value': "{}".format(label)}
                    )
            except (OSError, IOError):
                print("无法读取图像:", image_id)
                continue
        # print("split " + dataset + " finished")

    print(f'Number of samples: {len(ft_data)}')
    # return ft_data
    with open("msa_data.json", "w") as f:
        json.dump(ft_data, f, indent=2)


def merge_data():
    # 定义要合并的JSON文件列表
    json_files = ['tw15.json', 'tw17_new.json', 'm_m_new.json', 'm_s.json']

    # 创建一个空字典来保存合并后的数据
    merged_data = []

    # 遍历每个JSON文件并将其内容合并到merged_data中
    for file in json_files:
        with open(file, 'r') as f:
            data = json.load(f)
            print(len(data), file)
            merged_data.extend(data)

    print(len(merged_data))
    # for item in merged_data:
    #     if not re.search(r'\.jpg$', item['image']):
    #         item['image'] = item['image'] + '.jpg'
    # 将合并后的数据写入新的JSON文件
    with open('Msa_data_116.json', 'w') as f:
        json.dump(merged_data, f)

    print('Merge finished!')


if __name__ == '__main__':
    merge_data()
    # ft_data = gen_tw_data()
    # gen_mvsa_data(ft_data)
    # msa_path = '/ai/test/code/LLaVA/scripts/merged_msa_data.json'
    # with open(msa_path, 'r') as f:
    #     msa_data = json.load(f)
    #
    # vg_path = '/ai/test/code/LLaVA/scripts/vg_ft_cfr_new.json'
    # with open(vg_path, 'r') as f:
    #     vg_path = json.load(f)
    #
    # print('kk')