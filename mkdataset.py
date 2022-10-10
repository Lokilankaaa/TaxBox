import time
import os
import random
import json
import openai
import tqdm
import requests
import wikipedia
from nltk.corpus import wordnet as wn
import queue


def mk_label_map():
    label_path = 'labels'
    label_map = {}
    with open(label_path, 'r') as f:
        res = f.readlines()
        for line in res:
            _id, label = line.split(':')
            label_map[_id] = [l.strip() for l in label.split(',')]

    return label_map


def mk_bamboo_map():
    search_api = 'https://opengvlab.shlab.org.cn/api/search'
    form_data = {'keyword': None}
    bamboo_path = 'cls/train/images'
    id_map_path = 'cls/id_map/id2name.json'
    bamboo_dataset = {}
    with open(id_map_path, 'r') as f:
        id_map = json.load(f)
    for l in tqdm.tqdm(os.listdir(bamboo_path)):
        if l in id_map:
            form_data['keyword'] = id_map[l][0]
            while True:
                try:
                    rep = requests.post(search_api, params=form_data)
                    if rep.status_code == 200:
                        if rep.json()['result']['matching']:
                            des = rep.json()['result']['matching'][0]['desc']
                            if des == '':
                                try:
                                    des = wikipedia.summary(id_map[l][0], sentences=1)
                                except:
                                    des = id_map[l][0]
                            bamboo_dataset[l] = {'name': id_map[l], 'descriptions': des,
                                                 'train': [os.path.join(bamboo_path, l, i) for i in
                                                           os.listdir(os.path.join(bamboo_path, l))]}
                    break
                except Exception as e:
                    print(e)
                    time.sleep(10)
    with open('bamboo_dataset.json', 'w') as f:
        json.dump(bamboo_dataset, f)


def build_dataset(label_map, sample_image=100):
    image_path = os.path.abspath('../imagenet')
    dataset = {}
    num_images = sample_image
    for k, v in tqdm.tqdm(label_map.items()):
        img_dir = os.path.join(image_path, k)
        sampled_imgs = random.choices(os.listdir(img_dir), k=3 * num_images)
        descriptions = [wn.synsets(vv.replace(' ', '_'), pos=wn.NOUN)[0].definition() for vv in v]
        dataset[k] = {'name': v, 'descriptions': descriptions,
                      'train': [os.path.join(img_dir, p) for p in sampled_imgs[:num_images]],
                      'val': [os.path.join(img_dir, p) for p in sampled_imgs[num_images:2 * num_images]],
                      'test': [os.path.join(img_dir, p) for p in sampled_imgs[2 * num_images:]]}
    with open('imagenet_dataset.json', 'w') as f:
        json.dump(dataset, f)


def mk_nonvisual():
    data = {}
    labels = os.listdir('test_nonvisual')
    for i, l in enumerate(labels):
        des = [wn.synsets(l, pos=wn.NOUN)[0].definition()]
        data[str(i)] = {'name': l, 'descriptions': des, 'train': [os.path.join('test_nonvisual', l, i) for i in
                                                                  os.listdir(os.path.join('test_nonvisual', l))]}
    with open('nonvisual.json', 'w') as f:
        json.dump(data, f)


def statis():
    with open('res.txt', 'r') as f:
        res = f.readlines()

    mean, var = 0, 0
    for line in res:
        line = line.split(',')
        mean += float(line[1].split(':')[1])
        var += float(line[2].split(':')[1])
    print(mean / 100)
    print(var / 100)


def wordnet_dataset():
    word_queue = queue.Queue()
    pos = wn.NOUN
    word_tree = {}
    word_count = {}
    word_queue.put((word_tree, wn.synsets('entity', pos=pos)[0]))
    bar = tqdm.tqdm()

    while not word_queue.empty():
        cur_node, cur_query = word_queue.get()
        cur_node['name'] = cur_query.name().split('.')[0]
        cur_node['definition'] = cur_query.definition()
        cur_node['children'] = []
        word_count[cur_node['name']] = 1 if cur_node['name'] not in word_count else word_count[cur_node['name']] + 1

        for c in cur_query.hyponyms():
            cur_node['children'].append({})
            word_queue.put((cur_node['children'][-1], c))
        bar.update(1)

    with open('wordnet_dataset.json', 'w') as f:
        json.dump(word_tree, f)
    with open('wordnet_count.json', 'w') as f:
        json.dump(word_count, f)


prompt = "A concept is considered a visual concept if it meets both the following two conditions. " \
         "First, instances of the concept have a physical entity and can be seen in vision and share some common " \
         "visual features." \
         " Second, it is not a place nor a city nor a village nor a brand nor a company nor an abstract concept.\n" \
         "Is {}, {} a visual concept? Ju111t answer ye"


def gpt3_judge(concept, description):
    if word_count['concept'] == 1:
        description = None
    openai.api_key = os.getenv("OPENAI_API_KEY")
    response = openai.Completion.create(model="text-davinci-002", prompt=prompt.format(concept, description),
                                        temperature=0,
                                        max_tokens=6)
    answer = response['choices'][0]['text'].strip()
    return True if answer == 'Yes' else False


def word_tree_pruner(head):
    if len(head['children']) != 0:
        return gpt3_judge(head['name'], head['definition'])
    else:
        head_is_visual = False
        for child in head['children']:
            is_visual = word_tree_pruner(child)
            if not is_visual:
                head['children'].remove(child)
            head_is_visual = head_is_visual or is_visual
        if not head_is_visual:
            head_is_visual = head_is_visual or gpt3_judge(head['name'], head['definition'])
        return head_is_visual


word_count = None
if __name__ == '__main__':
    word_count = json.load(open('wordnet_count.json'))
    word_tree = json.load(open('wordnet_dataset.json'))
    pruned = word_tree_pruner(word_tree)
