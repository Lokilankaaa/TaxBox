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
import pprint
import networkx as nx
from pyvis.network import Network
from queue import Queue


def extract_tree_from_imagenet(word_path, imagenet_path, save=True):
    tree = json.load(open(word_path))

    imagenet_labels = [l.lower().replace(' ', '_') for l in os.listdir(imagenet_path)]
    contained = []

    def dfs(head):
        flag = False
        if head['name'] in imagenet_labels and head['name'] not in contained:
            flag = True
            contained.append(head['name'])

        if len(head['children']) == 0:
            return flag
        else:
            lazy_del = []
            random.shuffle(head['children'])
            for c in head['children']:
                res = dfs(c)
                if not res:
                    lazy_del.append(c)
                flag = flag or res
            for c in lazy_del:
                head['children'].remove(c)
            return flag

    dfs(tree)
    print(len(contained))
    if save:
        with open(word_path.replace('wordnet', 'imagenet'), 'w') as f:
            json.dump(tree, f)

    return tree


def split_tree_dataset(whole_tree_path, split=0.8):
    tree = json.load(open(whole_tree_path))
    _id = [0]
    edge = []
    descriptions = []
    names = []

    def dfs(head):
        if len(head['children']) == 0:
            head['id'] = _id[0]
            _id[0] += 1
        else:
            head['id'] = _id[0]
            _id[0] += 1
            for i, child in enumerate(head['children']):
                dfs(child)
            list([edge.append([head['id'], c['id']]) for c in head['children']])

    def bfs(tree):
        tree['id'] = _id[0]
        _id[0] += 1
        q = Queue()
        q.put(tree)
        while not q.empty():
            head = q.get()
            names.append(head['name'])
            descriptions.append(head['definition'])

            for c in head['children']:
                c['id'] = _id[0]
                _id[0] += 1
                edge.append([head['id'], c['id']])
                q.put(c)

    def reconstruct():
        for node in test_eval:
            children = G.successors(node)
            father = list(G.predecessors(node))[0]
            G.remove_node(node)
            for c in children:
                G.add_edge(father, c)

    bfs(tree)

    G = nx.DiGraph()
    G.add_edges_from(edge)

    start_id = count_n_level_id(G, 6)

    test_eval = random.sample(range(start_id, _id[0]), k=int((_id[0] - start_id) * (1 - split)))
    train = list(range(_id[0]))
    list(map(lambda x: train.remove(x), test_eval))
    test, eva = test_eval[:len(test_eval) // 2], test_eval[len(test_eval) // 2:]

    reconstruct()

    return G, names, descriptions, train, test, eva


def count_n_level_id(G, n):
    levels = [[0]]
    for i in range(n):
        fathers = levels[i]
        levels.append([])
        for f in fathers:
            levels[i + 1] += list(G.successors(f))

    res = []
    for l in levels:
        res += l
    return max(res) + 1


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
    id_map_path = '../cls/id_map/id2name.json'
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
    with open('../datasets_json/imagenet_dataset.json', 'w') as f:
        json.dump(dataset, f)


def mk_nonvisual():
    data = {}
    labels = os.listdir('../test_nonvisual')
    for i, l in enumerate(labels):
        des = [wn.synsets(l, pos=wn.NOUN)[0].definition()]
        data[str(i)] = {'name': l, 'descriptions': des, 'train': [os.path.join('../test_nonvisual', l, i) for i in
                                                                  os.listdir(os.path.join('../test_nonvisual', l))]}
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


def sample_subset_from_wordnet(wordnet_json_root):
    wordnet = json.load(open(wordnet_json_root))


def construct_tree_to_dict(tree, unique=False):
    d = {}

    def dfs(head, unique):
        if unique:
            if head['name'] in d.keys():
                d[head['name'] + '#' + str(random.randint(0, 99))] = head
            else:
                d[head['name']] = head
        else:
            d[head['name']] = head

        if len(head['children']) == 0:
            return
        else:
            for c in head['children']:
                dfs(c, unique)

    dfs(tree, unique)
    return d


def interactive_json_maker():
    name_to_pointer = {}
    tree = {}

    def load():
        if os.path.exists('../datasets_json/middle_handcrafted.json'):
            t = json.load(open('../datasets_json/middle_handcrafted.json'))
            n = construct_tree_to_dict(t)
            return t, n
        else:
            return {}, {}

    def insert_node(father, name, description):
        if father == '':
            assert len(tree.keys()) == 0
            tree['name'] = name
            tree['description'] = description
            tree['children'] = []
            name_to_pointer[name] = tree
        else:
            assert father in name_to_pointer.keys()
            name_to_pointer[father]['children'].append({
                'name': name,
                'description': description,
                'children': []
            })
            name_to_pointer[name] = name_to_pointer[father]['children'][-1]

    def save():
        with open('../datasets_json/middle_handcrafted.json', 'w') as f:
            json.dump(tree, f)

    tree, name_to_pointer = load()

    while True:
        command = input('whats your command? ')
        if command == 'q' or command == 'quit':
            break
        elif command == 'i' or command == 'insert':
            father = input('father: ').strip().lower()
            name = input('name: ').strip().lower()
            description = input('description: ').strip().lower()
            insert_node(father, name, description)
            print('insertion finished')
        elif command == 'p' or command == 'print':
            which = input('print which node? ').strip().lower()
            if which == '':
                pprint.pprint(tree)
            else:
                assert which in name_to_pointer.keys()
                pprint.pprint(name_to_pointer[which])
        elif command == 's' or command == 'save':
            save()
            print(len(name_to_pointer.keys()))
        elif command == 'l' or command == 'load':
            tree, name_to_pointer = load()
        elif command == 'd' or command == 'delete':
            which = input('print which node? ').strip().lower()
            assert which in name_to_pointer.keys()
            del name_to_pointer[which]
        elif command == 'n' or command == 'number':
            print(len(name_to_pointer.keys()))
        elif command == 'ls' or command == 'list':
            which = input('list which node children? ').strip().lower()
            assert which in name_to_pointer.keys()
            pprint.pprint([c['name'] for c in name_to_pointer[which]['children']])
        else:
            print('invalid')
            continue

    save()


word_count = None
if __name__ == '__main__':
    # word_count = json.load(open('wordnet_count.json'))
    # word_tree = json.load(open('wordnet_dataset.json'))
    # pruned = word_tree_pruner(word_tree)
    # interactive_json_maker()
    # t = extract_tree_from_imagenet('/data/home10b/xw/visualCon/datasets_json/wordnet_dataset.json',
    #                                '/data/home10b/xw/imagenet21k/imagenet_images')
    # dt = construct_tree_to_dict(t, unique=True)
    # imagenet_labels = [l.lower().replace(' ', '_') for l in os.listdir('/data/home10b/xw/imagenet21k/imagenet_images')]

    # con = [k for k, _ in dt.items() if k in imagenet_labels]
    # print(len(con), con)
    # print(len(dt))
    split_tree_dataset('/data/home10b/xw/visualCon/datasets_json/imagenet_dataset.json')
