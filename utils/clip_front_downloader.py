import time
import urllib.error
import urllib.request
from base64 import decode
import json
import argparse
import requests
import os
import tqdm
import multiprocessing as mp
from mkdataset import construct_tree_to_dict

splits = ['dev', 'test', 'train']

url = 'https://knn5.laion.ai/knn-service'
data_raw = {"text": None, "image": None, "image_url": None, "modality": "image", "num_images": 1000,
            "indice_name": "laion5B", "num_result_ids": 3000, "use_mclip": False, "deduplicate": True,
            "use_safety_model": True, "use_violence_detector": True, "aesthetic_score": "9", "aesthetic_weight": "0.5"}

num_workers = 30
datas = json.load(open("../datasets_json/middle_handcrafted.json"))
name_to_pointer = construct_tree_to_dict(datas)
name_to_pointer_list = [v for k, v in name_to_pointer.items() if len(v['children']) == 0]
i, j = len(name_to_pointer_list), len(name_to_pointer_list) // num_workers
name_to_pointer_list = [name_to_pointer_list[l:l + j] if l + j < i else name_to_pointer_list[l:] for l in
                        range(0, i, j)]
saved = {}

#
# def get_links(head):
#     if len(head['children']) == 0:
#         text = head['name'].replace('_', ' ') + ',' + head['description']
#         data_raw['text'] = text
#         while True:
#             try:
#                 response = requests.post(url, data=json.dumps(data_raw), timeout=60)
#                 if response.status_code == 200:
#                     res = response.json()
#                     saved[head['name']] = [r['url'] for r in res if 'url' in r.keys()]
#                     saved[head['name']] = saved[head['name']] if len(saved[head['name']]) <= 200 else saved[head['name']][:200]
#                     print(head['name'], len(saved[head['name']]))
#                     break
#             except Exception as e:
#                 print('err', e)
#                 time.sleep(60)
#
#     else:
#         for c in head['children']:
#             get_links(c)
#
#
# def get_list_links(l):
#     print("starting....")
#     for ll in l:
#         get_links(ll)
#
#
# get_links(datas)
# # with mp.Pool(num_workers) as p:
# #     p.map(get_list_links, name_to_pointer_list)
# #
# with open('../datasets_json/saved_middle_handcrafted_test.json', 'w', encoding='utf8') as f:
#     json.dump(saved, f, ensure_ascii=False)

headers = ("User-Agent",
           "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE")


def download_image(url, label, timeout, path, i):
    result = {
        "status": "",
        "url": url,
        "label": label,
    }
    cnt = 0
    while True:
        try:
            opener = urllib.request.build_opener()
            opener.addheaders = [headers]
            response = opener.open(url, timeout=timeout)
            label_dir = os.path.join(path, label)
            if not os.path.exists(label_dir):
                os.mkdir(label_dir)
            image_path = os.path.join(label_dir, str(i) + '.jpg')
            with open(image_path, "wb") as f:
                block_sz = 8192
                while True:
                    buffer = response.read(block_sz)
                    if not buffer:
                        break
                    f.write(buffer)
            result["status"] = "SUCCESS"
        except Exception as e:
            if isinstance(e, urllib.error.HTTPError):
                result["status"] = "EXPIRED"
                result["exception_message"] = str(e)
            else:
                result["status"] = "TIMEOUT"
                result["exception_message"] = str(e)
        break
    return result


def download_one_class(label, value):
    img_links = value
    print("start:", label)
    if label not in []:
        for i, link in enumerate(tqdm.tqdm(img_links)):
            res = download_image(link, label, 100, '/data/home10b/xw/visualCon/handcrafted', i)
            if res['status'] == 'EXPIRED':
                print(label, res['exception_message'], link)
            elif res['status'] == 'TIMEOUT':
                print(label, res['exception_message'], link)
    print("\n\ndone:", label)


datas = json.load(open('../datasets_json/saved_middle_handcrafted_test.json'))

pool = [mp.Process(target=download_one_class, args=(k, v)) for k, v in datas.items()]
list([p.start() for p in pool])
list([p.join() for p in pool])
