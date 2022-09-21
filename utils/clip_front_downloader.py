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

splits = ['dev', 'test', 'train']

url = 'https://knn5.laion.ai/knn-service'
data_raw = {"text": None, "image": None, "image_url": None, "modality": "image", "num_images": 10000,
            "indice_name": "laion5B", "num_result_ids": 10000, "use_mclip": False, "deduplicate": True,
            "use_safety_model": True, "use_violence_detector": True, "aesthetic_score": "9", "aesthetic_weight": "0.5"}

# datas = json.load(open("datasets_json/handcrafted.json"))
# datas = {
#     'pineapple': {
#         'description': 'large sweet fleshy tropical fruit with a terminal tuft of stiff leaves; widely cultivated'
#     },
#     'juice': {
#         'description': 'the liquid part that can be extracted from plant or animal tissue by squeezing or cooking'
#     },
#     'hamburger': {
#         'description': 'a sandwich consisting of a fried cake of minced beef served on a bun, often with other ingredients'
#     }
# }
# saved = {}
#
#
# def get_links(datas):
#     for k, v in datas.items():
#         if "children" not in v.keys():
#             text = k + ',' + v['description']
#             data_raw['text'] = text
#             while True:
#                 try:
#                     response = requests.post(url, data=json.dumps(data_raw))
#                     if response.status_code == 200:
#                         res = response.json()
#                         v['train'] = [r['url'] for r in res]
#                         v['train'] = v['train'] if len(v['train']) <= 500 else v['train'][:500]
#                         print(k, len(v['train']))
#                         break
#                 except Exception as e:
#                     print('err', e)
#                     time.sleep(60)
#             saved[k] = v
#
#         else:
#             for ck, cv in v['children'].items():
#                 get_links({ck: cv})
#
#
# get_links(datas)
# #
# with open('../datasets_json/saved_handcrafted_test.json', 'w', encoding='utf8') as f:
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
    img_links = value['train']
    print("start:", label)
    if label not in []:
        for i, link in enumerate(tqdm.tqdm(img_links)):
            res = download_image(link, label, 100, '/data/home10b/xw/visualCon/handcrafted', i)
            if res['status'] == 'EXPIRED':
                print(label, res['ex ception_message'], link)
            elif res['status'] == 'TIMEOUT':
                print(label, res['exception_message'], link)
    print("\n\ndone:", label)


datas = json.load(open('../datasets_json/saved_handcrafted_test.json'))

pool = [mp.Process(target=download_one_class, args=(k, v)) for k, v in datas.items()]
list([p.start() for p in pool])
list([p.join() for p in pool])
