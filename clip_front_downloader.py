import time
import urllib.error
import urllib.request
from base64 import decode
import json
import argparse
import requests
import os
import tqdm

# splits = ['dev', 'test', 'train']
#
# url = 'https://knn5.laion.ai/knn-service'
# data_raw = {"text": None, "image": None, "image_url": None, "modality": "image", "num_images": 50,
#             "indice_name": "laion5B", "num_result_ids": 50, "use_mclip": False, "deduplicate": True,
#             "use_safety_model": True, "use_violence_detector": True, "aesthetic_score": "9", "aesthetic_weight": "0.5"}
#
# datas = json.load(open('bamboo_dataset.json'))
# saved = {}
#
# for i, (k, v) in enumerate(tqdm.tqdm(datas.items())):
#     text = v['descriptions']
#     data_raw['text'] = text
#     while True:
#         try:
#             response = requests.post(url, data=json.dumps(data_raw))
#             if response.status_code == 200:
#                 res = response.json()
#                 v['train'] = [r['url'] for r in res]
#                 break
#         except Exception as e:
#             print(e)
#             time.sleep(10)
#     saved[k] = v
#     if i > 5000:
#         break
#
# with open('bamboo_back.json', 'w', encoding='utf8') as f:
#     json.dump(saved, f, ensure_ascii=False)

headers = ("User-Agent",
           "Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.132 Safari/537.36 QIHU 360SE")


def download_image(url, label, timeout, path):
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
            image_path = os.path.join(label_dir, url.split("/")[-1])
            with open(image_path, "wb") as f:
                block_sz = 8192
                while True:
                    buffer = response.read(block_sz)
                    if not buffer:
                        break
                    f.write(buffer)
            result["status"] = "SUCCESS"
        except Exception as e:
            if not isinstance(e, urllib.error.HTTPError):
                cnt += 1
                if cnt <= 10:
                    continue
            if isinstance(e, urllib.error.HTTPError):
                result["status"] = "EXPIRED"
                result["exception_message"] = str(e)
            else:
                result["status"] = "TIMEOUT"
        break
    return result


datas = json.load(open('bamboo_back.json'))
for i, (k, v) in enumerate(tqdm.tqdm(datas.items())):
    img_links = v['train'] if len(v['train']) <= 50 else v['train'][:50]
    for link in img_links:
        res = download_image(link, k, 10, '/data/home10b/xw/visualCon/clip_imgs')
        if res['status'] == 'EXPIRED':
            print(k, v['name'][0], res['exception_message'], link)
        elif res['status'] == 'TIMEOUT':
            print(k, v['name'][0], 'timeout', link)
