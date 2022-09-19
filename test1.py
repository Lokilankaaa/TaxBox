import torch
import numpy as np
import clip
import os

import wikipedia
from datasets_torch.imagenet_des import ImageNet_Des
import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
print('{} is available'.format(device))
model, preprocess = clip.load('ViT-B/32', device)

# Download the dataset
# cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
img_des = ImageNet_Des('bam_back.json', 'train', preprocess)
# img_des = ImageNet_Des('nonvisual.json', 'train', preprocess)

# Prepare the inputs
res = ''
old_id = ''
old_des = ''
statis = []
bar = tqdm.tqdm(total=len(img_des))
with open('res.txt', 'w') as f:
    for i, (image_input, des, name, _id) in enumerate(img_des):
        if des == '':
            bar.update(1)
            continue
        if _id != old_id:
            if statis:
                statis = np.array(statis)
                mean = statis.mean()
                var = statis.var()
                f.write('{}, {}, mean: {}, var: {}\n'.format(_id, name, mean, var))
            statis = []
            old_id = _id
            if des == '':
                try:
                    des = wikipedia.summary(name, sentences=1)
                except:
                    des = name
                finally:
                    old_des = des
        else:
            des = old_des
        des = 'The definition of {} is '.format(name) + des
        if type(des) != list:
            des = [des]
        text_inputs = torch.cat([clip.tokenize(d, truncate=True) for d in des]).to(device)

        # Calculate features
        with torch.no_grad():
            image_features = model.encode_image(image_input.unsqueeze(0).to(device))
            text_features = model.encode_text(text_inputs)
        # Pick the top 5 most similar labels for the image
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features @ text_features.T)
        values, indices = similarity[0].topk(1)

        statis.append(values[0].item())
        bar.update(1)

