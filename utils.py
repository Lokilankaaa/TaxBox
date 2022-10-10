import torch

model_dict = {'resnet50': 'nvidia_resnet50'}


def extract_feature(model_name: str, imgs, device):
    if model_name in model_dict.keys():
        model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', model_dict[model_name], pretrained=True)
        model_utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_convnets_processing_utils')
        model.eval().to(device)

        with torch.no_grad():
            pass
    else:
        return None


if __name__ == '__main__':
    extract_feature('resnet50', None, torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
