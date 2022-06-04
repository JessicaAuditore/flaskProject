import random, cv2, torch, os, time
import torch.nn.functional as F
from pathlib import Path
from albumentations import Resize
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
import numpy as np
from model.sfnet_resnet import DeepR50_SF_deeply
from model.network.encnet import mobilenetxt_EncNet


def get_model_DeepR50_SF_deeply(pth):
    model = DeepR50_SF_deeply(6)
    # model = model.cuda()
    model.load_state_dict(torch.load(pth, map_location=torch.device('cpu')))
    model.eval()
    return model


def get_model_mobilenetxt_EncNet(pth):
    model = mobilenetxt_EncNet(6, jpu='JPU', aspp=False)
    # model = model.cuda()
    model.load_state_dict(torch.load(pth, map_location=torch.device('cpu')))
    model.eval()
    return model


def get_prediction(input_path: str, model: str):
    img = cv2.imread(input_path)
    height, width, _ = img.shape
    if height >= 512:
        height -= 512
    else:
        height = 0
    if width >= 512:
        width -= 512
    else:
        width = 0
    hen = random.randint(0, height)
    zong = random.randint(0, width)
    img = img[hen:hen + 512, zong:zong + 512, :]

    img = transform(image=img)['image'].transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)

    output = models[model](img)
    output = F.softmax(output, dim=1)
    pre_label = output.max(dim=1)[1].data.cpu().numpy()

    pre = cm[pre_label[0]]
    output_path = input_path.replace('input', 'output').replace('.jpg', '_' + model + '.jpg')
    if Path(output_path).exists():
        os.remove(output_path)
    cv2.imwrite(output_path, pre[:, :, (2, 1, 0)].astype('uint8'))

    return output_path


models = {'deepR50': get_model_DeepR50_SF_deeply('model/pth/model_DeepR50_SF_deeply.pth'),
          'mobileNet': get_model_mobilenetxt_EncNet('model/pth/model_mobilenetxt_EncNet.pth')}
colormap = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0]]
cm = np.array(colormap).astype('uint8')
transform = Compose([
    Resize(512, 512),
    transforms.Normalize()
])
