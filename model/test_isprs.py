import argparse
import os
from glob import glob
import numpy as np

import cv2
import torch
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
import torch.nn.functional as F

from dataset import IsprsDataset
from sfnet_resnet import DeepR50_SF_deeply

# 验证部分（自己选择的数据集）上的分割效果
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')

    # dataset
    parser.add_argument('--dataset', default='datasets',
                        help='dataset name')
    # batch_size
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # train_num
    parser.add_argument('--train_num', default='6',
                        help='train_num ')

    args = parser.parse_args()
    return args


def main():
    args = vars(parse_args())
    # 根据name打开配置文件
    with open('models/%s/%s/%s/%s/config.yml' % (args['name'], args['loss'], args['batch_size'], args['train_num']),
              'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)


    model = DeepR50_SF_deeply(config['num_classes'], )
    model = model.cuda()

    # 读取数据
    img_ids = glob(os.path.join('view', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    val_img_ids = img_ids

    model.load_state_dict(torch.load('model.pth'))
    model.eval()

    val_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = IsprsDataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('view', config['dataset'], 'images'),
        mask_dir=os.path.join('view', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    # 创建放入图片数据的目录
    os.makedirs(
        os.path.join('outputs', args['name'], config['loss'], str(config['batch_size']), str(config['train_num']),
                     'pre'), exist_ok=True)

    # pre_label可视化，多了一个上色的过程
    colormap = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
                [255, 255, 0], [255, 0, 0]]

    cm = np.array(colormap).astype('uint8')

    with torch.no_grad():
        for input, target, meta in val_loader:
            input = input.cuda()

            output = model(input)
            output = F.softmax(output, dim=1)

            pre_label = output.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            for i in range(len(pre_label)):
                pre = cm[pre_label[i]]
                cv2.imwrite(os.path.join('outputs', args['name'], config['loss'], str(config['batch_size']),
                                         str(config['train_num']), 'pre',
                                         meta['img_id'][i] + '.png'),
                            pre[:, :, (2, 1, 0)].astype('uint8'))


if __name__ == '__main__':
    main()
