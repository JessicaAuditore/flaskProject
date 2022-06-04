import argparse
import os
from glob import glob
from collections import OrderedDict
import numpy as np

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
import losses
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn.functional as F

import archs
from dataset import Dataset, IsprsDataset
from metrics import iou_score, get_accuracy
from network.encnet import mobilenetxt_EncNet, xt_EncNet_nojpu
from network.sfnet_resnet import DeepR50_SF_deeply
from utils import AverageMeter
from evalution_segmentaion import eval_semantic_segmentation
from losses import BCEDiceLoss, LovaszHingeLoss, DiceLoss, FocalLossBinaryLoss, DiceLovaszHingeLoss, BCELovaszLoss
from network.sfnet_mobilenet import mobilenetv1_SF_deeply, mobilenetv2_SF_deeply, mobilenetv3_SF_deeply, \
    mobilenetxt_SF_deeply
import time

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('DiceLovaszHingeLoss')
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1" if use_cuda else "cpu")


# 验证部分（自己选择的数据集）上的分割效果
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')
    # loss
    parser.add_argument('--loss', default='BCEWithLogitsLoss',
                        # DiceLoss  LovaszHingeLoss  BCEDiceLoss BCEWithLogitsLoss BCEDiceLovaszLoss,FocalLossBinaryLoss
                        # DiceLovaszHingeLoss BCELovaszLoss
                        choices=LOSS_NAMES,
                        help='loss: ' +
                             ' | '.join(LOSS_NAMES) +
                             ' (default: BCEDiceLoss)')
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='mobilenetxt_senet',  # xt_senet_nojpu  mobilenetxt_senet
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                             ' | '.join(ARCH_NAMES) +
                             ' (default: NestedUNet)')

    # dataset
    parser.add_argument('--dataset', default='datasets',
                        help='dataset name')
    # batch_size
    parser.add_argument('-b', '--batch_size', default=8, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    # train_num
    parser.add_argument('--train_num', default='6',
                        help='train_num ')

    # JPU版本
    parser.add_argument('--jpu', default='JPU', type=str,  # 'JPU_X',NONE
                        help='jpu')
    # 是否用aspp
    parser.add_argument('--aspp', default=False, type=str,  # True
                        help='if aspp available')

    args = parser.parse_args()

    return args


def main():
    args = vars(parse_args())

    if args['name'] is None:
        args['name'] = '%s_%s_' % (args['dataset'], args['arch'])
    # 根据name打开配置文件,FPN
    # with open('models/%s/%s/%s/%s/config.yml' % (args['name'], args['loss'], args['batch_size'], args['train_num']),
    #           'r') as f:
    #     config = yaml.load(f, Loader=yaml.FullLoader)

    # 根据name打开配置文件,EncNet
    with open('models/%s/%s/%s/%s/%s/config.yml' % (
            args['name'], args['loss'], args['batch_size'], args['jpu'], args['aspp']),
              'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    # model = archs.__dict__[config['arch']](config['num_classes'],
    #                                        config['input_channels'],
    #                                        config['deep_supervision'])

    model = mobilenetxt_EncNet(config['num_classes'], config['jpu'], config['aspp'])
    print(model)
    model = model.cuda()

    # 读取数据
    # Data loading code
    img_ids = glob(os.path.join('view', config['dataset'], 'images', '*' + config['img_ext']))
    # img_ids = glob(os.path.join('test', 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

    val_img_ids = img_ids
    # _, val_img_ids = train_test_split(img_ids, test_size=0.1, random_state=41)
    print(args['name'])

    # 导入权重model.pth
    model.load_state_dict(torch.load(
        'models/%s/%s/%s/%s/%s/model.pth' % (
            args['name'], args['loss'], args['batch_size'], args['jpu'], args['aspp'])))
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
        os.path.join('outputs', args['name'], config['loss'], str(config['batch_size']), str(config['jpu']), str(config['aspp']),
                     'pre'), exist_ok=True)
    os.makedirs(
        os.path.join('outputs', args['name'], config['loss'], str(config['batch_size']), str(config['jpu']), str(config['aspp']),
                     'true'), exist_ok=True)

    # 评价指标
    avg_meters = {
        'iou': AverageMeter(),
        'acc': AverageMeter()}
    test_oa = 0
    test_pa = 0
    test_mpa = 0
    test_iou = 0
    test_miou = 0
    test_f1 = 0
    test_mf1 = 0

    # pre_label可视化，多了一个上色的过程
    colormap = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0],
                [255, 255, 0], [255, 0, 0]]

    # 将colormap由列表转换为数组
    cm = np.array(colormap).astype('uint8')
    fps = 0.0
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, meta in val_loader:
            input = input.cuda()
            target = target.cuda()

            # 记录初始时间
            t1 = time.time()
            output = model(input)
            output = F.softmax(output, dim=1)
            # 计算fps
            fps = (fps + (4 / (time.time() - t1))) / 2

            # 拿预测图计算最大值（即模型认为最准确的预测结果），固定形式记住
            pre_label = output.max(dim=1)[1].data.cpu().numpy()
            pre_label = [i for i in pre_label]

            true_label = target.max(dim=1)[1].data.cpu().numpy()
            true_label = [i for i in true_label]

            # 返回混淆矩阵
            eval_metric = eval_semantic_segmentation(pre_label, true_label)
            test_oa += eval_metric['oa']
            test_pa += eval_metric["class_accuracy"]
            test_mpa += eval_metric['mean_class_accuracy']
            test_iou += eval_metric['iou']
            test_miou += eval_metric["miou"]
            test_f1 += eval_metric['f1']
            test_mf1 += eval_metric["mf1"]

            postfix = OrderedDict([
                ('oa', eval_metric["oa"]),
                ('miou', eval_metric["miou"]),
                ('mpa', eval_metric["mean_class_accuracy"]),
                ('mf1', eval_metric["mf1"])
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)

            # 将标签放入colormap中
            for i in range(len(pre_label)):
                pre = cm[pre_label[i]]
                tur = cm[true_label[i]]
                cv2.imwrite(os.path.join('outputs', args['name'], config['loss'], str(config['batch_size']),
                                         str(config['jpu']), str(config['aspp']), 'pre',
                                         meta['img_id'][i] + '.png'),
                            pre[:, :, (2, 1, 0)].astype('uint8'))
                cv2.imwrite(os.path.join('outputs', args['name'], config['loss'], str(config['batch_size']),
                                        str(config['jpu']), str(config['aspp']), 'true',
                                         meta['img_id'][i] + '.png'),
                            tur[:, :, (2, 1, 0)].astype('uint8'))
                # print(pre)

    print('test_oa: %.4f,mpa: %.4f,mf1: %.4f' % (
        eval_metric["oa"], eval_metric["mean_class_accuracy"], eval_metric["mf1"]))

    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
