import os
import torch
import argparse
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.c2_model_loading import load_c2_format
from pdb import set_trace as bb


def removekey(d, listofkeys):
    r = dict(d)
    for key in listofkeys:
        print('key: {} is removed'.format(key))
        r.pop(key)
    return r


# https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md
parser = argparse.ArgumentParser(description="Trim Detection weights and save in PyTorch format.")
parser.add_argument(
    "--pretrained_path",
    default="../e2e_mask_rcnn_X_101_32x8d_FPN_1x.pkl",
    help="path to detectron pretrained weight(.pkl)",
    type=str,
)
parser.add_argument(
    "--save_path",
    default="../trimmed_e2e_mask_rcnn_X_101_32x8d_FPN_1x.pth",
    help="path to save the converted model",
    type=str,
)
parser.add_argument(
    "--cfg",
    default="configs/e2e_mask_rcnn_X_101_32x8d_FPN_1x.yaml",
    help="path to config file",
    type=str,
)

args = parser.parse_args()
#
DETECTRON_PATH = os.path.expanduser(args.pretrained_path)
print('detectron path: {}'.format(DETECTRON_PATH))

cfg.merge_from_file(args.cfg)
_d = load_c2_format(cfg, DETECTRON_PATH)
newdict = _d

newdict['model'] = removekey(_d['model'],
                             ['rpn.head.bbox_pred.bias', 'rpn.head.bbox_pred.weight', 'rpn.head.cls_logits.bias', 'rpn.head.cls_logits.weight'])
#                              ['cls_score.bias', 'cls_score.weight', 'bbox_pred.bias', 'bbox_pred.weight'])
torch.save(newdict, args.save_path)
print('saved to {}.'.format(args.save_path))
