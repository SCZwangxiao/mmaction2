import argparse

import mmcv
import numpy as np
import torch
from mmcv.runner import load_checkpoint

from mmaction.models import build_model


def pytorch2pt(model,
               input_shape,
               output_file='tmp.pt',
               verify=False):
    """Convert pytorch model to onnx model.

    Args:
        model (:obj:`nn.Module`): The pytorch model to be exported.
        input_shape (tuple[int]): The input tensor shape of the model.
        output_file (str): Output onnx model name. Default: 'tmp.onnx'.
        verify (bool): Determines whether to verify the onnx model.
            Default: False.
    """
    model.cpu().eval()

    input_tensor = torch.randn(input_shape)
    
    trace_jit = torch.jit.trace(model, input_tensor)
    torch.jit.save(trace_jit, output_file) 

    print(f'Successfully exported pt model: {output_file}')
    if verify:
        raise NotImplementedError


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert MMAction2 models to TorchScript')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--output-file', type=str, default='tmp.onnx')
    parser.add_argument(
        '--verify',
        action='store_true',
        help='verify the onnx model output against pytorch output')
    parser.add_argument(
        '--is-localizer',
        action='store_true',
        help='whether it is a localizer')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[1, 8, 3, 224, 224],
        help='input video size')
    parser.add_argument(
        '--softmax',
        action='store_true',
        help='wheter to add softmax layer at the end of recognizers')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    # import modules from string list.

    if not args.is_localizer:
        cfg.model.backbone.pretrained = None

    # build the model
    model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))

    # pt.trace does not support kwargs
    if hasattr(model, 'forward_dummy'):
        from functools import partial
        model.forward = partial(model.forward_dummy, softmax=args.softmax)
    elif hasattr(model, '_forward') and args.is_localizer:
        model.forward = model._forward
    else:
        raise NotImplementedError(
            'Please implement the forward method for exporting.')

    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # conver model to onnx file
    pytorch2pt(
        model,
        args.shape,
        output_file=args.output_file,
        verify=args.verify)
