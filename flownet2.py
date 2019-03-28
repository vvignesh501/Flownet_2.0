#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>

import os
import cv2
from helper import Flow
import argparse
import tensorflow as tf
import numpy as np
import glob
from tensorpack import *

import flownet_models as models

enable_argscope_for_module(tf.layers)


MODEL_MAP = {'flownet2-s': models.FlowNet2S,
             'flownet2-c': models.FlowNet2C,
             'flownet2': models.FlowNet2}


def apply(model_name, model_path, left, right, ground_truth=None):
  model = MODEL_MAP[model_name]
  left = cv2.imread(left).astype(np.float32).transpose(2, 0, 1)[None, ...]
  right = cv2.imread(right).astype(np.float32).transpose(2, 0, 1)[None, ...]

  predict_func = OfflinePredictor(PredictConfig(
      model=model(),
      session_init=get_model_loader(model_path),
      input_names=['left', 'right'],
      output_names=['prediction']))

  output = predict_func(left, right)[0].transpose(0, 2, 3, 1)
  flow = Flow()

  img = flow.visualize(output[0])
  if ground_truth is not None:
    img = np.concatenate([img, flow.visualize(Flow.read(ground_truth))], axis=1)

  cv2.imshow('flow output', img)
  cv2.waitKey(0)




if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu', default=0, help='comma separated list of GPU(s) to use.')
  parser.add_argument('--load', default= '/home/vigneshv/sfuhome/vigneshv/Multimedia/flownet2.npz', help='load model')
  parser.add_argument('--left', default='house_1.ppm', help='input', type=str)
  parser.add_argument('--right', default='house_2.ppm', help='input', type=str)
  parser.add_argument('--model', default="flownet2" ,help='model', type=str)
  parser.add_argument('--gt', help='ground_truth', type=str, default=None)
  args = parser.parse_args()






  if args.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


  apply(args.model, args.load, args.left, args.right, args.gt)
