# -*- coding: utf-8 -*-  
import models.basic_layers as bl
import tensorflow as tf
from config.config import cfg
from data import loader
from models import backbone
from models import headnet
from models import loss

class ContfuseNetwork(object):
    def __init__(self):
        self.cls_num = cfg.CONTFUSE.CLASSES_NUM
        self.bev_bbox_dim = self.cls_num * cfg.BEV.BBOX_DIM
        self.img_anchor_num = len(loader.load_anchors(cfg.IMAGE.ANCHORS))
        self.img_output_z = self.img_anchor_num * (cfg.IMAGE.BBOX_DIM + cfg.CONTFUSE.CLASSES_NUM + 1)


    def net(self, bev_input, img_input, mapping1x, mapping2x, mapping4x, mapping8x, trainable):
        with tf.variable_scope('contfuse_backbone') as scope:
            bev_block, img_block = backbone.resnet_backbone(bev_input, img_input, mapping1x, mapping2x, 
                                                            mapping4x, mapping8x, trainable)
        with tf.variable_scope('contfuse_headnet') as scope:
            bev_pred, img_pred = headnet.res_headnet(bev_block, img_block, self.cls_num, self.bev_bbox_dim,
                                                    self.img_output_z, trainable)
        return bev_pred, img_pred


    def load(self):
        bev_shape = [None, cfg.BEV.INPUT_X, cfg.BEV.INPUT_Y, cfg.BEV.INPUT_Z]
        img_shape = [None, cfg.IMAGE.INPUT_H, cfg.IMAGE.INPUT_W, 3]
        bev_label_shape = [None, cfg.BEV.OUTPUT_X, cfg.BEV.OUTPUT_Y, cfg.BEV.LABEL_Z]
        img_label_shape = [None, cfg.IMAGE.OUTPUT_H, cfg.IMAGE.OUTPUT_W, cfg.IMAGE.LABEL_Z]
        mapping1x_shape = [None, cfg.BEV.INPUT_X, cfg.BEV.INPUT_Y, 2]
        mapping2x_shape = [None, int(cfg.BEV.INPUT_X / 2), int(cfg.BEV.INPUT_Y / 2), 2]
        mapping4x_shape = [None, int(cfg.BEV.INPUT_X / 4), int(cfg.BEV.INPUT_Y / 4), 2]
        mapping8x_shape = [None, int(cfg.BEV.INPUT_X / 8), int(cfg.BEV.INPUT_Y / 8), 2]
        bev_input = tf.placeholder(dtype=tf.float32, shape=bev_shape, name='bev_input_placeholder')
        img_input = tf.placeholder(dtype=tf.float32, shape=img_shape, name='img_input_placeholder')
        bev_label = tf.placeholder(dtype=tf.float32, shape=bev_label_shape, name='bev_label_placeholder')
        img_label = tf.placeholder(dtype=tf.float32, shape=img_label_shape, name='img_label_placeholder')
        mapping1x = tf.placeholder(dtype=tf.int32, shape=mapping1x_shape, name='mapping1x_placeholder')
        mapping2x = tf.placeholder(dtype=tf.int32, shape=mapping2x_shape, name='mapping2x_placeholder')
        mapping4x = tf.placeholder(dtype=tf.int32, shape=mapping4x_shape, name='mapping4x_placeholder')
        mapping8x = tf.placeholder(dtype=tf.int32, shape=mapping8x_shape, name='mapping8x_placeholder')
        bev_loss_scale = tf.placeholder(dtype=tf.float32, shape=[6], name='bev_loss_scale')
        img_loss_scale = tf.placeholder(dtype=tf.float32, shape=[6], name='img_loss_scale')
        trainable = tf.placeholder(dtype=tf.bool, name='training')
        bev_pred, img_pred = self.net(bev_input, img_input, mapping1x, mapping2x, 
                                    mapping4x, mapping8x, trainable)

        with tf.variable_scope('bev_loss') as scope:
            bev_loss = loss.bev_loss(bev_pred, bev_label, bev_loss_scale)

        with tf.variable_scope('img_loss') as scope:
            img_loss = loss.img_loss(img_pred, img_label, self.img_anchor_num, img_loss_scale)

        return {'bev_input':bev_input,
                'img_input':img_input,
                'bev_label':bev_label,
                'img_label':img_label,
                'bev_pred':bev_pred,
                'img_pred':img_pred,
                'mapping1x':mapping1x,
                'mapping2x':mapping2x,
                'mapping4x':mapping4x,
                'mapping8x':mapping8x,
                'bev_loss_scale':bev_loss_scale,
                'img_loss_scale':img_loss_scale,
                'trainable':trainable,
                'contfuse_loss': bev_loss[0] + img_loss[0],
                'bev_loss': bev_loss[0],
                'bev_obj_loss': bev_loss[1],
                'bev_cls_loss': bev_loss[2],
                'bev_bbox_loss': bev_loss[3],
                'img_loss': img_loss[0],
                'img_obj_loss': img_loss[1],
                'img_cls_loss': img_loss[2],
                'img_bbox_loss': img_loss[3]
                }



