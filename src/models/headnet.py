# -*- coding: utf-8 -*-  
import models.basic_layers as bl
import tensorflow as tf


def res_headnet(bev_block, img_block, cls_num, bev_bbox_dim, img_output_z, trainable):
    with tf.variable_scope('rh_block') as scope:
        bev_block = bl.resnet_block(bev_block, 180, trainable, 'head_bev_res1')
        bev_block = bl.resnet_block(bev_block, 180, trainable, 'head_bev_res2')
        bev_block = bl.resnet_block(bev_block, 180, trainable, 'head_bev_res3')
        bev_block = bl.resnet_block(bev_block, 180, trainable, 'head_bev_res4')
        bev_obj_cls = bl.convolutional(bev_block, (1, 1 + cls_num), trainable, 'bev_obj_cls')
        bev_bbox = bl.convolutional(bev_block, (3, bev_bbox_dim), trainable, 'bev_bbox')
        bev_pred = tf.concat([bev_obj_cls, bev_bbox], -1)
        img_block = bl.resnet_block(img_block, 64, trainable, 'head_img_res1')
        img_block = bl.resnet_block(img_block, 64, trainable, 'head_img_res2')
        img_block = bl.resnet_block(img_block, 64, trainable, 'head_img_res3')
        img_block = bl.resnet_block(img_block, 64, trainable, 'head_img_res4')
        img_pred = bl.convolutional(img_block, (1, img_output_z), trainable, 'img_pred')
    return bev_pred, img_pred