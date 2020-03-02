# -*- coding: utf-8 -*-  
import models.basic_layers as bl
import tensorflow as tf


def resnet_backbone(bev_input, img_input, mapping1x, mapping2x, mapping4x, mapping8x, trainable):
    with tf.variable_scope('rb_block_0') as scope:
        fused_block_0 = tf.gather_nd(img_input, mapping1x, name='fusion_gather_0', batch_dims=1)
        fused_block_0 = tf.stop_gradient(fused_block_0, name='fused_block_0')
        fused_block_0 = tf.concat([bev_input, fused_block_0], -1)

    with tf.variable_scope('rb_block_1') as scope:
        bev_block = bl.convolutional(bev_input, (3, 32), trainable, 'bev_conv1')
        bev_block = bl.convolutional(bev_input, (1, 16), trainable, 'bev_conv2')
        bev_block_1 = bl.convolutional(bev_block, (3, 32), trainable, 'bev_conv3', downsample=True)
        img_block = bl.convolutional(img_input, (3, 32), trainable, 'img_conv1')
        img_block = bl.convolutional(img_block, (1, 16), trainable, 'img_conv2')
        img_block = bl.convolutional(img_block, (3, 32), trainable, 'img_conv3', downsample=True) 
        fused_block_1 = tf.gather_nd(img_block, mapping2x, name='fusion_gather_1', batch_dims=1)
        fused_block_1 = tf.stop_gradient(fused_block_1, name='fused_block_1')
        fused_block_1 = tf.concat([bev_block_1, fused_block_1], -1)

    with tf.variable_scope('rb_block_2') as scope:
        bev_block = bl.resnet_block(fused_block_1, 32, trainable, 'bev_res1')
        bev_block = bl.resnet_block(bev_block, 32, trainable, 'bev_res2')
        bev_block_2 = bl.convolutional(bev_block, (3, 96), trainable, 'bev_conv1', downsample=True)
        img_block = bl.resnet_block(img_block, 18, trainable, 'img_res1')
        img_block = bl.resnet_block(img_block, 18, trainable, 'img_res2')
        img_block = bl.convolutional(img_block, (3, 64), trainable, 'img_conv1', downsample=True)
        fused_block_2 = tf.gather_nd(img_block, mapping4x, name='fusion_gather_2', batch_dims=1)
        fused_block_2 = tf.stop_gradient(fused_block_2, name='fused_block_2')
        fused_block_2 = tf.concat([bev_block_2, fused_block_2], -1)

    with tf.variable_scope('rb_block_3') as scope:
        bev_block =  bl.resnet_block(fused_block_2, 80, trainable, 'bev_res1')
        bev_block =  bl.resnet_block(bev_block, 80, trainable, 'bev_res2')
        bev_block =  bl.resnet_block(bev_block, 80, trainable, 'bev_res3')
        bev_block_3 = bl.convolutional(bev_block, (3, 240), trainable, 'bev_conv1', downsample=True)
        img_block = bl.resnet_block(img_block, 32, trainable, 'img_res1')
        img_block = bl.resnet_block(img_block, 32, trainable, 'img_res2')
        img_block = bl.resnet_block(img_block, 32, trainable, 'img_res3')
        img_last_block = bl.convolutional(img_block, (3, 128), trainable, 'img_conv1', downsample=True)
        fused_block_3 = tf.gather_nd(img_block, mapping8x, name='fusion_gather_3', batch_dims=1)
        fused_block_3 = tf.stop_gradient(fused_block_3, name='fused_block_3')
        fused_block_3 = tf.concat([bev_block_3, fused_block_3], -1)

    with tf.variable_scope('rb_upsample_block') as scope:
        up_block_4d = bl.upsample(fused_block_3, "deconv2d_1")
        up_block_4d = bl.convolutional(up_block_4d, (3, 256), trainable, 'bev_conv1')
        up_block_4d =  bl.resnet_block(up_block_4d, 128, trainable, 'bev_res1')
        up_block_4d = tf.concat([up_block_4d, bev_block_2], -1) 
    return up_block_4d, img_last_block