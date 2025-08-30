import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

class Block35(layers.Layer):
    """Inception-ResNet-A block for 35x35 grid."""
    def __init__(self, scale=1.0, activation_fn=tf.nn.relu, name=None):
        super(Block35, self).__init__(name=name)
        self.scale = scale
        self.activation_fn = activation_fn

        # Branch 0 - 1x1 conv
        self.branch0_conv = layers.Conv2D(32, 1, padding='same', name='Branch_0/Conv2d_1x1')
        self.branch0_bn = layers.BatchNormalization(name='Branch_0/Conv2d_1x1/BatchNorm')

        # Branch 1 - 1x1 conv -> 3x3 conv
        self.branch1_conv0 = layers.Conv2D(32, 1, padding='same', name='Branch_1/Conv2d_0a_1x1')
        self.branch1_conv0_bn = layers.BatchNormalization(name='Branch_1/Conv2d_0a_1x1/BatchNorm')
        self.branch1_conv1 = layers.Conv2D(32, 3, padding='same', name='Branch_1/Conv2d_0b_3x3')
        self.branch1_conv1_bn = layers.BatchNormalization(name='Branch_1/Conv2d_0b_3x3/BatchNorm')

        # Branch 2 - 1x1 conv -> 3x3 conv -> 3x3 conv
        self.branch2_conv0 = layers.Conv2D(32, 1, padding='same', name='Branch_2/Conv2d_0a_1x1')
        self.branch2_conv0_bn = layers.BatchNormalization(name='Branch_2/Conv2d_0a_1x1/BatchNorm')
        self.branch2_conv1 = layers.Conv2D(48, 3, padding='same', name='Branch_2/Conv2d_0b_3x3')
        self.branch2_conv1_bn = layers.BatchNormalization(name='Branch_2/Conv2d_0b_3x3/BatchNorm')
        self.branch2_conv2 = layers.Conv2D(64, 3, padding='same', name='Branch_2/Conv2d_0c_3x3')
        self.branch2_conv2_bn = layers.BatchNormalization(name='Branch_2/Conv2d_0c_3x3/BatchNorm')

        # Final 1x1 conv for residual connection
        self.conv_end = layers.Conv2D(192, 1, padding='same', use_bias=False, name='Conv2d_1x1')

    def call(self, inputs, training=False):
        # Branch 0
        branch0 = self.branch0_conv(inputs)
        branch0 = self.branch0_bn(branch0, training=training)
        branch0 = tf.nn.relu(branch0)

        # Branch 1
        branch1 = self.branch1_conv0(inputs)
        branch1 = self.branch1_conv0_bn(branch1, training=training)
        branch1 = tf.nn.relu(branch1)
        branch1 = self.branch1_conv1(branch1)
        branch1 = self.branch1_conv1_bn(branch1, training=training)
        branch1 = tf.nn.relu(branch1)

        # Branch 2
        branch2 = self.branch2_conv0(inputs)
        branch2 = self.branch2_conv0_bn(branch2, training=training)
        branch2 = tf.nn.relu(branch2)
        branch2 = self.branch2_conv1(branch2)
        branch2 = self.branch2_conv1_bn(branch2, training=training)
        branch2 = tf.nn.relu(branch2)
        branch2 = self.branch2_conv2(branch2)
        branch2 = self.branch2_conv2_bn(branch2, training=training)
        branch2 = tf.nn.relu(branch2)

        # Concatenate branches and apply residual connection
        mixed = tf.concat([branch0, branch1, branch2], axis=3)
        up = self.conv_end(mixed)
        net = inputs + self.scale * up
        
        if self.activation_fn:
            net = self.activation_fn(net)
        return net

class Block17(layers.Layer):
    """Inception-ResNet-B block for 17x17 grid."""
    def __init__(self, scale=1.0, activation_fn=tf.nn.relu, name=None):
        super(Block17, self).__init__(name=name)
        self.scale = scale
        self.activation_fn = activation_fn

        # Branch 0 - 1x1 conv
        self.branch0_conv = layers.Conv2D(192, 1, padding='same', name='Branch_0/Conv2d_1x1')
        self.branch0_bn = layers.BatchNormalization(name='Branch_0/Conv2d_1x1/BatchNorm')

        # Branch 1 - 1x1 -> 1x7 -> 7x1 conv
        self.branch1_conv0 = layers.Conv2D(128, 1, padding='same', name='Branch_1/Conv2d_0a_1x1')
        self.branch1_conv0_bn = layers.BatchNormalization(name='Branch_1/Conv2d_0a_1x1/BatchNorm')
        self.branch1_conv1 = layers.Conv2D(160, [1, 7], padding='same', name='Branch_1/Conv2d_0b_1x7')
        self.branch1_conv1_bn = layers.BatchNormalization(name='Branch_1/Conv2d_0b_1x7/BatchNorm')
        self.branch1_conv2 = layers.Conv2D(192, [7, 1], padding='same', name='Branch_1/Conv2d_0c_7x1')
        self.branch1_conv2_bn = layers.BatchNormalization(name='Branch_1/Conv2d_0c_7x1/BatchNorm')

        # Final 1x1 conv for residual connection
        self.conv_end = layers.Conv2D(960, 1, padding='same', use_bias=False, name='Conv2d_1x1')

    def call(self, inputs, training=False):
        # Branch 0
        branch0 = self.branch0_conv(inputs)
        branch0 = self.branch0_bn(branch0, training=training)
        branch0 = tf.nn.relu(branch0)

        # Branch 1
        branch1 = self.branch1_conv0(inputs)
        branch1 = self.branch1_conv0_bn(branch1, training=training)
        branch1 = tf.nn.relu(branch1)
        branch1 = self.branch1_conv1(branch1)
        branch1 = self.branch1_conv1_bn(branch1, training=training)
        branch1 = tf.nn.relu(branch1)
        branch1 = self.branch1_conv2(branch1)
        branch1 = self.branch1_conv2_bn(branch1, training=training)
        branch1 = tf.nn.relu(branch1)

        # Concatenate branches and apply residual connection
        mixed = tf.concat([branch0, branch1], axis=3)
        up = self.conv_end(mixed)
        net = inputs + self.scale * up
        
        if self.activation_fn:
            net = self.activation_fn(net)
        return net

class Block8(layers.Layer):
    """Inception-ResNet-C block for 8x8 grid."""
    def __init__(self, scale=1.0, activation_fn=tf.nn.relu, name=None):
        super(Block8, self).__init__(name=name)
        self.scale = scale
        self.activation_fn = activation_fn

        # Branch 0 - 1x1 conv
        self.branch0_conv = layers.Conv2D(192, 1, padding='same', name='Branch_0/Conv2d_1x1')
        self.branch0_bn = layers.BatchNormalization(name='Branch_0/Conv2d_1x1/BatchNorm')

        # Branch 1 - 1x1 -> 1x3 -> 3x1 conv
        self.branch1_conv0 = layers.Conv2D(192, 1, padding='same', name='Branch_1/Conv2d_0a_1x1')
        self.branch1_conv0_bn = layers.BatchNormalization(name='Branch_1/Conv2d_0a_1x1/BatchNorm')
        self.branch1_conv1 = layers.Conv2D(224, [1, 3], padding='same', name='Branch_1/Conv2d_0b_1x3')
        self.branch1_conv1_bn = layers.BatchNormalization(name='Branch_1/Conv2d_0b_1x3/BatchNorm')
        self.branch1_conv2 = layers.Conv2D(256, [3, 1], padding='same', name='Branch_1/Conv2d_0c_3x1')
        self.branch1_conv2_bn = layers.BatchNormalization(name='Branch_1/Conv2d_0c_3x1/BatchNorm')

        # Final 1x1 conv for residual connection
        self.conv_end = layers.Conv2D(1952, 1, padding='same', use_bias=False, name='Conv2d_1x1')

    def call(self, inputs, training=False):
        # Branch 0
        branch0 = self.branch0_conv(inputs)
        branch0 = self.branch0_bn(branch0, training=training)
        branch0 = tf.nn.relu(branch0)

        # Branch 1
        branch1 = self.branch1_conv0(inputs)
        branch1 = self.branch1_conv0_bn(branch1, training=training)
        branch1 = tf.nn.relu(branch1)
        branch1 = self.branch1_conv1(branch1)
        branch1 = self.branch1_conv1_bn(branch1, training=training)
        branch1 = tf.nn.relu(branch1)
        branch1 = self.branch1_conv2(branch1)
        branch1 = self.branch1_conv2_bn(branch1, training=training)
        branch1 = tf.nn.relu(branch1)

        # Concatenate branches and apply residual connection
        mixed = tf.concat([branch0, branch1], axis=3)
        up = self.conv_end(mixed)
        net = inputs + self.scale * up
        
        if self.activation_fn:
            net = self.activation_fn(net)
        return net

class InceptionResNetV2(Model):
    """Inception-ResNet v2 model for the R&P defense."""
    def __init__(self, num_classes=1001, dropout_keep_prob=0.8):
        super(InceptionResNetV2, self).__init__(name='InceptionResNetV2')
        
        # Enable mixed precision
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        self._set_dtype_policy(policy)

        # Stem network
        self.conv2d_1a = layers.Conv2D(32, 3, strides=2, padding='valid', name='Conv2d_1a_3x3')
        self.conv2d_1a_bn = layers.BatchNormalization(name='Conv2d_1a_3x3/BatchNorm')
        
        self.conv2d_2a = layers.Conv2D(32, 3, padding='valid', name='Conv2d_2a_3x3')
        self.conv2d_2a_bn = layers.BatchNormalization(name='Conv2d_2a_3x3/BatchNorm')
        
        self.conv2d_2b = layers.Conv2D(64, 3, padding='same', name='Conv2d_2b_3x3')
        self.conv2d_2b_bn = layers.BatchNormalization(name='Conv2d_2b_3x3/BatchNorm')
        
        self.maxpool_3a = layers.MaxPooling2D(3, strides=2, padding='valid', name='MaxPool_3a_3x3')
        
        self.conv2d_3b = layers.Conv2D(80, 1, padding='valid', name='Conv2d_3b_1x1')
        self.conv2d_3b_bn = layers.BatchNormalization(name='Conv2d_3b_1x1/BatchNorm')
        
        self.conv2d_4a = layers.Conv2D(192, 3, padding='valid', name='Conv2d_4a_3x3')
        self.conv2d_4a_bn = layers.BatchNormalization(name='Conv2d_4a_3x3/BatchNorm')
        
        self.maxpool_5a = layers.MaxPooling2D(3, strides=2, padding='valid', name='MaxPool_5a_3x3')

        # Mixed blocks
        self.mixed_5b_branch0 = layers.Conv2D(96, 1, name='Mixed_5b/Branch_0/Conv2d_1x1')
        self.mixed_5b_branch0_bn = layers.BatchNormalization(name='Mixed_5b/Branch_0/Conv2d_1x1/BatchNorm')

        # Inception-ResNet blocks
        self.blocks35 = [Block35(scale=0.17, name=f'Block35_{i}') for i in range(10)]
        self.blocks17 = [Block17(scale=0.10, name=f'Block17_{i}') for i in range(20)]
        self.blocks8 = [Block8(scale=0.20, name=f'Block8_{i}') for i in range(9)]
        self.block8_final = Block8(scale=1.0, activation_fn=None, name='Block8_final')

        # Reduction blocks
        # Reduction block B layers
        self.reduction_b = {
            'branch0_conv': layers.Conv2D(384, 3, strides=2, padding='same',
                                        name='Mixed_6a/Branch_0/Conv2d_1a_3x3'),
            'branch0_bn': layers.BatchNormalization(
                name='Mixed_6a/Branch_0/Conv2d_1a_3x3/BatchNorm'),
            'branch1_conv1': layers.Conv2D(256, 1, padding='same',
                                            name='Mixed_6a/Branch_1/Conv2d_0a_1x1'),
            'branch1_bn1': layers.BatchNormalization(
                name='Mixed_6a/Branch_1/Conv2d_0a_1x1/BatchNorm'),
            'branch1_conv2': layers.Conv2D(256, 3, padding='same',
                                            name='Mixed_6a/Branch_1/Conv2d_0b_3x3'),
            'branch1_bn2': layers.BatchNormalization(
                name='Mixed_6a/Branch_1/Conv2d_0b_3x3/BatchNorm'),
            'branch1_conv3': layers.Conv2D(384, 3, strides=2, padding='same',
                                            name='Mixed_6a/Branch_1/Conv2d_1a_3x3'),
            'branch1_bn3': layers.BatchNormalization(
                name='Mixed_6a/Branch_1/Conv2d_1a_3x3/BatchNorm'),
            'branch2_pool': layers.MaxPooling2D(3, strides=2, padding='same',
                                                name='Mixed_6a/Branch_2/MaxPool_1a_3x3')
            }

        # Reduction block C layers
        self.reduction_c = {
            'branch0_conv1': layers.Conv2D(256, 1, padding='same',
                                            name='Mixed_7a/Branch_0/Conv2d_0a_1x1'),
            'branch0_bn1': layers.BatchNormalization(
                name='Mixed_7a/Branch_0/Conv2d_0a_1x1/BatchNorm'),
            'branch0_conv2': layers.Conv2D(384, 3, strides=2, padding='same',
                                            name='Mixed_7a/Branch_0/Conv2d_1a_3x3'),
            'branch0_bn2': layers.BatchNormalization(
                name='Mixed_7a/Branch_0/Conv2d_1a_3x3/BatchNorm'),
            
            'branch1_conv1': layers.Conv2D(256, 1, padding='same',
                                            name='Mixed_7a/Branch_1/Conv2d_0a_1x1'),
            'branch1_bn1': layers.BatchNormalization(
                name='Mixed_7a/Branch_1/Conv2d_0a_1x1/BatchNorm'),
            'branch1_conv2': layers.Conv2D(288, 3, strides=2, padding='same',
                                            name='Mixed_7a/Branch_1/Conv2d_1a_3x3'),
            'branch1_bn2': layers.BatchNormalization(
                name='Mixed_7a/Branch_1/Conv2d_1a_3x3/BatchNorm'),
            
            'branch2_conv1': layers.Conv2D(256, 1, padding='same',
                                            name='Mixed_7a/Branch_2/Conv2d_0a_1x1'),
            'branch2_bn1': layers.BatchNormalization(
                name='Mixed_7a/Branch_2/Conv2d_0a_1x1/BatchNorm'),
            'branch2_conv2': layers.Conv2D(288, 3, padding='same',
                                            name='Mixed_7a/Branch_2/Conv2d_0b_3x3'),
            'branch2_bn2': layers.BatchNormalization(
                name='Mixed_7a/Branch_2/Conv2d_0b_3x3/BatchNorm'),
            'branch2_conv3': layers.Conv2D(320, 3, strides=2, padding='same',
                                            name='Mixed_7a/Branch_2/Conv2d_1a_3x3'),
            'branch2_bn3': layers.BatchNormalization(
                name='Mixed_7a/Branch_2/Conv2d_1a_3x3/BatchNorm'),
            
            'branch3_pool': layers.MaxPooling2D(3, strides=2, padding='same',
                                                name='Mixed_7a/Branch_3/MaxPool_1a_3x3')
            }

        # Final layers
        self.conv2d_7b = layers.Conv2D(1536, 1, name='Conv2d_7b_1x1')
        self.conv2d_7b_bn = layers.BatchNormalization(name='Conv2d_7b_1x1/BatchNorm')
            
        self.avgpool = layers.GlobalAveragePooling2D(name='AvgPool_1a')
        self.dropout = layers.Dropout(1.0 - dropout_keep_prob)
        self.logits = layers.Dense(num_classes, name='Logits')
        self.predictions = layers.Activation('softmax', name='Predictions')

    def call(self, inputs, training=False):
        # Cast input to compute dtype (float16 for mixed precision)
        net = tf.cast(inputs, self._dtype_policy.compute_dtype)

        # Stem network
        net = self._build_stem(net, training)

        # Main inception blocks
        net = self._build_inception_blocks(net, training)

        # Final layers and predictions
        net = self._build_final_layers(net, training)

        # Cast output back to float32
        return tf.cast(net, tf.float32)

    def _build_stem(self, net, training):
        """Build the stem network."""
        # Initial convolution block
        net = self.conv2d_1a(net)
        net = self.conv2d_1a_bn(net, training=training)
        net = tf.nn.relu(net)
        
        net = self.conv2d_2a(net)
        net = self.conv2d_2a_bn(net, training=training)
        net = tf.nn.relu(net)
        
        net = self.conv2d_2b(net)
        net = self.conv2d_2b_bn(net, training=training)
        net = tf.nn.relu(net)
        
        # Reduction block A
        net = self.maxpool_3a(net)
        net = self.conv2d_3b(net)
        net = self.conv2d_3b_bn(net, training=training)
        net = tf.nn.relu(net)
        
        net = self.conv2d_4a(net)
        net = self.conv2d_4a_bn(net, training=training)
        net = tf.nn.relu(net)
        
        net = self.maxpool_5a(net)
        return net

    def _build_inception_blocks(self, net, training):
        """Build the main Inception-ResNet blocks."""
        # Inception-ResNet-A blocks
        for block in self.blocks35:
            net = block(net, training=training)

        # Reduction block B (Mixed_6a)
        net = self._build_reduction_b(net, training)

        # Inception-ResNet-B blocks
        for block in self.blocks17:
            net = block(net, training=training)

        # Reduction block C (Mixed_7a)
        net = self._build_reduction_c(net, training)

        # Inception-ResNet-C blocks
        for block in self.blocks8:
            net = block(net, training=training)
            
        net = self.block8_final(net, training=training)
        return net

    def _build_reduction_b(self, net, training):
        """Build reduction block B (Mixed_6a)."""
        with tf.name_scope('Mixed_6a'):
            # Branch 0
            branch0 = self.reduction_b['branch0_conv'](net)
            branch0 = self.reduction_b['branch0_bn'](branch0, training=training)
            branch0 = tf.nn.relu(branch0)
            
            # Branch 1
            branch1 = self.reduction_b['branch1_conv1'](net)
            branch1 = self.reduction_b['branch1_bn1'](branch1, training=training)
            branch1 = tf.nn.relu(branch1)
            branch1 = self.reduction_b['branch1_conv2'](branch1)
            branch1 = self.reduction_b['branch1_bn2'](branch1, training=training)
            branch1 = tf.nn.relu(branch1)
            branch1 = self.reduction_b['branch1_conv3'](branch1)
            branch1 = self.reduction_b['branch1_bn3'](branch1, training=training)
            branch1 = tf.nn.relu(branch1)
            
            # Branch 2
            branch2 = self.reduction_b['branch2_pool'](net)
            
            return tf.concat([branch0, branch1, branch2], axis=3)

    def _build_reduction_c(self, net, training):
        """Build reduction block C (Mixed_7a)."""
        with tf.name_scope('Mixed_7a'):
            # Branch 0
            branch0 = self.reduction_c['branch0_conv1'](net)
            branch0 = self.reduction_c['branch0_bn1'](branch0, training=training)
            branch0 = tf.nn.relu(branch0)
            branch0 = self.reduction_c['branch0_conv2'](branch0)
            branch0 = self.reduction_c['branch0_bn2'](branch0, training=training)
            branch0 = tf.nn.relu(branch0)

            # Branch 1
            branch1 = self.reduction_c['branch1_conv1'](net)
            branch1 = self.reduction_c['branch1_bn1'](branch1, training=training)
            branch1 = tf.nn.relu(branch1)
            branch1 = self.reduction_c['branch1_conv2'](branch1)
            branch1 = self.reduction_c['branch1_bn2'](branch1, training=training)
            branch1 = tf.nn.relu(branch1)

            # Branch 2
            branch2 = self.reduction_c['branch2_conv1'](net)
            branch2 = self.reduction_c['branch2_bn1'](branch2, training=training)
            branch2 = tf.nn.relu(branch2)
            branch2 = self.reduction_c['branch2_conv2'](branch2)
            branch2 = self.reduction_c['branch2_bn2'](branch2, training=training)
            branch2 = tf.nn.relu(branch2)
            branch2 = self.reduction_c['branch2_conv3'](branch2)
            branch2 = self.reduction_c['branch2_bn3'](branch2, training=training)
            branch2 = tf.nn.relu(branch2)

            # Branch 3
            branch3 = self.reduction_c['branch3_pool'](net)

            return tf.concat([branch0, branch1, branch2, branch3], axis=3)
    
    def _build_final_layers(self, net, training):
       """Build the final layers and predictions."""
       # Final convolution block
       net = self.conv2d_7b(net)
       net = self.conv2d_7b_bn(net, training=training)
       net = tf.nn.relu(net)

       # Global pooling and classification
       net = self.avgpool(net)
       net = self.dropout(net, training=training)
       net = self.logits(net)
       return self.predictions(net)
