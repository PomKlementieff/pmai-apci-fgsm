import tensorflow as tf
from tensorflow.keras import layers, Model

class BasicConv2D(layers.Layer):
    def __init__(self, filters, kernel_size, strides=1, padding='valid', use_bias=False):
        super(BasicConv2D, self).__init__()
        self.conv = layers.Conv2D(filters, kernel_size, 
                                strides=strides,
                                padding=padding,
                                use_bias=use_bias)
        self.bn = layers.BatchNormalization(epsilon=0.001)
        self.relu = layers.ReLU()

    def call(self, x):
        x = tf.cast(x, self.compute_dtype)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DenoiseBlock(layers.Layer):
    def __init__(self, filters, strides=1):
        super(DenoiseBlock, self).__init__()
        self.conv = BasicConv2D(filters, 3, strides=strides, padding='same')

    def call(self, x):
        x = tf.cast(x, self.compute_dtype)
        return self.conv(x)

class Denoise(layers.Layer):
    def __init__(self, input_shape=(299, 299, 3)):
        super(Denoise, self).__init__()
        
        # Forward path
        self.fwd_layers = [
            [DenoiseBlock(64), DenoiseBlock(64)],  # First block
            [DenoiseBlock(128, strides=2), DenoiseBlock(128), DenoiseBlock(128)],
            [DenoiseBlock(256, strides=2), DenoiseBlock(256), DenoiseBlock(256)],
            [DenoiseBlock(256, strides=2), DenoiseBlock(256), DenoiseBlock(256)],
            [DenoiseBlock(256, strides=2), DenoiseBlock(256), DenoiseBlock(256)]
        ]

        # Backward path (upsampling)
        self.back_layers = [
            [DenoiseBlock(256), DenoiseBlock(256), DenoiseBlock(256)],
            [DenoiseBlock(256), DenoiseBlock(256), DenoiseBlock(256)],
            [DenoiseBlock(128), DenoiseBlock(128), DenoiseBlock(128)],
            [DenoiseBlock(64), DenoiseBlock(64)]
        ]
        
        # Calculate feature map sizes
        self.feature_map_sizes = []
        h, w = input_shape[0], input_shape[1]
        for i in range(5):  # 5 forward blocks
            if i > 0:  # After first block, size is halved
                h = (h + 1) // 2  # Add 1 before division to handle odd sizes
                w = (w + 1) // 2
            self.feature_map_sizes.append((h, w))
        
        # Reverse sizes for upsampling (excluding the last forward size)
        self.upsample_sizes = self.feature_map_sizes[-2::-1]
        
        # Final conv to match input channels
        self.final = layers.Conv2D(3, 1, use_bias=False)

    def call(self, x):
        x = tf.cast(x, self.compute_dtype)
        
        # Forward path
        forward_features = []
        out = x
        
        for i, block in enumerate(self.fwd_layers):
            # Apply convolution blocks
            for layer in block:
                out = layer(out)
            
            # Store intermediate features for skip connections
            if i < len(self.fwd_layers) - 1:  # Don't store the last feature map
                forward_features.append(tf.cast(out, self.compute_dtype))
        
        # Backward path with skip connections
        for i, block in enumerate(self.back_layers):
            # Resize to match corresponding forward feature map
            target_size = self.upsample_sizes[i]
            out = tf.image.resize(out, target_size, method='bilinear')
            out = tf.cast(out, self.compute_dtype)
            
            # Skip connection
            skip = tf.cast(forward_features[-(i+1)], self.compute_dtype)
            out = tf.concat([out, skip], axis=-1)
            
            # Apply convolution blocks
            for layer in block:
                out = layer(out)
        
        # Final convolution
        out = self.final(out)
        
        # Residual connection
        x = tf.cast(x, out.dtype)
        return x + out

class DenoiseInceptionV3(Model):
   def __init__(self):
       super(DenoiseInceptionV3, self).__init__()
       self.denoise = Denoise()
       self.inception = tf.keras.applications.InceptionV3(include_top=True)
       
   def call(self, x, training=False, defense=True):
       if defense:
           x = self.denoise(x)
       return self.inception(x, training=training)

class DenoiseInceptionResNetV2(Model):
   def __init__(self):
       super(DenoiseInceptionResNetV2, self).__init__()
       self.denoise = Denoise()
       self.inception_resnet = tf.keras.applications.InceptionResNetV2(include_top=True)
       
   def call(self, x, training=False, defense=True):
       if defense:
           x = self.denoise(x)
       return self.inception_resnet(x, training=training)

class DenoiseResNet(Model):
   def __init__(self):
       super(DenoiseResNet, self).__init__()
       self.denoise = Denoise(input_shape=(224, 224, 3))
       self.resnet = tf.keras.applications.ResNet101V2(include_top=True)
       
   def call(self, x, training=False, defense=True):
       if defense:
           x = self.denoise(x)
       return self.resnet(x, training=training)

def create_denoise_inception():
   return DenoiseInceptionV3()

def create_denoise_inception_resnet():
   return DenoiseInceptionResNetV2()

def create_denoise_resnet():
   return DenoiseResNet()
