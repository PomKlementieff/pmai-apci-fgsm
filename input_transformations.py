import tensorflow as tf

def apply_dim(x, resize_rate=1.1, diversity_prob=0.5):
    def input_diversity(x):
        orig_size = tf.shape(x)[1:3]
        min_size = tf.cast(orig_size[0], tf.float32)
        max_size = tf.cast(tf.cast(orig_size[0], tf.float32) * resize_rate, tf.int32)
        rnd = tf.random.uniform((), min_size, tf.cast(max_size, tf.float32), dtype=tf.float32)
        rescaled = tf.image.resize(x, [tf.cast(rnd, tf.int32), tf.cast(rnd, tf.int32)])
        h_rem = tf.maximum(orig_size[0] - tf.cast(rnd, tf.int32), 0)
        w_rem = tf.maximum(orig_size[1] - tf.cast(rnd, tf.int32), 0)
        pad_top = tf.random.uniform((), 0, h_rem + 1, dtype=tf.int32)
        pad_bottom = h_rem - pad_top
        pad_left = tf.random.uniform((), 0, w_rem + 1, dtype=tf.int32)
        pad_right = w_rem - pad_left
        padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
        return tf.image.resize(padded, orig_size)
    
    return tf.cond(tf.random.uniform(()) < diversity_prob, lambda: input_diversity(x), lambda: x)

def apply_tim(grad, kernel_size=5):
    kernel = tf.ones((kernel_size, kernel_size, 1, 1)) / (kernel_size * kernel_size)
    kernel = tf.tile(kernel, [1, 1, 3, 1])
    return tf.nn.depthwise_conv2d(grad, kernel, strides=[1, 1, 1, 1], padding='SAME')

def apply_sim(x, scale_factors=[1.0, 0.9, 0.8, 0.7, 0.6]):
    orig_size = tf.shape(x)[1:3]
    return [tf.image.resize(tf.image.resize(x, tf.cast(tf.cast(orig_size, tf.float32) * scale, tf.int32)), orig_size) for scale in scale_factors]

def apply_pim(x, grad, patch_size=16, alpha=1.0):
    """
    Apply Patch-wise Iterative Method (PIM)
    Reference: "Patch-wise attack for fooling deep neural network"
    """
    _, height, width, channels = x.shape
    
    # Adjust patch size if necessary
    if height % patch_size != 0 or width % patch_size != 0:
        new_patch_size = min(height, width)
        while new_patch_size > 1:
            if height % new_patch_size == 0 and width % new_patch_size == 0:
                patch_size = new_patch_size
                break
            new_patch_size -= 1
    
    # Split image and gradient into patches
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    
    # Reshape into patches
    grad_patches = tf.image.extract_patches(
        grad,
        sizes=[1, patch_size, patch_size, 1],
        strides=[1, patch_size, patch_size, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    
    # Normalize each patch independently
    grad_patches_norm = tf.reshape(grad_patches, [-1, num_patches_h * num_patches_w, patch_size * patch_size * channels])
    grad_patches_mean = tf.reduce_mean(tf.abs(grad_patches_norm), axis=2, keepdims=True)
    normalized_grad_patches = grad_patches_norm / (grad_patches_mean + 1e-8)
    
    # Apply smooth transition between patches
    kernel = tf.expand_dims(tf.expand_dims(
        tf.constant([[0.5, 1.0, 0.5],
                    [1.0, 1.0, 1.0],
                    [0.5, 1.0, 0.5]], dtype=tf.float32), -1), -1)
    kernel = kernel / tf.reduce_sum(kernel)
    
    # Reshape back to image space
    normalized_grad = tf.reshape(normalized_grad_patches, [-1, num_patches_h, num_patches_w, patch_size * patch_size * channels])
    normalized_grad = tf.reshape(normalized_grad, [-1, num_patches_h, num_patches_w, patch_size, patch_size, channels])
    normalized_grad = tf.transpose(normalized_grad, [0, 1, 3, 2, 4, 5])
    normalized_grad = tf.reshape(normalized_grad, [-1, height, width, channels])
    
    # Apply smoothing
    smoothed_grad = tf.nn.conv2d(
        normalized_grad,
        tf.tile(kernel, [1, 1, channels, 1]),
        strides=[1, 1, 1, 1],
        padding='SAME'
    )
    
    # Generate adversarial perturbation
    perturbation = alpha * tf.sign(smoothed_grad)
    
    # Add perturbation to input
    x_adv = x + perturbation
    
    return x_adv