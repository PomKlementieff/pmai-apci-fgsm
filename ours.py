import tensorflow as tf

def pmai_fgsm(model, x, y, eps, steps, decay, m, loss_fn=None):
    if loss_fn is None:
        loss_fn = lambda x: tf.keras.losses.categorical_crossentropy(y, model(x))
    
    alpha = eps / steps
    momentum = tf.zeros_like(x)
    x_adv = tf.identity(x)
    x_tilde = tf.identity(x)
    z = tf.identity(x)
    
    for k in range(steps):
        s = k // m
        j = k % m
        
        beta = s / (s + 2)
        
        x_k = (1 - beta) * z + beta * x_tilde
        
        with tf.GradientTape() as tape:
            tape.watch(x_k)
            loss = loss_fn(x_k)
        grad = tape.gradient(loss, x_k)
        
        grad = grad / tf.reduce_mean(tf.abs(grad), axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 93.5%
        #grad = grad / tf.math.reduce_euclidean_norm(grad, axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 92.9%
        momentum = decay * momentum + grad
        
        z = z - alpha * tf.sign(momentum)
        z = tf.clip_by_value(z, x - eps, x + eps)
        #z = tf.clip_by_value(z, 0, 1) # ASR when added: 91.5%
        
        if (k + 1) % m == 0:
            x_tilde = (1 - beta) / m * tf.reduce_sum([z for _ in range(m)], axis=0) + beta * x_tilde
    
    return z

def tmi_fgsm(model, x, y, eps, steps, decay, m, loss_fn=None):
    if loss_fn is None:
        loss_fn = lambda x: tf.keras.losses.categorical_crossentropy(y, model(x))
    
    alpha = eps / steps
    momentum = tf.zeros_like(x)
    x_adv = tf.identity(x)
    z = tf.identity(x)
    
    # Initialize the point table
    phi = tf.tile(tf.expand_dims(x, axis=1), [1, m, 1, 1, 1])
    
    for k in range(steps):
        beta = (k // m) / ((k // m) + 2)
        
        # Randomly select an index
        j = tf.random.uniform([], 0, m, dtype=tf.int32)
        
        # Compute x_k
        x_k = (1 - beta) * z + beta * phi[:, j]
        
        with tf.GradientTape() as tape:
            tape.watch(x_k)
            loss = loss_fn(x_k)
        grad = tape.gradient(loss, x_k)
        
        #grad = grad / tf.reduce_mean(tf.abs(grad), axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 86.9%
        grad = grad / tf.math.reduce_euclidean_norm(grad, axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 88.7%
        momentum = decay * momentum + grad
        
        z = z - alpha * tf.sign(momentum)
        z = tf.clip_by_value(z, x - eps, x + eps)
        #z = tf.clip_by_value(z, 0, 1) # ASR when added: 82.0%
        
        # Update the point table
        phi_update = (1 - beta) * z + beta * phi[:, j]
        indices = tf.stack([tf.range(tf.shape(phi)[0]), tf.repeat(j, tf.shape(phi)[0])], axis=1)
        phi = tf.tensor_scatter_nd_update(phi, indices, phi_update)
    
    # Compute the average of the point table for the final result
    x_adv = tf.reduce_mean(phi, axis=1)
    return x_adv

def apci_fgsm(model, x, y, eps, steps, beta1=0.9, beta2=0.999, learning_rate=0.01, weight_decay=0.004, epsilon=1e-7, amsgrad=False, clipnorm=None, clipvalue=None, global_clipnorm=None, loss_fn=None):
    if loss_fn is None:
        loss_fn = lambda x: tf.keras.losses.categorical_crossentropy(y, model(x))
    
    alpha = eps / steps
    x_adv = tf.identity(x)
    m = tf.zeros_like(x)
    v = tf.zeros_like(x)
    if amsgrad:
        v_hat = tf.zeros_like(x)
    
    for t in range(1, steps + 1):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            loss = loss_fn(x_adv)
        grad = tape.gradient(loss, x_adv)

        grad = grad / tf.reduce_mean(tf.abs(grad), axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 82.1%
        #grad = grad / tf.math.reduce_euclidean_norm(grad, axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 67.8%
        
        # Apply weight decay
        grad += weight_decay * x_adv
        
        # Apply gradient clipping
        if clipnorm is not None:
            grad = tf.clip_by_norm(grad, clipnorm)
        if clipvalue is not None:
            grad = tf.clip_by_value(grad, -clipvalue, clipvalue)
        if global_clipnorm is not None:
            grad, _ = tf.clip_by_global_norm([grad], global_clipnorm)
            grad = grad[0]
                
        # AdamW update
        beta1_power = tf.pow(beta1, tf.cast(t, tf.float32))
        beta2_power = tf.pow(beta2, tf.cast(t, tf.float32))
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * tf.square(grad)
        
        m_hat = m / (1 - beta1_power)
        v_hat_t = v / (1 - beta2_power)
        
        if amsgrad:
            v_hat = tf.maximum(v_hat, v_hat_t)
            update = m_hat / (tf.sqrt(v_hat) + epsilon)
        else:
            update = m_hat / (tf.sqrt(v_hat_t) + epsilon)
        
        # Apply update
        x_adv = x_adv - learning_rate * update
        
        # Project to epsilon ball
        x_adv = x_adv + alpha * tf.sign(x_adv - x)
        x_adv = tf.clip_by_value(x_adv, x - eps, x + eps)
        x_adv = tf.clip_by_value(x_adv, 0, 1)
    
    return x_adv