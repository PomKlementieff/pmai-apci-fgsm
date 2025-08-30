import tensorflow as tf

def fgsm(model, x, y, eps, loss_fn=None):
    if loss_fn is None:
        loss_fn = lambda x: tf.keras.losses.categorical_crossentropy(y, model(x))
    
    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = loss_fn(x)
    grad = tape.gradient(loss, x)
    
    x_adv = x + eps * tf.sign(grad)
    x_adv = tf.clip_by_value(x_adv, x - eps, x + eps)
    x_adv = tf.clip_by_value(x_adv, 0, 1)
    
    return x_adv

def i_fgsm(model, x, y, eps, steps, loss_fn=None):
    if loss_fn is None:
        loss_fn = lambda x: tf.keras.losses.categorical_crossentropy(y, model(x))
    
    alpha = eps / steps
    x_adv = tf.identity(x)
    
    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            loss = loss_fn(x_adv)
        grad = tape.gradient(loss, x_adv)
        
        x_adv = x_adv + alpha * tf.sign(grad)
        x_adv = tf.clip_by_value(x_adv, x - eps, x + eps)
        x_adv = tf.clip_by_value(x_adv, 0, 1)
    
    return x_adv

def mi_fgsm(model, x, y, eps, steps, decay, loss_fn=None):
    if loss_fn is None:
        loss_fn = lambda x: tf.keras.losses.categorical_crossentropy(y, model(x))
    
    alpha = eps / steps
    momentum = tf.zeros_like(x)
    x_adv = tf.identity(x)
    
    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            loss = loss_fn(x_adv)
        grad = tape.gradient(loss, x_adv)
        
        grad = grad / tf.reduce_mean(tf.abs(grad), axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 48.2%
        #grad = grad / tf.math.reduce_euclidean_norm(grad, axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 48.0%
        momentum = decay * momentum + grad
        
        x_adv = x_adv + alpha * tf.sign(momentum)
        x_adv = tf.clip_by_value(x_adv, x - eps, x + eps)
        x_adv = tf.clip_by_value(x_adv, 0, 1)
    
    return x_adv

def ni_fgsm(model, x, y, eps, steps, decay, loss_fn=None):
    if loss_fn is None:
        loss_fn = lambda x: tf.keras.losses.categorical_crossentropy(y, model(x))
    
    alpha = eps / steps
    momentum = tf.zeros_like(x)
    x_adv = tf.identity(x)
    
    for _ in range(steps):
        x_nes = x_adv + decay * alpha * momentum
        
        with tf.GradientTape() as tape:
            tape.watch(x_nes)
            loss = loss_fn(x_nes)
        grad = tape.gradient(loss, x_nes)
        
        grad = grad / tf.reduce_mean(tf.abs(grad), axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 48.1%
        #grad = grad / tf.math.reduce_euclidean_norm(grad, axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 48.1%
        momentum = decay * momentum + grad
        
        x_adv = x_adv + alpha * tf.sign(momentum)
        x_adv = tf.clip_by_value(x_adv, x - eps, x + eps)
        x_adv = tf.clip_by_value(x_adv, 0, 1)
    
    return x_adv

def pi_fgsm(model, x, y, eps, steps, decay, loss_fn=None):
    if loss_fn is None:
        loss_fn = lambda x: tf.keras.losses.categorical_crossentropy(y, model(x))
    
    alpha = eps / steps
    momentum = tf.zeros_like(x)
    grad_prev = tf.zeros_like(x)
    x_adv = tf.identity(x)
    
    for _ in range(steps):
        x_nes = x_adv + alpha * grad_prev
        
        with tf.GradientTape() as tape:
            tape.watch(x_nes)
            loss = loss_fn(x_nes)
        grad = tape.gradient(loss, x_nes)
        
        grad = grad / tf.reduce_mean(tf.abs(grad), axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 48.1%
        #grad = grad / tf.math.reduce_euclidean_norm(grad, axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 48.1%
        momentum = decay * momentum + grad
        grad_prev = grad
        
        x_adv = x_adv + alpha * tf.sign(momentum)
        x_adv = tf.clip_by_value(x_adv, x - eps, x + eps)
        x_adv = tf.clip_by_value(x_adv, 0, 1)
    
    return x_adv

def emi_fgsm(model, x, y, eps, steps, decay, N, eta, loss_fn=None):
    if loss_fn is None:
        loss_fn = lambda x: tf.keras.losses.categorical_crossentropy(y, model(x))
    
    alpha = eps / steps
    momentum = tf.zeros_like(x)
    x_adv = tf.identity(x)
    
    for _ in range(steps):
        grad_avg = tf.zeros_like(x)
        for _ in range(N):
            c = tf.random.uniform(shape=(), minval=-eta, maxval=eta)
            x_sample = x_adv + c * momentum
            
            with tf.GradientTape() as tape:
                tape.watch(x_sample)
                loss = loss_fn(x_sample)
            grad = tape.gradient(loss, x_sample)
            
            #grad = grad / tf.reduce_mean(tf.abs(grad), axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 48.5%
            grad = grad / tf.math.reduce_euclidean_norm(grad, axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 48.7%
            grad_avg += grad
        
        grad_avg = grad_avg / N
        momentum = decay * momentum + grad_avg
        
        x_adv = x_adv + alpha * tf.sign(momentum)
        x_adv = tf.clip_by_value(x_adv, x - eps, x + eps)
        x_adv = tf.clip_by_value(x_adv, 0, 1)
    
    return x_adv

def sni_fgsm(model, x, y, eps, steps, decay, loss_fn=None):
    if loss_fn is None:
        loss_fn = lambda x: tf.keras.losses.categorical_crossentropy(y, model(x))
    
    alpha = eps / steps
    momentum = tf.zeros_like(x)
    x_adv = tf.identity(x)
    
    for _ in range(steps):
        x_nes = x_adv + decay * momentum
        
        with tf.GradientTape() as tape:
            tape.watch(x_nes)
            loss = loss_fn(x_nes)
        grad = tape.gradient(loss, x_nes)
        
        grad = grad / tf.reduce_mean(tf.abs(grad), axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 46.6%
        #grad = grad / tf.math.reduce_euclidean_norm(grad, axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 46.6%
        momentum = decay * momentum + (1 - decay) * grad
        
        x_adv = x_adv + alpha * tf.sign(momentum)
        x_adv = tf.clip_by_value(x_adv, x - eps, x + eps)
        x_adv = tf.clip_by_value(x_adv, 0, 1)
    
    return x_adv

def bni_fgsm(model, x, y, eps, steps, decay, loss_fn=None):
    if loss_fn is None:
        loss_fn = lambda x: tf.keras.losses.categorical_crossentropy(y, model(x))
    
    alpha = eps / steps
    v = tf.zeros_like(x)
    x_adv = tf.identity(x)
    
    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            loss = loss_fn(x_adv)
        grad = tape.gradient(loss, x_adv)
        
        #grad = grad / tf.reduce_mean(tf.abs(grad), axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 58.7%
        grad = grad / tf.math.reduce_euclidean_norm(grad, axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 60.1%
        
        x_adv = x_adv + decay * decay * v - (1 + decay) * alpha * tf.sign(grad)
        v = decay * v - alpha * tf.sign(grad)
        
        x_adv = tf.clip_by_value(x_adv, x - eps, x + eps)
        #x_adv = tf.clip_by_value(x_adv, 0, 1) ASR when added: 74.6%
    
    return x_adv

def dni_fgsm(model, x, y, eps, steps, decay, loss_fn=None):
    if loss_fn is None:
        loss_fn = lambda x: tf.keras.losses.categorical_crossentropy(y, model(x))
    
    alpha = eps / steps
    m_prime = tf.zeros_like(x)
    x_adv = tf.identity(x)
    
    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            loss = loss_fn(x_adv)
        grad = tape.gradient(loss, x_adv)
        
        grad = grad / tf.reduce_mean(tf.abs(grad), axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 59.9%
        #grad = grad / tf.math.reduce_euclidean_norm(grad, axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 59.4%
        
        s = -alpha * tf.sign(grad)
        v = (1 + decay) * s + decay * decay * m_prime
        x_adv = x_adv + v
        m_prime = decay * m_prime + s
        
        x_adv = tf.clip_by_value(x_adv, x - eps, x + eps)
        #x_adv = tf.clip_by_value(x_adv, 0, 1) # ASR when added: 73.3%
    
    return x_adv

def qhmi_fgsm(model, x, y, eps, steps, decay, nu, loss_fn=None):
    if loss_fn is None:
        loss_fn = lambda x: tf.keras.losses.categorical_crossentropy(y, model(x))
    
    alpha = eps / steps
    q = tf.zeros_like(x)
    x_adv = tf.identity(x)
    
    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            loss = loss_fn(x_adv)
        grad = tape.gradient(loss, x_adv)
        
        #grad = grad / tf.reduce_mean(tf.abs(grad), axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 48.1%
        grad = grad / tf.math.reduce_euclidean_norm(grad, axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 48.2%
        
        q = decay * q + grad
        x_adv = x_adv + alpha * tf.sign((1 - nu) * grad + nu * q)
        
        x_adv = tf.clip_by_value(x_adv, x - eps, x + eps)
        x_adv = tf.clip_by_value(x_adv, 0, 1)
    
    return x_adv

def anagi_fgsm(model, x, y, eps, steps, decay, loss_fn=None):
    if loss_fn is None:
        loss_fn = lambda x: tf.keras.losses.categorical_crossentropy(y, model(x))
    
    alpha = eps / steps
    v = tf.zeros_like(x)
    x_adv = x + tf.random.uniform(tf.shape(x), -eps, eps)
    x_adv = tf.clip_by_value(x_adv, 0, 1)
    
    for _ in range(steps):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            loss = loss_fn(x_adv)
        grad = tape.gradient(loss, x_adv)

        grad = grad / tf.reduce_mean(tf.abs(grad), axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 63.2%
        #grad = grad / tf.math.reduce_euclidean_norm(grad, axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 50.3%
        
        v = decay * (v - alpha * grad) - alpha * grad
        x_adv = x_adv + v
        v = v + alpha * grad
        
        x_adv = tf.clip_by_value(x_adv, x - eps, x + eps)
        x_adv = tf.clip_by_value(x_adv, 0, 1)
    
    return x_adv

def ai_fgsm(model, x, y, eps, steps, beta1=0.9, beta2=0.999, learning_rate=0.01, epsilon=1e-7, amsgrad=False, weight_decay=0.0, lr_schedule=None, loss_fn=None):
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

        grad = grad / tf.reduce_mean(tf.abs(grad), axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 48.2%
        #grad = grad / tf.math.reduce_euclidean_norm(grad, axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 37.7%
        
        if weight_decay > 0:
            grad += weight_decay * x_adv
        
        # Adam update
        beta1_power = tf.pow(beta1, t)
        beta2_power = tf.pow(beta2, t)
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * tf.square(grad)
        
        m_hat = m / (1 - beta1_power)
        v_hat_t = v / (1 - beta2_power)
        
        if amsgrad:
            v_hat = tf.maximum(v_hat, v_hat_t)
            update = m_hat / (tf.sqrt(v_hat) + epsilon)
        else:
            update = m_hat / (tf.sqrt(v_hat_t) + epsilon)
        
        # Apply learning rate schedule if provided
        if lr_schedule is not None:
            current_lr = lr_schedule(t)
        else:
            current_lr = learning_rate
        
        # Compute and apply update
        update = current_lr * update
        x_adv = x_adv + alpha * tf.sign(update)
        x_adv = tf.clip_by_value(x_adv, x - eps, x + eps)
        x_adv = tf.clip_by_value(x_adv, 0, 1)
    
    return x_adv

def api_fgsm(model, x, y, eps, steps, beta_1=0.9, beta_2=0.999, learning_rate=0.001, epsilon=1e-8, weight_decay=0.0, delta=0.1, wd_ratio=0.1, nesterov=False, loss_fn=None):
    if loss_fn is None:
        loss_fn = lambda x: tf.keras.losses.categorical_crossentropy(y, model(x))
    
    def channel_view(x):
        return tf.reshape(x, shape=[x.shape[0], -1])

    def layer_view(x):
        return tf.reshape(x, shape=[1, -1])

    def cosine_similarity(x, y, eps, view_func):
        x = view_func(x)
        y = view_func(y)
        
        x_norm = tf.norm(x, axis=-1) + eps
        y_norm = tf.norm(y, axis=-1) + eps
        dot = tf.reduce_sum(x * y, axis=-1)
        
        return tf.abs(dot) / x_norm / y_norm

    def projection(var, grad, perturb, delta, wd_ratio, eps):
        def project(view_func):
            cosine_sim = cosine_similarity(grad, var, eps, view_func)
            cosine_max = tf.reduce_max(cosine_sim)
            compare_val = delta / tf.sqrt(tf.cast(tf.shape(view_func(var))[-1], dtype=tf.float32))
            
            if cosine_max < compare_val:
                var_n = var / (tf.norm(view_func(var), axis=-1, keepdims=True) + eps)
                new_perturb = perturb - var_n * tf.reduce_sum(view_func(var_n * perturb), axis=-1, keepdims=True)
                return new_perturb, wd_ratio
            return None

        channel_proj = project(channel_view)
        if channel_proj is not None:
            return channel_proj
        
        layer_proj = project(layer_view)
        if layer_proj is not None:
            return layer_proj
        
        return perturb, 1.0

    x_adv = tf.identity(x)
    m = tf.zeros_like(x)
    v = tf.zeros_like(x)
    
    alpha = eps / steps  # 알파 추가
    
    for t in range(1, steps + 1):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            loss = loss_fn(x_adv)
        grad = tape.gradient(loss, x_adv)
        
        # Gradient normalization
        grad = grad / (tf.reduce_mean(tf.abs(grad), axis=[1,2,3], keepdims=True) + epsilon) # SM: InceptionV3, TM: ResNet101, ASR: 46.6%
        #grad = grad / tf.math.reduce_euclidean_norm(grad, axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 46.6%
        
        # AdamP update
        beta_1_power = tf.pow(beta_1, tf.cast(t, dtype=tf.float32))
        beta_2_power = tf.pow(beta_2, tf.cast(t, dtype=tf.float32))
        
        m = beta_1 * m + (1 - beta_1) * grad
        v = beta_2 * v + (1 - beta_2) * tf.square(grad)
        
        m_hat = m / (1 - beta_1_power)
        v_hat = v / (1 - beta_2_power)
        
        denominator = tf.sqrt(v_hat) / tf.sqrt(1 - beta_2_power) + epsilon
        step_size = learning_rate / (1 - beta_1_power)
        
        if nesterov:
            perturb = (beta_1 * m_hat + (1 - beta_1) * grad) / denominator
        else:
            perturb = m_hat / denominator
        
        # Projection
        if len(x_adv.shape) > 1:
            perturb, proj_wd_ratio = projection(x_adv, grad, perturb, delta, wd_ratio, epsilon)
        else:
            proj_wd_ratio = 1.0
        
        # Weight decay
        if weight_decay > 0:
            x_adv = x_adv * (1 - step_size * weight_decay * proj_wd_ratio)
        
        # Apply update
        x_adv = x_adv + alpha * step_size * tf.sign(perturb)  # alpha 사용
        x_adv = tf.clip_by_value(x_adv, x - eps, x + eps)
        x_adv = tf.clip_by_value(x_adv, 0, 1)
    
    return x_adv

def sgdp_fgsm(model, x, y, eps, delta=0.1, wd_ratio=0.1, momentum=0.9, dampening=0.0, nesterov=False, epsilon=1e-8, loss_fn=None):
    if loss_fn is None:
        loss_fn = lambda x: tf.keras.losses.categorical_crossentropy(y, model(x))
    
    def _channel_view(x):
        return tf.reshape(x, shape=[tf.shape(x)[0], -1])

    def _layer_view(x):
        return tf.reshape(x, shape=[1, -1])

    def _cosine_similarity(x, y, eps, view_func):
        x = view_func(x)
        y = view_func(y)

        x_norm = tf.norm(x, axis=-1) + eps
        y_norm = tf.norm(y, axis=-1) + eps
        dot = tf.reduce_sum(x * y, axis=-1)

        return tf.abs(dot) / x_norm / y_norm

    def _projection(var, grad, perturb, delta, wd_ratio, eps):
        cosine_sim = _cosine_similarity(grad, var, eps, _channel_view)
        cosine_max = tf.reduce_max(cosine_sim)
        compare_val = delta / tf.sqrt(tf.cast(tf.shape(_channel_view(var))[-1], delta.dtype))

        def channel_projection():
            var_n = var / (tf.reshape(tf.norm(_channel_view(var), axis=-1), shape=[-1] + [1] * (len(var.shape) - 1)) + eps)
            new_perturb = perturb - var_n * tf.reshape(tf.reduce_sum(_channel_view(var_n * perturb), axis=-1), shape=[-1] + [1] * (len(var.shape) - 1))
            return new_perturb, wd_ratio

        def layer_projection():
            cosine_sim = _cosine_similarity(grad, var, eps, _layer_view)
            cosine_max = tf.reduce_max(cosine_sim)
            compare_val = delta / tf.sqrt(tf.cast(tf.shape(_layer_view(var))[-1], delta.dtype))

            def do_layer_projection():
                var_n = var / (tf.reshape(tf.norm(_layer_view(var), axis=-1), shape=[-1] + [1] * (len(var.shape) - 1)) + eps)
                new_perturb = perturb - var_n * tf.reshape(tf.reduce_sum(_layer_view(var_n * perturb), axis=-1), shape=[-1] + [1] * (len(var.shape) - 1))
                return new_perturb, wd_ratio

            def no_projection():
                return perturb, tf.constant(1.0, dtype=perturb.dtype)

            return tf.cond(cosine_max < compare_val, do_layer_projection, no_projection)

        return tf.cond(cosine_max < compare_val, channel_projection, layer_projection)
    
    # delta와 wd_ratio를 텐서로 변환
    delta = tf.cast(delta, dtype=x.dtype)
    wd_ratio = tf.cast(wd_ratio, dtype=x.dtype)

    with tf.GradientTape() as tape:
        tape.watch(x)
        loss = loss_fn(x)
    grad = tape.gradient(loss, x)
    
    buf = tf.zeros_like(x)
    if momentum > 0:
        buf = momentum * buf + (1 - dampening) * grad
        if nesterov:
            grad = grad + momentum * buf
        else:
            grad = buf

    if len(x.shape) > 1:
        grad, wd_ratio = _projection(x, grad, grad, delta, wd_ratio, epsilon)
    
    x_adv = x + eps * tf.sign(grad)
    x_adv = tf.clip_by_value(x_adv, x - eps, x + eps)
    x_adv = tf.clip_by_value(x_adv, 0, 1)
    
    return x_adv

def nai_fgsm(model, x, y, eps, steps, beta1=0.9, beta2=0.999, epsilon=1e-7, learning_rate=0.001, decay=0.96, loss_fn=None):
    if loss_fn is None:
        loss_fn = lambda x: tf.keras.losses.categorical_crossentropy(y, model(x))
    
    alpha = eps / steps
    x_adv = tf.identity(x)
    m = tf.zeros_like(x)
    v = tf.zeros_like(x)
    u_product = tf.constant(1.0, dtype=x.dtype)
    
    for t in range(1, steps + 1):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            loss = loss_fn(x_adv)
        grad = tape.gradient(loss, x_adv)
        
        grad = grad / tf.reduce_mean(tf.abs(grad), axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 47.8%
        #grad = grad / tf.math.reduce_euclidean_norm(grad, axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 47.7%
        
        # Nadam update
        local_step = tf.cast(t, x.dtype)
        next_step = tf.cast(t + 1, x.dtype)
        decay_t = tf.cast(decay, x.dtype)
        
        u_t = beta1 * (1.0 - 0.5 * (tf.pow(decay_t, local_step)))
        u_t_1 = beta1 * (1.0 - 0.5 * (tf.pow(decay_t, next_step)))
        
        u_product_t = u_product * u_t
        u_product_t_1 = u_product_t * u_t_1
        
        beta2_power = tf.pow(beta2, local_step)
        
        m_t = beta1 * m + (1 - beta1) * grad
        v_t = beta2 * v + (1 - beta2) * tf.square(grad)
        
        m_hat = (u_t_1 * m_t / (1 - u_product_t_1)) + ((1 - u_t) * grad / (1 - u_product_t))
        v_hat = v_t / (1 - beta2_power)
        
        update = learning_rate * m_hat / (tf.sqrt(v_hat) + epsilon)
        
        # Apply update
        x_adv = x_adv + alpha * tf.sign(update)
        x_adv = tf.clip_by_value(x_adv, x - eps, x + eps)
        x_adv = tf.clip_by_value(x_adv, 0, 1)
        
        # Update for next iteration
        m = m_t
        v = v_t
        u_product = u_product_t
    
    return x_adv

def soap_fgsm(model, x, y, eps, steps, beta1=0.95, beta2=0.95, learning_rate=0.001, epsilon=1e-8, preconditioning_freq=10, loss_fn=None):
    if loss_fn is None:
        loss_fn = lambda x: tf.keras.losses.categorical_crossentropy(y, model(x))
    
    def update_eigenvectors(P, Q):
        S = tf.matmul(P, Q)
        Q, _ = tf.linalg.qr(S)
        return Q
    
    alpha = eps / steps
    x_adv = tf.identity(x)
    m = tf.zeros_like(x)
    v = tf.zeros_like(x)
    
    x_shape = tf.shape(x)
    channels = x_shape[-1]
    
    L = tf.zeros([channels, channels])
    Q_L = tf.eye(channels)
    
    for t in range(1, steps + 1):
        with tf.GradientTape() as tape:
            tape.watch(x_adv)
            loss = loss_fn(x_adv)
        grad = tape.gradient(loss, x_adv)

        #grad = grad / tf.reduce_mean(tf.abs(grad), axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 47.9%
        grad = grad / tf.math.reduce_euclidean_norm(grad, axis=[1,2,3], keepdims=True) # SM: InceptionV3, TM: ResNet101, ASR: 48.0%
        
        # Update L
        grad_flat = tf.reshape(grad, [-1, channels])
        L = beta2 * L + (1 - beta2) * tf.matmul(tf.transpose(grad_flat), grad_flat)
        
        # Update Q_L every preconditioning_freq steps
        if t % preconditioning_freq == 0:
            Q_L = update_eigenvectors(L, Q_L)
        
        # Project gradient to eigenbasis (channel-wise)
        grad_proj = tf.tensordot(grad, Q_L, axes=[[3], [0]])
        
        # Update momentum and second moment estimates
        m = beta1 * m + (1 - beta1) * grad_proj
        v = beta2 * v + (1 - beta2) * tf.square(grad_proj)
        
        # Compute bias-corrected estimates
        m_hat = m / (1 - tf.pow(beta1, tf.cast(t, dtype=tf.float32)))
        v_hat = v / (1 - tf.pow(beta2, tf.cast(t, dtype=tf.float32)))
        
        # Compute update
        update = learning_rate * m_hat / (tf.sqrt(v_hat) + epsilon)
        
        # Project update back to original space (channel-wise)
        update = tf.tensordot(update, tf.transpose(Q_L), axes=[[3], [0]])
        
        # Apply update
        x_adv = x_adv + alpha * tf.sign(update)
        x_adv = tf.clip_by_value(x_adv, x - eps, x + eps)
        x_adv = tf.clip_by_value(x_adv, 0, 1)
    
    return x_adv