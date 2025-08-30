import tensorflow as tf
from input_transformations import apply_dim, apply_tim, apply_sim, apply_pim

def hybrid_fgsm(model, x, y, eps,
                use_dim=False, use_tim=False, use_sim=False, use_pim=False,
                resize_rate=1.1, diversity_prob=0.5, kernel_size=5,
                scale_factors=[1.0, 0.9, 0.8, 0.7, 0.6], patch_size=16,
                loss_fn=None):
    """
    Hybrid FGSM with optional input transformations (DIM, TIM, SIM, PIM)
    
    Args:
        model: Target model
        x: Input images
        y: Target labels
        eps: Maximum perturbation
        use_dim: Whether to use Diverse Input Method
        use_tim: Whether to use Translation-Invariant Method
        use_sim: Whether to use Scale-Invariant Method
        use_pim: Whether to use Patch-wise Iterative Method
        resize_rate: Resize rate for DIM
        diversity_prob: Probability for DIM
        kernel_size: Kernel size for TIM
        scale_factors: Scale factors for SIM
        patch_size: Patch size for PIM
        loss_fn: Custom loss function (optional)
    """
    if loss_fn is None:
        loss_fn = lambda x: tf.keras.losses.categorical_crossentropy(y, model(x))
    
    # Apply input diversity if enabled
    x_input = apply_dim(x, resize_rate, diversity_prob) if use_dim else x
    
    # Compute gradients based on transformation methods
    if use_sim:
        x_scaled = apply_sim(x_input, scale_factors)
        grad_sum = tf.zeros_like(x)
        
        for x_s in x_scaled:
            with tf.GradientTape() as tape:
                tape.watch(x_s)
                loss = loss_fn(x_s)
            grad = tape.gradient(loss, x_s)
            if use_tim:
                grad = apply_tim(grad, kernel_size)
            grad_sum += grad
        grad = grad_sum / len(scale_factors)
    else:
        with tf.GradientTape() as tape:
            tape.watch(x_input)
            loss = loss_fn(x_input)
        grad = tape.gradient(loss, x_input)
        
        if use_tim:
            grad = apply_tim(grad, kernel_size)
    
    # Update step (similar to original FGSM)
    if use_pim:
        x_adv = apply_pim(x, eps * tf.sign(grad), patch_size)
    else:
        x_adv = x + eps * tf.sign(grad)
    
    # Project to valid image space (identical to original FGSM)
    x_adv = tf.clip_by_value(x_adv, x - eps, x + eps)
    x_adv = tf.clip_by_value(x_adv, 0, 1)
    
    return x_adv

def hybrid_pmai_fgsm(model, x, y, eps, steps, decay, m, 
                     use_dim=False, use_tim=False, use_sim=False, use_pim=False,
                     resize_rate=1.1, diversity_prob=0.5, kernel_size=5,
                     scale_factors=[1.0, 0.9, 0.8, 0.7, 0.6], patch_size=16,
                     loss_fn=None):
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
        
        # Apply input diversity if enabled
        x_k = apply_dim(x_k, resize_rate, diversity_prob) if use_dim else x_k
        
        # Compute gradients based on transformation methods
        if use_sim:
            x_scaled = apply_sim(x_k, scale_factors)
            grad_sum = tf.zeros_like(x)
            
            for x_s in x_scaled:
                with tf.GradientTape() as tape:
                    tape.watch(x_s)
                    loss = loss_fn(x_s)
                grad = tape.gradient(loss, x_s)
                if use_tim:
                    grad = apply_tim(grad, kernel_size)
                grad_sum += grad
            grad = grad_sum / len(scale_factors)
        else:
            with tf.GradientTape() as tape:
                tape.watch(x_k)
                loss = loss_fn(x_k)
            grad = tape.gradient(loss, x_k)
            
            if use_tim:
                grad = apply_tim(grad, kernel_size)

        grad = grad / tf.reduce_mean(tf.abs(grad), axis=[1,2,3], keepdims=True)
        momentum = decay * momentum + grad
        
        update = alpha * tf.sign(momentum)
        if use_pim:
            z = apply_pim(z, update, patch_size)
        else:
            z = z - update
            
        z = tf.clip_by_value(z, x - eps, x + eps)
        
        if (k + 1) % m == 0:
            x_tilde = (1 - beta) / m * tf.reduce_sum([z for _ in range(m)], axis=0) + beta * x_tilde
    
    return z

def hybrid_apci_fgsm(model, x, y, eps, steps, beta1=0.9, beta2=0.999, 
                    learning_rate=0.01, weight_decay=0.004, epsilon=1e-7, 
                    amsgrad=False, clipnorm=None, clipvalue=None, 
                    global_clipnorm=None, use_dim=False, use_tim=False, 
                    use_sim=False, use_pim=False, resize_rate=1.1, 
                    diversity_prob=0.5, kernel_size=5,
                    scale_factors=[1.0, 0.9, 0.8, 0.7, 0.6], 
                    patch_size=16, loss_fn=None):
    if loss_fn is None:
        loss_fn = lambda x: tf.keras.losses.categorical_crossentropy(y, model(x))
    
    alpha = eps / steps
    x_adv = tf.identity(x)
    m = tf.zeros_like(x)
    v = tf.zeros_like(x)
    if amsgrad:
        v_hat = tf.zeros_like(x)
    
    for t in range(1, steps + 1):
        # Apply input diversity if enabled
        x_cur = apply_dim(x_adv, resize_rate, diversity_prob) if use_dim else x_adv
        
        # Compute gradients based on transformation methods
        if use_sim:
            x_scaled = apply_sim(x_cur, scale_factors)
            grad_sum = tf.zeros_like(x)
            
            for x_s in x_scaled:
                with tf.GradientTape() as tape:
                    tape.watch(x_s)
                    loss = loss_fn(x_s)
                grad = tape.gradient(loss, x_s)
                if use_tim:
                    grad = apply_tim(grad, kernel_size)
                grad_sum += grad
            grad = grad_sum / len(scale_factors)
        else:
            with tf.GradientTape() as tape:
                tape.watch(x_cur)
                loss = loss_fn(x_cur)
            grad = tape.gradient(loss, x_cur)
            
            if use_tim:
                grad = apply_tim(grad, kernel_size)

        grad = grad / tf.reduce_mean(tf.abs(grad), axis=[1,2,3], keepdims=True)
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
        if use_pim:
            x_adv = apply_pim(x_adv, learning_rate * update, patch_size)
        else:
            x_adv = x_adv - learning_rate * update
        
        # Project to epsilon ball
        x_adv = x_adv + alpha * tf.sign(x_adv - x)
        x_adv = tf.clip_by_value(x_adv, x - eps, x + eps)
        x_adv = tf.clip_by_value(x_adv, 0, 1)
    
    return x_adv