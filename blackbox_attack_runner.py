import argparse
import tensorflow as tf
from attack_utils import load_model, load_dataset, get_input_size, resize_batch
from adversarial_methods import (
    fgsm, i_fgsm, mi_fgsm, ni_fgsm, pi_fgsm, emi_fgsm,
    sni_fgsm, bni_fgsm, dni_fgsm, qhmi_fgsm, anagi_fgsm,
    ai_fgsm, api_fgsm, sgdp_fgsm, nai_fgsm, soap_fgsm
)
from ours import pmai_fgsm, tmi_fgsm, apci_fgsm
from hybrid_methods import hybrid_fgsm, hybrid_pmai_fgsm, hybrid_apci_fgsm

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

def main(args):
    substitute_model = load_model(args.substitute_model)
    target_model = load_model(args.target_model)
    
    substitute_input_size = get_input_size(args.substitute_model)
    target_input_size = get_input_size(args.target_model)
    
    dataset = load_dataset(args.data_dir, args.batch_size, image_size=substitute_input_size)
    
    # Define learning rate schedule if specified
    if args.lr_schedule:
        lr_schedule = lambda t: args.learning_rate * (0.1 ** (t // args.lr_decay_steps))
    else:
        lr_schedule = None

    attack_functions = {
        'fgsm':      lambda x, y: fgsm(substitute_model, x, y, args.eps),
        'i_fgsm':    lambda x, y: i_fgsm(substitute_model, x, y, args.eps, args.steps),
        'mi_fgsm':   lambda x, y: mi_fgsm(substitute_model, x, y, args.eps, args.steps, args.decay),
        'ni_fgsm':   lambda x, y: ni_fgsm(substitute_model, x, y, args.eps, args.steps, args.decay),
        'pi_fgsm':   lambda x, y: pi_fgsm(substitute_model, x, y, args.eps, args.steps, args.decay),
        'emi_fgsm':  lambda x, y: emi_fgsm(substitute_model, x, y, args.eps, args.steps, args.decay, 
                                        args.N, args.eta),
        'sni_fgsm':  lambda x, y: sni_fgsm(substitute_model, x, y, args.eps, args.steps, args.decay),
        'bni_fgsm':  lambda x, y: bni_fgsm(substitute_model, x, y, args.eps, args.steps, args.decay),
        'dni_fgsm':  lambda x, y: dni_fgsm(substitute_model, x, y, args.eps, args.steps, args.decay),
        'qhmi_fgsm': lambda x, y: qhmi_fgsm(substitute_model, x, y, args.eps, args.steps, args.decay, 
                                            args.nu),
        'anagi_fgsm':lambda x, y: anagi_fgsm(substitute_model, x, y, args.eps, args.steps, args.decay),
        'ai_fgsm':   lambda x, y: ai_fgsm(substitute_model, x, y, args.eps, args.steps, args.beta1, 
                                        args.beta2, args.learning_rate, args.epsilon, args.amsgrad, 
                                        args.weight_decay, lr_schedule),
        'api_fgsm':  lambda x, y: api_fgsm(substitute_model, x, y, args.eps, args.steps, args.beta1, 
                                        args.beta2, args.learning_rate, args.epsilon, 
                                        args.weight_decay, args.delta, args.wd_ratio, args.nesterov),
        'nai_fgsm':  lambda x, y: nai_fgsm(substitute_model, x, y, args.eps, args.steps, args.beta1, 
                                        args.beta2, args.epsilon, args.learning_rate, args.decay),
        'soap_fgsm': lambda x, y: soap_fgsm(substitute_model, x, y, args.eps, args.steps, args.beta1, 
                                            args.beta2, args.learning_rate, args.epsilon, 
                                            args.preconditioning_freq),
        'sgdp_fgsm': lambda x, y: sgdp_fgsm(substitute_model, x, y, args.eps, delta=args.delta, 
                                            wd_ratio=args.wd_ratio, momentum=args.momentum, 
                                            dampening=args.dampening, nesterov=args.nesterov, 
                                            epsilon=args.epsilon),
        # Our methods - basic versions
        'pmai_fgsm': lambda x, y: pmai_fgsm(substitute_model, x, y, args.eps, args.steps, args.decay, 
                                            args.m),
        'tmi_fgsm': lambda x, y: tmi_fgsm(substitute_model, x, y, args.eps, args.steps, args.decay, 
                                            args.m),
        'apci_fgsm':  lambda x, y: apci_fgsm(substitute_model, x, y, args.eps, args.steps, args.beta1, 
                                        args.beta2, args.learning_rate, args.weight_decay, 
                                        args.epsilon, args.amsgrad, args.clipnorm, args.clipvalue, 
                                        args.global_clipnorm),
        # Hybrid versions
        'hybrid_fgsm': lambda x, y: hybrid_fgsm(substitute_model, x, y, args.eps,
                                           use_dim=args.use_dim, use_tim=args.use_tim,
                                           use_sim=args.use_sim, use_pim=args.use_pim,
                                           resize_rate=args.resize_rate,
                                           diversity_prob=args.diversity_prob,
                                           kernel_size=args.kernel_size,
                                           scale_factors=args.scale_factors,
                                           patch_size=args.patch_size),
        'hybrid_pmai_fgsm': lambda x, y: hybrid_pmai_fgsm(substitute_model, x, y, args.eps, 
                                                      args.steps, args.decay, args.m,
                                                      use_dim=args.use_dim, use_tim=args.use_tim,
                                                      use_sim=args.use_sim, use_pim=args.use_pim,
                                                      resize_rate=args.resize_rate,
                                                      diversity_prob=args.diversity_prob,
                                                      kernel_size=args.kernel_size,
                                                      scale_factors=args.scale_factors,
                                                      patch_size=args.patch_size),
        'hybrid_apci_fgsm': lambda x, y: hybrid_apci_fgsm(substitute_model, x, y, args.eps, 
                                                    args.steps, args.beta1, args.beta2,
                                                    args.learning_rate, args.weight_decay,
                                                    args.epsilon, args.amsgrad,
                                                    args.clipnorm, args.clipvalue,
                                                    args.global_clipnorm,
                                                    use_dim=args.use_dim, use_tim=args.use_tim,
                                                    use_sim=args.use_sim, use_pim=args.use_pim,
                                                    resize_rate=args.resize_rate,
                                                    diversity_prob=args.diversity_prob,
                                                    kernel_size=args.kernel_size,
                                                    scale_factors=args.scale_factors,
                                                    patch_size=args.patch_size),
    }
    if args.attack not in attack_functions:
        raise ValueError(f"Unknown attack: {args.attack}")
    
    attack_fn = attack_functions[args.attack]
    
    total_images = 0
    successful_attacks = 0
    
    for batch in dataset:
        y_pred_substitute = substitute_model.predict(batch)
        y_true_substitute = tf.one_hot(tf.argmax(y_pred_substitute, axis=1), depth=1000)
        
        x_adv = attack_fn(batch, y_true_substitute)
        
        x_adv_resized = resize_batch(x_adv, target_input_size)
        batch_resized = resize_batch(batch, target_input_size)
        
        y_pred_target = target_model.predict(batch_resized)
        y_adv_target = target_model.predict(x_adv_resized)
        
        successful_attacks += tf.reduce_sum(tf.cast(tf.argmax(y_adv_target, axis=1) != tf.argmax(y_pred_target, axis=1), tf.float32))
        total_images += batch.shape[0]
    
    success_rate = successful_attacks / total_images
    print(f"Black-box attack success rate: {success_rate:.2%}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform black-box adversarial attacks.')

    # Dataset arguments
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Path to the dataset directory.')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for processing. (Default: 32)')

    # Model arguments
    parser.add_argument('--substitute_model', type=str, required=True,
                        help='Name of the substitute model to use for generating adversarial examples.')
    parser.add_argument('--target_model', type=str, required=True,
                        help='Name of the target model to attack.')

    # General attack arguments
    parser.add_argument('--attack', type=str, required=True, 
                    choices=['fgsm', 'i_fgsm', 'mi_fgsm', 'ni_fgsm', 'pi_fgsm', 'emi_fgsm',
                             'sni_fgsm', 'bni_fgsm', 'dni_fgsm', 'qhmi_fgsm', 'anagi_fgsm',
                             'pmai_fgsm', 'tmi_fgsm', 'ai_fgsm', 'apci_fgsm',
                             'api_fgsm', 'sgdp_fgsm', 'nai_fgsm', 'soap_fgsm',
                             'hybrid_fgsm', 'hybrid_pmai_fgsm', 'hybrid_apci_fgsm'],
                    help="Choose the attack method from the available options.")
    parser.add_argument('--eps', type=float, default=0.3, 
                        help="The maximum perturbation allowed. (Default: 0.3)")
    parser.add_argument('--steps', type=int, default=10, 
                        help="The number of iterations for iterative attacks. (Default: 10)")

    # Common optimizer parameters
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help="Learning rate for optimization-based attacks. (Default: 0.01)")
    parser.add_argument('--beta1', type=float, default=0.9,
                        help="Beta1 parameter for Adam-based attacks. (Default: 0.9)")
    parser.add_argument('--beta2', type=float, default=0.999,
                        help="Beta2 parameter for Adam-based attacks. (Default: 0.999)")
    parser.add_argument('--epsilon', type=float, default=1e-7,
                        help="Epsilon parameter for optimization-based attacks. (Default: 1e-7)")
    parser.add_argument('--weight_decay', type=float, default=0.0,
                        help="Weight decay parameter for optimization-based attacks. (Default: 0.0)")
    parser.add_argument('--momentum', type=float, default=0.9, 
                        help="Momentum for momentum-based attacks. (Default: 0.9)")
    parser.add_argument('--decay', type=float, default=1.0, 
                        help="Decay factor for momentum-based attacks. (Default: 1.0)")

    # Specific attack arguments
    parser.add_argument('--N', type=int, default=5, 
                        help="Number of sampled gradients for EMI-FGSM. (Default: 5)")
    parser.add_argument('--eta', type=float, default=0.1, 
                        help="Scaling factor for gradient averaging in EMI-FGSM. (Default: 0.1)")
    parser.add_argument('--nu', type=float, default=0.7, 
                        help="Balance factor between gradient and momentum in QHM-FGSM. (Default: 0.7)")
    parser.add_argument('--m', type=int, default=5, 
                        help="Amortization length for AMI-FGSM. (Default: 5)")
    parser.add_argument('--amsgrad', action='store_true',
                        help="Whether to use AMSGrad variant of Adam in AI-FGSM.")
    parser.add_argument('--lr_schedule', action='store_true',
                        help="Whether to use learning rate scheduling in AI-FGSM.")
    parser.add_argument('--lr_decay_steps', type=int, default=3,
                        help="Number of steps for each learning rate decay in AI-FGSM. (Default: 3)")
    parser.add_argument('--clipnorm', type=float, default=None,
                        help="Gradient norm clipping value for APCI-FGSM.")
    parser.add_argument('--clipvalue', type=float, default=None,
                        help="Gradient value clipping for APCI-FGSM.")
    parser.add_argument('--global_clipnorm', type=float, default=None,
                        help="Global gradient norm clipping value for APCI-FGSM.")
    parser.add_argument('--delta', type=float, default=0.1,
                        help="Delta parameter for API-FGSM and SGDP-FGSM. (Default: 0.1)")
    parser.add_argument('--wd_ratio', type=float, default=0.1,
                        help="Weight decay ratio for API-FGSM and SGDP-FGSM. (Default: 0.1)")
    parser.add_argument('--nesterov', action='store_true',
                        help="Whether to use Nesterov momentum for momentum-based attacks.")
    parser.add_argument('--dampening', type=float, default=0.0, 
                        help="Dampening for SGDP-FGSM. (Default: 0.0)")
    parser.add_argument('--preconditioning_freq', type=int, default=10,
                        help="Frequency of preconditioning for SOAP-FGSM. (Default: 10)")
    
    # Arguments for input transformation methods (DIM, TIM, SIM) to enhance transferability
    parser.add_argument('--use_dim', action='store_true', 
                        help='Use Diverse Input Method')
    parser.add_argument('--use_tim', action='store_true', 
                        help='Use Translation-Invariant Method')
    parser.add_argument('--use_sim', action='store_true', 
                        help='Use Scale-Invariant Method')
    parser.add_argument('--use_pim', action='store_true',
                        help='Use Patch-wise Iterative Method')
    parser.add_argument('--resize_rate', type=float, default=1.1, 
                        help='Resize rate for DIM')
    parser.add_argument('--diversity_prob', type=float, default=0.5, 
                        help='Probability for DIM')
    parser.add_argument('--kernel_size', type=int, default=5, 
                        help='Kernel size for TIM')
    parser.add_argument('--scale_factors', type=float, nargs='+', default=[1.0, 0.9, 0.8, 0.7, 0.6], 
                        help='Scale factors for SIM')
    parser.add_argument('--patch_size', type=int, default=16,
                        help='Patch size for PIM')

    args = parser.parse_args()
    main(args)