import argparse
import tensorflow as tf
from defense_attack_utils import load_model, load_dataset, get_input_size, resize_batch
from ours import pmai_fgsm, apci_fgsm
from hybrid_methods import hybrid_pmai_fgsm, hybrid_apci_fgsm
from third_party.Bit_Red.bit_red import suppress_tensorflow_warnings

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

def get_attack_function(attack_name, args):
    return {
        'pmai_fgsm': lambda x, y: pmai_fgsm(
            substitute_model, x, y, args.eps, args.steps, args.decay, args.m
        ),
        'apci_fgsm': lambda x, y: apci_fgsm(
            substitute_model, x, y, args.eps, args.steps, args.beta1, args.beta2, 
            args.learning_rate, args.weight_decay, args.epsilon, args.amsgrad,
            args.clipnorm, args.clipvalue, args.global_clipnorm
        ),
        'hybrid_pmai_fgsm': lambda x, y: hybrid_pmai_fgsm(
            substitute_model, x, y, args.eps, args.steps, args.decay, args.m,
            use_dim=args.use_dim, use_tim=args.use_tim, use_sim=args.use_sim, 
            use_pim=args.use_pim, resize_rate=args.resize_rate,
            diversity_prob=args.diversity_prob, kernel_size=args.kernel_size,
            scale_factors=args.scale_factors, patch_size=args.patch_size
        ),
        'hybrid_apci_fgsm': lambda x, y: hybrid_apci_fgsm(
            substitute_model, x, y, args.eps, args.steps, args.beta1, args.beta2,
            args.learning_rate, args.weight_decay, args.epsilon, args.amsgrad,
            args.clipnorm, args.clipvalue, args.global_clipnorm,
            use_dim=args.use_dim, use_tim=args.use_tim, use_sim=args.use_sim, 
            use_pim=args.use_pim, resize_rate=args.resize_rate,
            diversity_prob=args.diversity_prob, kernel_size=args.kernel_size,
            scale_factors=args.scale_factors, patch_size=args.patch_size
        )
    }.get(attack_name)

def main(args):
    suppress_tensorflow_warnings()
    
    global substitute_model
    substitute_model = load_model(args.substitute_model, verify=True)
    target_model = load_model(args.target_model, verify=True)
    
    substitute_input_size = get_input_size(args.substitute_model)
    target_input_size = get_input_size(args.target_model)
    
    dataset = load_dataset(
        args.data_dir, 
        args.batch_size, 
        image_size=substitute_input_size,
        model_name=args.target_model
    )
    
    lr_schedule = (lambda t: args.learning_rate * (0.1 ** (t // args.lr_decay_steps))) if args.lr_schedule else None
    
    attack_fn = get_attack_function(args.attack, args)
    if attack_fn is None:
        raise ValueError(f"Unknown attack: {args.attack}")
    
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
        
        successful_attacks += tf.reduce_sum(tf.cast(
            tf.argmax(y_adv_target, axis=1) != tf.argmax(y_pred_target, axis=1), 
            tf.float32))
        total_images += batch.shape[0]
    
    print(f"Attack success rate: {(successful_attacks / total_images):.2%}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Attack defense models.')
    
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Path to the dataset directory')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for processing')

    parser.add_argument('--substitute_model', type=str, required=True,
                    choices=['Inc-v3'],
                    help='Model for generating adversarial examples')

    parser.add_argument('--target_model', type=str, required=True,
                    choices=['hgd', 'rs', 'r_p', 'bit_red',
                            'feature_distill', 'jpeg_defense', 'nips_r3',
                            'comdefend', 'purify'],
                    help='Target defense model to attack')
    
    parser.add_argument('--attack', type=str, required=True,
                       choices=['pmai_fgsm', 'apci_fgsm',
                               'hybrid_pmai_fgsm', 'hybrid_apci_fgsm'],
                       help='Attack method to use')
                       
    parser.add_argument('--eps', type=float, default=0.3,
                       help='Maximum perturbation allowed')
    parser.add_argument('--steps', type=int, default=10,
                       help='Number of iterations')
    
    parser.add_argument('--learning_rate', type=float, default=0.01,
                       help='Learning rate')
    parser.add_argument('--beta1', type=float, default=0.9,
                       help='Beta1 for Adam')
    parser.add_argument('--beta2', type=float, default=0.999,
                       help='Beta2 for Adam')
    parser.add_argument('--epsilon', type=float, default=1e-7,
                       help='Epsilon for optimization')
    parser.add_argument('--weight_decay', type=float, default=0.0,
                       help='Weight decay')
    parser.add_argument('--decay', type=float, default=1.0,
                       help='Momentum decay factor')
    parser.add_argument('--m', type=int, default=5,
                       help='Amortization length for AMI-FGSM')
    parser.add_argument('--amsgrad', action='store_true',
                       help='Use AMSGrad variant of Adam')

    parser.add_argument('--lr_schedule', action='store_true',
                       help='Use learning rate scheduling')
    parser.add_argument('--lr_decay_steps', type=int, default=3,
                       help='Steps for learning rate decay')
    
    parser.add_argument('--clipnorm', type=float, default=None,
                       help='Gradient norm clipping')
    parser.add_argument('--clipvalue', type=float, default=None,
                       help='Gradient value clipping')
    parser.add_argument('--global_clipnorm', type=float, default=None,
                       help='Global gradient norm clipping')
    
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
    parser.add_argument('--scale_factors', type=float, nargs='+',
                       default=[1.0, 0.9, 0.8, 0.7, 0.6],
                       help='Scale factors for SIM')
    parser.add_argument('--patch_size', type=int, default=16,
                       help='Patch size for PIM')

    args = parser.parse_args()
    main(args)