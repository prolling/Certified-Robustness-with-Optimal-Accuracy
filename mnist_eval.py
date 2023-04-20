"""Run MNIST experiments for the paper.
"""
import time
import os
import math
from multiprocessing import Pool

import models.params
from datasets import cifar, mnist, svhn
import models
from models.train import train
from models.evaluate import evaluate
from models import pixeldp_cnn, pixeldp_resnet
import attacks
from attacks import pgd, params, train_attack, evaluate_attack
import plots.plot_robust_accuracy
import plots.plot_accuracy_under_attack

from experiments.experiment import train_eval_model, train_eval_attack

import tensorflow as tf
import numpy as np

from flags import FLAGS

import shutil

# def run(plots_only=False):


def run(plots_only=False):
    param_dict = {
        'name_prefix': '',
        'steps_num': 1000,  # changed from 40000 to run faster
        'eval_data_size': 10000,
        'image_size': 28,
        'n_channels': 1,
        'num_classes': 10,
        'relu_leakiness': 0.0,
        'lrn_rate': 0.1,
        'lrn_rte_changes': [30000],
        'lrn_rte_vals': [0.01],
        'num_residual_units': 4,
        'use_bottleneck': False,
        'weight_decay_rate': 0.0002,
        'optimizer': 'mom',
        'image_standardization': False,
        # 'dp_epsilon': 0.9,
        # 'dp_delta': 0.000000006,
        'robustness_confidence_proba': 0.05,
        'attack_norm': 'l2',
        'sensitivity_norm': 'l2',  # changed to l2 for Gaussian noise
        'sensitivity_control_scheme': 'bound',  # bound or optimize
        'layer_sensitivity_bounds': ['l2_l2'],
        'noise_after_activation': True,
        'parseval_loops': 10,
        'parseval_step': 0.0003,
    }
    epsilons = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
    deltas = [0.000171, 0.000043, 0.000011, 0.000003, 0.000001,
              0.0000002, 0.00000006, 0.00000002, 0.000000006]

    for epsilon in epsilons:
        for delta in deltas:
            if folder_exists(epsilon, delta):
                print(f"Skipping experiment for epsilon {epsilon} and delta {delta} as the folder already exists.")
                continue
        
            param_dict['dp_epsilon'] = epsilon
            param_dict['dp_delta'] = delta
            

            # Keep the rest of the existing code

            params = []
            num_gpus = max(1, FLAGS.num_gpus)
            parallelizable_arguments_list = []
        
            # Ls = [0.0, 0.03, 0.1, 0.3, 1.0]
            Ls = [0.0, 0.1, 0.3]
        
            # First, create all params for train/eval models.
            for model_name in ["pixeldp_cnn"]:
                for attack_norm_bound in Ls:
                    for noise_after_n_layers in [-1, 1]:
                        if attack_norm_bound == 0.0 and noise_after_n_layers > -1:
                            continue  # The baseline can only have -1.
                        if attack_norm_bound > 0.0 and noise_after_n_layers < 0:
                            # PixelDP nets need a noise layer at position >= 0.
                            continue
                        if attack_norm_bound == 0.0:
                            param_dict['parseval_loops'] = 0
                        else:
                            param_dict['parseval_loops'] = math.ceil(
                                100 * attack_norm_bound)
                        param_dict['attack_norm_bound'] = attack_norm_bound
                        param_dict['noise_after_n_layers'] = noise_after_n_layers
                        if not plots_only:
                            parallelizable_arguments_list.append(
                                (
                                    'mnist',
                                    model_name,
                                    dict(param_dict),
                                    len(parallelizable_arguments_list) % num_gpus
                                )
                            )
                        else:
                            param_dict = dict(param_dict)
                            param_dict['batch_size'] = 1
                            param_dict['n_draws'] = 1
                            hps = models.params.HParams(**param_dict)
                            parallelizable_arguments_list.append((hps, model_name))
        
            # Run train/eval of models.
            if not plots_only:
                print("\nTrain/Eval models:: Experiments: {}".
                      format(parallelizable_arguments_list))
                print("Train/Eval models:: Total experiments: {}".
                      format(len(parallelizable_arguments_list)))
                print("Train/Eval models:: Running on {} GPUs\n\n".format(num_gpus))
                results = []
                for i in range(0, len(parallelizable_arguments_list), num_gpus):
                    p = Pool(processes=num_gpus)
                    current = p.map(train_eval_model, parallelizable_arguments_list[i:min(
                        i+num_gpus, len(parallelizable_arguments_list))])
                    results.extend(current)
                    p.close()
                    p.join()
                    time.sleep(5)
            else:
                results = parallelizable_arguments_list
        
            # Second, create all params for train/eval attacks on models.
            parallelizable_arguments_list = []
            _attack_param_dict = {
                'restarts': 10,
                'n_draws_attack': 10,
                'n_draws_eval':   500,
                'attack_norm': 'l2',
                'max_attack_size': -1,
                'num_examples': 1000,
                'attack_methodolody': 'pgd',
                'targeted': False,
                'sgd_iterations': 100,
                'use_softmax': False,
            }
        
            use_attack_methodology = 'carlini'
            pgd_sizes = [round(x, 2) for x in np.arange(0.25, 4.0, 0.25).tolist()]
            for (hps, model_name) in results:
                if use_attack_methodology == 'pgd':
                    attack_param_dict = dict(_attack_param_dict)
                    for attack_size in pgd_sizes:
                        attack_size = round(attack_size, 2)
                        attack_param_dict['max_attack_size'] = attack_size
                        if not plots_only:
                            parallelizable_arguments_list.append(
                                (
                                    'mnist',
                                    hps,
                                    model_name,
                                    attack_size,
                                    len(parallelizable_arguments_list) % num_gpus
                                )
                            )
                        else:
                            attack_params = attacks.params.AttackParams(
                                **attack_param_dict)
                            parallelizable_arguments_list.append((
                                hps, attack_params, model_name
                            ))
        
                if use_attack_methodology == 'carlini':
                    attack_param_dict = dict(_attack_param_dict)
                    attack_param_dict['max_attack_size'] = 5
                    attack_param_dict['restarts'] = 1
                    attack_param_dict['n_draws_attack'] = 20
                    attack_param_dict['n_draws_eval'] = 500
                    attack_param_dict['attack_methodolody'] = "carlini"
                    if hps.attack_norm_bound == 0.0:
                        use_softmaxs = [False]
                    else:
                        use_softmaxs = [False, True]
                    for use_softmax in use_softmaxs:
                        attack_param_dict['use_softmax'] = use_softmax
                        if not plots_only:
                            parallelizable_arguments_list.append(
                                (
                                    'mnist',
                                    hps,
                                    model_name,
                                    dict(attack_param_dict),
                                    len(parallelizable_arguments_list) % num_gpus
                                )
                            )
                        else:
                            attack_params = attacks.params.AttackParams(
                                **attack_param_dict)
                            parallelizable_arguments_list.append((
                                hps, attack_params, model_name
                            ))
        
            if not plots_only:
                # Run train/eval of attracks on models.
                print("\nTrain/Eval attacks:: Experiments: {}".
                      format(parallelizable_arguments_list))
                print("Train/Eval attacks:: Total experiments: {}".
                      format(len(parallelizable_arguments_list)))
                print("Train/Eval attacks:: Running on {} GPUs\n\n".format(num_gpus))
                results = []
                for i in range(0, len(parallelizable_arguments_list), num_gpus):
                    p = Pool(processes=num_gpus)
                    current = p.map(train_eval_attack, parallelizable_arguments_list[i:min(
                        i+num_gpus, len(parallelizable_arguments_list))])
                    results.extend(current)
                    p.close()
                    p.join()
                    print("Finished experiments: {}/{}".
                          format(len(results), len(parallelizable_arguments_list)))
                    time.sleep(5)
            else:
                results = parallelizable_arguments_list
        
            _robust_model_names = set()
            _robust_models = []
            _robust_params = []
            _models_argmax = []
            _params_argmax = []
            _models_softmax = []
            _params_softmax = []
            nonbaseline_attack_params_softmax = []
            nonbaseline_attack_params_argmax = []
            baseline_attack_params = []
            for (hps, attack_params, model_name) in results:
                if hps.attack_norm_bound == 0.0 and model_name != 'madry':
                    baseline_model = models.module_from_name(model_name)
                    baseline_params = hps
                    baseline_attack_params.append(attack_params)
                else:
                    if model_name != 'madry':
                        model_module = models.module_from_name(model_name)
                        _name = models.params.name_from_params(model_module, hps)
                        if _name not in _robust_model_names:
                            _robust_model_names.add(_name)
                            _robust_models.append(model_module)
                            _robust_params.append(hps)
                    if hps.attack_norm_bound not \
                            in list(map(lambda x: x.attack_norm_bound, params)):
                        #  models.append(modelname2module(model_name))
                        if attack_params.use_softmax:
                            _models_softmax.append(models.module_from_name(model_name))
                            _params_softmax.append(hps)
                            nonbaseline_attack_params_softmax.append([])
                        else:
                            _models_argmax.append(models.module_from_name(model_name))
                            _params_argmax.append(hps)
                            nonbaseline_attack_params_argmax.append([])
                    if attack_params.use_softmax:
                        nonbaseline_attack_params_softmax[-1].append(attack_params)
                    else:
                        nonbaseline_attack_params_argmax[-1].append(attack_params)
                        
        
            # Plot robust accuracy results
            dir_name = os.path.join(FLAGS.models_dir, 'mnist')
            plots.plot_robust_accuracy.plot("mnist_robust_accuracy_argmax",
                                            baseline_model,
                                            baseline_params,
                                            _robust_models,
                                            _robust_params,
                                            x_range=(0, 0.6, 0.025),
                                            dir_name=dir_name)
            plots.plot_robust_accuracy.plot("mnist_robust_accuracy_softmax",
                                            baseline_model,
                                            baseline_params,
                                            _robust_models,
                                            _robust_params,
                                            x_range=(0, 0.6, 0.025),
                                            dir_name=dir_name,
                                            expectation_layer='softmax')
            # Plot accuracy under attack
            x_ticks = [x/10 for x in range(1, 31)]
            plots.plot_accuracy_under_attack.plot("mnist_accuracy_under_attack_argmax",
                                                  [baseline_model] + _models_argmax,
                                                  [baseline_params] + _params_argmax,
                                                  [baseline_attack_params] +
                                                  nonbaseline_attack_params_argmax,
                                                  x_range=(0, 3.0),
                                                  x_ticks=x_ticks,
                                                  dir_name=dir_name)
            plots.plot_accuracy_under_attack.plot("mnist_accuracy_under_attack_softmax",
                                                  [baseline_model] + _models_softmax,
                                                  [baseline_params] + _params_softmax,
                                                  [baseline_attack_params] +
                                                  nonbaseline_attack_params_softmax,
                                                  x_range=(0, 3.0),
                                                  x_ticks=x_ticks,
                                                  dir_name=dir_name,
                                                  expectation_layer='softmax')
                                                  
                                                  
            save_and_cleanup(epsilon, delta)
                                                  
                
        
import glob

# function to save the results for each epsilon and delta pair
def save_and_cleanup(epsilon, delta):
    # Create a directory for the current epsilon and delta pair
    results_dir = os.path.join(FLAGS.models_dir, f'mnist_epsilon_{epsilon}_delta_{delta}')
    os.makedirs(results_dir, exist_ok=True)

    # Define the mnist_models_dir
    mnist_models_dir = os.path.join(FLAGS.models_dir, 'mnist')

    # Move the generated text files and pdfs to the new directory
    for file in os.listdir(mnist_models_dir):
        if file.endswith(".txt") or file.endswith(".pdf"):
            src_path = os.path.join(mnist_models_dir, file)
            dst_path = os.path.join(results_dir, file)
            shutil.move(src_path, dst_path)

    # Delete everything inside the trained_models/mnist folder
    files_to_delete = glob.glob(os.path.join(mnist_models_dir, '*'))
    for f in files_to_delete:
        if os.path.isfile(f):
            os.remove(f)
        elif os.path.isdir(f):
            shutil.rmtree(f)
        
# function to check if the results for the epsilon and delta pair exists before running    
def folder_exists(epsilon, delta):
    folder_path = os.path.join(FLAGS.models_dir, f'mnist_epsilon_{epsilon}_delta_{delta}')
    return os.path.exists(folder_path)


        
def main(_):
    run()
    
    
    
