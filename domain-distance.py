# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Adopted from Invariant Risk Minimization
# https://github.com/facebookresearch/InvariantRiskMinimization
# 

import os
from sem import DataModel
from models import *

import argparse
import torch
import numpy
import matplotlib.pyplot as plt
import tabulate as tbl
from scipy.stats import sem


def pretty(vector):
    vlist = vector.view(-1).tolist()
    return "[" + ", ".join("{:+.3f}".format(vi) for vi in vlist) + "]"


def errors(w, w_hat):
    w = w.view(-1)
    w_hat = w_hat.view(-1)

    i_causal = torch.where(w != 0)[0].view(-1)
    i_noncausal = torch.where(w == 0)[0].view(-1)

    if len(i_causal):
        error_causal = (w[i_causal] - w_hat[i_causal]).pow(2).mean()
        error_causal = error_causal.item()
    else:
        error_causal = 0

    if len(i_noncausal):
        error_noncausal = (w[i_noncausal] - w_hat[i_noncausal]).pow(2).mean()
        error_noncausal = error_noncausal.item()
    else:
        error_noncausal = 0

    return error_causal, error_noncausal


def run_experiment(args):
    if args["seed"] >= 0:
        torch.manual_seed(args["seed"])
        numpy.random.seed(args["seed"])
        torch.set_num_threads(1)

    if args["setup_sem"] == "chain":
        setup_str = "shift={}_hetero={}_scramble={}".format(
            args["shift"],
            args["setup_hetero"],
            args["setup_scramble"])
    elif args["setup_sem"] == "icp":
        setup_str = "sem_icp"
    else:
        raise NotImplementedError

    all_methods = {
        "ERM": EmpiricalRiskMinimizer,
        "IRM": InvariantRiskMinimization,
        "REX": REXv21,
    }

    if args["methods"] == "all":
        methods = all_methods
    else:
        methods = {m: all_methods[m] for m in args["methods"].split(',')}

    all_sems = []
    all_environments = []
    results = {}

    for key in all_methods.keys():
        results[key] = {
            "errs_causal": [],
            "errs_noncausal": []
        }

    for rep_i in range(args["n_reps"]):
        if args["setup_sem"] == "chain":
            sem = DataModel(args["dim"],
                            shift=args["shift"],
                            ones=args["setup_ones"],
                            scramble=args["setup_scramble"],
                            hetero=args["setup_hetero"],
                            confounder_on_x=args["setup_confounder_on_x"])

            env_list = [float(e) for e in args["env_list"].split(",")]
            environments = [sem(args["n_samples"], e) for e in env_list]
        else:
            raise NotImplementedError

        all_sems.append(sem)
        all_environments.append(environments)

    i = 1
    for sem, environments in zip(all_sems, all_environments):
        print("Repetition: " + str(i))
        i += 1

        sem_solution, sem_scramble = sem.solution()

        for method_name, method_constructor in methods.items():
            print("Running " + method_name + "...")
            method = method_constructor(environments, args)

            method_solution = sem_scramble @ method.solution()

            err_causal, err_noncausal = errors(sem_solution, method_solution)

            results[method_name]["errs_causal"].append(err_causal)
            results[method_name]["errs_noncausal"].append(err_noncausal)

    return results

def runSampleComplexityExp():
    """
    Experiment on all shifts to yield model estimation error vs number of samples.
    """
    print("Running sample complexity experiment...")

    # Config
    shifts = ["CS", "CF", "AC", "HB"]
    methods = ["ERM","IRM","REX"]

    # Create environments to test on
    var_list = range(1,11)
    markers = {"ERM": "P", "IRM": ".", "REX": "x"}

    # Create options for ones, hetero, scramble, confounder
    options = [[1,0,0,0]]

    # Loop over option configurations
    for o in range(len(options)):
        # Config string and debugging
        config_string = '-'.join([str(elem) for elem in options[o]]) 
        print("--- OPTION CONFIG: " + config_string)

        # First = mean, second mean of causal, third mean of noncausal
        res = {"AVG": {}, "CAU": {}, "NON": {}}

        # Create dictonaries for the shifts
        for t in res.keys():
            for m in methods:
                res[t][m] = {"CS":[[],[]], "CF":[[],[]], "AC":[[],[]], "HB": [[],[]]}

        # Loop over shifts
        for s in shifts:
            print("--  SHIFT: " + s)
            env_labels = []

            # For each variance
            for v in var_list:
                print("- TRAIN: " + str(0.2*v))
                # Run for current Shift, Samples and Environment
                settings = getSettings(env_list=",".join([str(0.2), str(float(0.2*v)), str(5.)]), n_reps=5, n_iterations=100, shift=s, methods=",".join(methods), setup_ones=options[o][0], setup_hetero=options[o][1], setup_scramble=options[o][2], setup_confounder_on_x=options[o][3])
                results = run_experiment(settings)
                env_labels.append(0.2*v)

                # Add all methods to results
                for m in methods:
                    res["AVG"][m][s][0].append(numpy.mean(results[m]["errs_causal"] + results[m]["errs_noncausal"]))
                    res["AVG"][m][s][1].append(sem(results[m]["errs_causal"] + results[m]["errs_noncausal"]))
                    res["CAU"][m][s][0].append(numpy.mean(results[m]["errs_causal"]))
                    res["CAU"][m][s][1].append(sem(results[m]["errs_causal"]))
                    res["NON"][m][s][0].append(numpy.mean(results[m]["errs_noncausal"]))
                    res["NON"][m][s][1].append(sem(results[m]["errs_noncausal"]))

            # Loop over avg, cau and non
            for t in res.keys():
                # Path to file
                path = "./results/domain-distance/" + config_string + "/" + t + "/"

                # Create plot
                for m in methods:
                    plt.plot(env_labels, res[t][m][s][0], label=m, marker=markers[m])

                # Plot information
                plt.title(s + "-regression")
                plt.xlabel("Train domain value")
                plt.ylabel("Model estimation error")
                plt.legend(loc="best")
                os.makedirs(path + "figures/", exist_ok=True)
                plt.savefig(path + "figures/" + s + "-regression.png")
                plt.clf()

                # Create table
                rows = []
                for i in range(len(env_labels)):
                    for m in methods:
                        rows.append([m, env_labels[i], res[t][m][s][0][i], res[t][m][s][1][i]])
                table = tbl.tabulate(rows, headers=["Method", "Domain distance", "Model estimation error", "Standard error"])

                # Write table to file
                os.makedirs(path + "txt/", exist_ok=True)
                f = open(path + "txt/" + s + "-regression.txt", "a")
                f.write(table + "\n\n")
                f.close()

def getSettings(dim=10, n_samples=1000, n_reps=10, skip_reps=0, seed=0, print_vectors=1,
            n_iterations=100000, lr=0.001, verbose=0, methods="ERM,ICP,IRM", alpha=0.05,
            env_list=".2,2.,5.", setup_sem="chain", setup_ones=1, setup_hetero=1, setup_scramble=0, setup_confounder_on_x=0, shift="AC"):
    settings = {
        'dim': dim,
        'n_samples': n_samples,
        'n_reps': n_reps,
        'skip_reps': skip_reps,
        'seed': seed,
        'print_vectors': print_vectors,
        'n_iterations': n_iterations,
        'lr': lr,
        'verbose': verbose,
        'methods': methods,
        'alpha': alpha,
        'env_list': env_list,
        'setup_sem': setup_sem,
        'setup_ones': setup_ones,
        'setup_hetero': setup_hetero,
        'setup_scramble': setup_scramble,
        'setup_confounder_on_x': setup_confounder_on_x,
        'shift': shift
    }

    return settings

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Invariant regression')
    parser.add_argument('--dim', type=int, default=10)
    parser.add_argument('--n_samples', type=int, default=50)
    parser.add_argument('--n_reps', type=int, default=2)
    parser.add_argument('--skip_reps', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)  # Negative is random
    parser.add_argument('--print_vectors', type=int, default=1)
    parser.add_argument('--n_iterations', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--verbose', type=int, default=0)
    parser.add_argument('--methods', type=str, default="ERM,IRM")
    parser.add_argument('--alpha', type=float, default=0.05)
    parser.add_argument('--env_list', type=str, default=".2,2.,5.")
    parser.add_argument('--setup_sem', type=str, default="chain")
    parser.add_argument('--setup_ones', type=int, default=1)
    parser.add_argument('--setup_hetero', type=int, default=1)
    parser.add_argument('--setup_scramble', type=int, default=0)
    parser.add_argument('--setup_confounder_on_x', type=int, default=0)
    parser.add_argument('--shift', type=str, default="HB")
    args = dict(vars(parser.parse_args()))

    runSampleComplexityExp()

