import matplotlib.pyplot as plt
import tabulate as tbl
import os
import numpy as np

types = ["AVG", "CAU", "NON"]
shifts = ["AC", "CF", "CS", "HB"]
methods = ["ERM", "IRM", "REX"]
markers = {"ERM": "P", "IRM": ".", "REX": "x"}
experiments = {"domain-distance": "Domain distance", "quantity-of-training": "Number of training domains", "sample-complexity": "Number of samples"}
results = {}

plt.rc('font', size=17)
plt.tight_layout()
plt.clf()

for experiment in experiments.keys():
    for shift in shifts:
        print("- SHIFT: " + shift)
        results[shift] = {}
        path = './paper-causality/' + experiment + '/'
        for type in types:
            print("-- TYPE: " + type)
            results[shift][type] = {}
            for method in methods:
                results[shift][type][method] = [[],[],[]]
            cc = './results/' + experiment + '/1-0-0-0/' + type + '/txt/' + shift + "-regression.txt"
            with open(cc) as data:
                lines = data.readlines()[2:]
                for line in lines:
                    ls = line.split()
                    if line == "\n":
                        print("No more lines to read")
                    else:
                        method, tested, error, deviation = str(ls[0]), float(ls[1]), float(ls[2]), float(ls[3])
                        results[shift][type][method][0].append(error)
                        results[shift][type][method][1].append(deviation)
                        results[shift][type][method][2].append(tested)
        for method in methods:
            avg = np.array(results[shift]["AVG"][method][0])
            cau = np.array(results[shift]["CAU"][method][0])
            non = np.array(results[shift]["NON"][method][0])
            tested = np.array(results[shift][type][method][2])
            plt.plot(tested, avg, label=method, marker=markers[method])
            plt.fill_between(tested, cau, non, alpha=0.2)
        plt.margins(0)
        plt.title(shift + "-regression")
        plt.xlabel(experiments[experiment])
        plt.ylabel("Model estimation error")
        plt.legend(loc="best")
        os.makedirs(path + "/", exist_ok=True)
        plt.savefig(path + "/" + shift + "-regression.png", bbox_inches='tight', pad_inches=0)
        plt.clf()
    
    
