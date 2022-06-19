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
        t_errors, t_div = {}, {}
        path = './paper-results/' + experiment + '/'
        for type in types:
            print("-- TYPE: " + type)
            results[shift][type] = {}
            t_errors[type] = {}
            t_div[type] = {}
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
                error = np.array(results[shift][type][method][0])
                dev = np.array(results[shift][type][method][1])
                tested = np.array(results[shift][type][method][2])
                plt.plot(tested, error, label=method, marker=markers[method])
                plt.fill_between(tested, error-dev, error+dev, alpha=0.2)
                t_errors[type][method] = error
                t_div[type][method] = dev
            plt.margins(0)
            plt.title(shift + "-regression")
            plt.xlabel(experiments[experiment])
            plt.ylabel("Model estimation error")
            plt.legend(loc="best")
            os.makedirs(path + "figures/" + type + "/", exist_ok=True)
            plt.savefig(path + "figures/" + type + "/" + shift + "-" + type + "-regression.png", bbox_inches='tight', pad_inches=0)
            plt.clf()
        
        pm = "$\pm$"
        rows = []
        if experiment == "domain-distance":
            tested = list(map(lambda x: x - 0.2, results[shift]["AVG"]["REX"][2]))
        for i in range(len(tested)):
            for m in methods:
                avg = '{0:.4f}'.format(round(t_errors["AVG"][m][i], 4)) + pm + '{0:.4f}'.format(round(t_div["AVG"][m][i], 4))
                cau = '{0:.4f}'.format(round(t_errors["CAU"][m][i], 4)) + pm + '{0:.4f}'.format(round(t_div["CAU"][m][i], 4))
                non = '{0:.4f}'.format(round(t_errors["NON"][m][i], 4)) + pm + '{0:.4f}'.format(round(t_div["NON"][m][i], 4))
                rows.append([m, tested[i], avg, cau, non])
        table = tbl.tabulate(rows, headers=["Method", experiments[experiment], "Average", "Causal", "Non-causal"], tablefmt='latex_raw')
        
        # Write table to file
        os.makedirs(path + "table/", exist_ok=True)
        f = open(path + "table/" + shift + "-regression.tex", "a")
        f.write(table)
        f.close()
    
