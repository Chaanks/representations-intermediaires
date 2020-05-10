import numpy as np


def import_data(path):
    features = []
    targets = []
    with open(path) as file:
        for line in file:
            if line.strip() == '':
                continue

            line = line.rstrip()
            spl = line.split(',')[1:]
            targets.append(spl[0])

            buff = []
            for f in spl[1].split(' '):
                buff.append(float(f))
            features.append(np.asarray(buff))

    return np.asarray(features), np.asarray(targets)