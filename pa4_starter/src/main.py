import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec
import random

sns.set()

import argparse

from utils import load_data
from decompose import PCA
from clustering import KMeans


def load_args():
    parser = argparse.ArgumentParser(description='arguments')
    parser.add_argument('--pca', default=0, type=int,
                        help='set to 1 if we desire running pca, otherwise 0')
    parser.add_argument('--kmeans', default=1, type=int,
                        help='set to 1 if we desire running kmeans, otherwise 0')

    parser.add_argument('--pca_retain_ratio', default=.9, type=float)
    parser.add_argument('--kmeans_max_k', default=15, type=int)
    parser.add_argument('--kmeans_max_iter', default=20, type=int)
    parser.add_argument('--root_dir', default='../data/', type=str)
    args = parser.parse_args()

    return args


def plot_y_vs_x_list(y_vs_x, x_label, y_label, save_path):
    fld = os.path.join(args.root_dir, save_path)
    if not os.path.exists((fld)):
        os.mkdir(fld)

    plots_per_fig = 2
    if(type(y_vs_x) != 'numpy.ndarray'):
        y_vs_x = [y_vs_x]

    ks_sses_keys = list(range(0, len(y_vs_x)))
    js = list(range(0, len(ks_sses_keys), plots_per_fig))

    for j in js:
        pp = ks_sses_keys[j:j + plots_per_fig]
        fig = plt.figure(constrained_layout=True)
        gs = gridspec.GridSpec(len(pp), 1, figure=fig)
        i = 0
        for k in pp:
            ax = fig.add_subplot(gs[i, :])
            ax.set_ylabel('%s (k=%d)' % (y_label, k))
            ax.set_xlabel(x_label)
            ax.plot(range(1, len(y_vs_x[k]) + 1), [x for x in y_vs_x[k]], linewidth=2)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            i += 1

        fig.savefig(os.path.join(fld, '%d_%d.png' % (pp[0], pp[-1])))

    print('Saved at : %s' % fld)

# Function to plot y values versus x values
def plot_y_vs_x(ys_vs_x, x_label, y_label, save_path):
    fld = os.path.join(args.root_dir, save_path)
    if not os.path.exists((fld)):
        os.mkdir(fld)

    fig = plt.figure(constrained_layout=True)
    gs = gridspec.GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0, :])
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.plot(range(1, len(ys_vs_x) + 1), ys_vs_x, linewidth=2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.savefig(os.path.join(fld, 'plot.png'))

    print('Saved at : %s' % fld)


def visualize(x_train, y_train):
    print(x_train)

    new_x_train = []
    for item in range(len(x_train)):
        new_x_train.append(-np.sort(-x_train[item]))
    component1 = []
    component2 = []
    for item in range(len(new_x_train)):
        component1.append(new_x_train[item][0])
        component2.append(new_x_train[item][1])

    plt.scatter(component1, component2, c=y_train)
    plt.xlabel("component1")
    plt.ylabel("component2")
    plt.savefig("visualize.png")
    plt.clf()
    
    print('Saved at : %s' % args.root_dir)

    ##################################
    #      YOUR CODE GOES HERE       #
    ##################################


def apply_kmeans(do_pca, x_train, y_train, kmeans_max_iter, kmeans_max_k):
    print('kmeans\n')
    train_sses_vs_iter = []
    train_sses_vs_k = []
    train_purities_vs_k = []

    for k in range(1, kmeans_max_k):
        print("kmeans loop: " + str(k))
        # Part 1.2 here
        inner_train_sses_vs_iter = []
        inner_train_sses_vs_k = []
        inner_train_purities_vs_k = []
        
        for i in range(5):
            kmeans = KMeans(k, kmeans_max_iter)
            sse_vs_iter = kmeans.fit(x_train)

            inner_train_sses_vs_iter.append(sse_vs_iter)
            inner_train_purities_vs_k.append(kmeans.get_purity(x_train, y_train))
            inner_train_sses_vs_k.append(min(sse_vs_iter))

            # train_sses_vs_iter.append(sse_vs_iter)
            # train_purities_vs_k.append(kmeans.get_purity(x_train, y_train))
            # train_sses_vs_k.append(min(sse_vs_iter))
            

        train_purities_vs_k = np.mean(inner_train_purities_vs_k)
        train_sses_vs_iter  = np.mean(inner_train_sses_vs_iter)
        train_sses_vs_k     = np.mean(inner_train_sses_vs_k)

    # plot_y_vs_x_list(train_sses_vs_iter, x_label='iter', y_label='sse',
                    # save_path='plot_sse_vs_k_subplots_%d'%do_pca)
    plot_y_vs_x(train_sses_vs_k, x_label='k', y_label='sse',
                save_path='plot_sse_vs_k_%d'%do_pca)
    plot_y_vs_x(train_purities_vs_k, x_label='k', y_label='purities',
                save_path='plot_purity_vs_k_%d'%do_pca)



if __name__ == '__main__':
    args = load_args()
    random.seed()
    x_train, y_train = load_data(args.root_dir)

    if args.pca == 1:
        pca = PCA(args.pca_retain_ratio)
        pca.fit(x_train)
        x_train = pca.transform(x_train)
        visualize(x_train, y_train)

    if args.kmeans == 1:
        apply_kmeans(args.pca, x_train, y_train, args.kmeans_max_iter, args.kmeans_max_k)

    print('Done')
