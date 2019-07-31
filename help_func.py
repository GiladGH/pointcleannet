import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def load_h5_data_label(h5_filename):
    """ load the data from the hdf5 files """
    f = h5py.File(h5_filename)
    data = f['data'][:]
    labels = f['label'][:]
    # normal = f['normal'][:]
    return (data, labels)


def load_dataset(train_files='train_files.txt'):
    with open('data/modelnet40_ply_hdf5_2048/train_files.txt') as f:
        files = [line.strip() for line in f.readlines()]

    h5_filename = os.path.join(os.getcwd(), files[0])
    data, labels = load_h5_data_label(h5_filename)

    for filename in files[1:]:
        h5_filename = os.path.join(os.getcwd(), filename)
        new_data, new_labels = load_h5_data_label(h5_filename)
        np.concatenate((data, new_data))
        np.concatenate((labels, new_labels))
    return data, labels

def pointnet_to_cleanpoint(data_dir, samples, labels):
    """ read pointnet dataset point cloud and transfer it to pcpnet dataset format """
    shape_names_dir = os.path.join(data_dir, 'shape_names.txt')

    with open(shape_names_dir) as f:
        label_names = f.readlines()

    # save in the pcp data set format
    new_dir = os.path.join(data_dir, '../modelNetDataset')
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)

    for i, _ in enumerate(samples, 0):
        sample = samples[i, :, :]

        # save clean sample
        num = 0
        filename = os.path.join(new_dir, label_names[labels[i][0]].strip() + '_{:d}_{:02d}.xyz'.format(labels[i][0], num))
        while os.path.exists(filename):
            num = num + 1
            filename = os.path.join(new_dir, label_names[labels[i][0]].strip() + '_{:d}_{:02d}.xyz'.format(labels[i][0], num))
        if num > 10:
            continue
        with open(filename, 'w') as f:
            f.write('\n'.join([' '.join(map(str, x)) for x in sample]))
            f.close()

        # save noisy sample - white noise std 0.25%
        filename_1 = os.path.splitext(filename)[0]
        noisy_sample = sample + np.random.normal(scale=0.0025, size=sample.shape)
        filename_1 = filename_1 + '_2.50e-03.xyz'
        with open(filename_1, 'w') as f:
            f.write('\n'.join([' '.join(map(str, x)) for x in noisy_sample]))
            f.close()

        # save noisy sample - white noise std 1%
        filename_2 = os.path.splitext(filename)[0]
        noisy_sample = sample + np.random.normal(scale=0.01, size=sample.shape)
        filename_2 = filename_2 + '_1.00e-02.xyz'
        with open(filename_2, 'w') as f:
            f.write('\n'.join([' '.join(map(str, x)) for x in noisy_sample]))
            f.close()

        # save noisy sample - white noise std 2.5%
        filename_3 = os.path.splitext(filename)[0]
        noisy_sample = sample + np.random.normal(scale=0.025, size=sample.shape)
        filename_3 = filename_3 + '_2.50e-02.xyz'
        with open(filename_3, 'w') as f:
            f.write('\n'.join([' '.join(map(str, x)) for x in noisy_sample]))
            f.close()

        # for each file create clean copy for GT
        filename = os.path.splitext(filename)[0] + '.clean_xyz'
        filename_1 = os.path.splitext(filename_1)[0] + '.clean_xyz'
        filename_2 = os.path.splitext(filename_2)[0] + '.clean_xyz'
        filename_3 = os.path.splitext(filename_3)[0] + '.clean_xyz'

        with open(filename, 'w') as f, open(filename_1, 'w') as f1, open(filename_2, 'w') as f2, open(filename_3, 'w') as f3:
            f.write('\n'.join([' '.join(map(str, x)) for x in sample]))
            f1.write('\n'.join([' '.join(map(str, x)) for x in sample]))
            f2.write('\n'.join([' '.join(map(str, x)) for x in sample]))
            f3.write('\n'.join([' '.join(map(str, x)) for x in sample]))
            f.close()
            f1.close()
            f2.close()
            f3.close()
    return


def visualize_point_cloud(pc, output_filename='null', fig_num=0, color=np.array([1])):
    """ points is a Nx3 numpy array """
    fig = plt.figure(fig_num)
    ax = fig.add_subplot(111, projection='3d')
    if color[0] == 1:
        color = pc[:, 2]

    ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2], c=color, s=5, marker='.', depthshade=True)
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z')
    ax.axis('on')

    # plt.savefig(output_filename)


def prep_training_set_file(data_dir):
    """ prep trainingset.txt and validationset.txt files """

    trainset_file = os.path.join(data_dir, 'trainingset.txt')
    valiset_file = os.path.join(data_dir, 'validationset.txt')

    with open(trainset_file, 'w') as f1, open(valiset_file, 'w') as f2:
        for path, subdirs, files in os.walk(data_dir):
            for file in files:
                file = os.path.splitext(file)[0]
                if file != 'trainingset' and file != 'validationset':
                    if int(file.split('_')[2]) <= 8:
                        f1.write(file + '\n')
                    else:
                        f2.write(file + '\n')
        f1.close()
        f2.close()


if __name__ == '__main__':
    clean = np.loadtxt('data/modelNetDataset/airplane_0_09_1.00e-02.clean_xyz')
    pc1 = np.loadtxt('results/airplane_0_09_1.00e-02_0.xyz')
    pc2 = np.loadtxt('results/airplane_0_09_1.00e-02_1.xyz')
    err1 = np.sum(np.square(pc1 - clean), axis=1)
    err2 = np.sum(np.square(pc2 - clean), axis=1)

    visualize_point_cloud(pc1, fig_num=1)
    visualize_point_cloud(pc2, fig_num=2)
    visualize_point_cloud(pc1, fig_num=3, color=err1)
    visualize_point_cloud(pc2, fig_num=4, color=err2)

    plt.show()

    # samples, labels = load_dataset()
    # in_dir = os.path.join(os.getcwd(), 'data/modelnet40_ply_hdf5_2048')
    # pointnet_to_cleanpoint(in_dir, samples, labels)
    # prep_training_set_file(os.path.join(os.getcwd(), 'data/modelNetDataset'))
