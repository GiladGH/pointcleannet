from __future__ import print_function

import argparse
import os
import random
import math
import shutil
import torch
import torch.nn.parallel
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
import matplotlib.pyplot as plt
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from dataset import PointcloudPatchDataset, RandomPointcloudPatchSampler, SequentialShapeRandomPointcloudPatchSampler
from pcpnet import ResMSPCPNet, ResPCPNet


def parse_arguments():
    parser = argparse.ArgumentParser()

    # naming / file handling
    parser.add_argument(
        '--name', type=str, default='PointCleanNet', help='training run name')
    parser.add_argument(
        '--desc', type=str, default='My training run for PointCleanNet noise removal', help='description')
    parser.add_argument('--indir', type=str, default='../data/pointCleanNetDataset',
                        help='input folder (point clouds)')
    parser.add_argument('--outdir', type=str, default='../models',
                        help='output folder (trained models)')
    parser.add_argument('--logdir', type=str,
                        default='./logs', help='training log folder')
    parser.add_argument('--trainset', type=str,
                        default='trainingset.txt', help='training set file name')
    parser.add_argument('--testset', type=str,
                        default='validationset.txt', help='test set file name')
    parser.add_argument('--saveinterval', type=int,
                        default='5', help='save model each n epochs')
    parser.add_argument('--refine', type=str, default='',
                        help='refine model at this path')

    # training parameters
    parser.add_argument('--nepoch', type=int, default=1000,
                        help='number of epochs to train for')
    parser.add_argument('--batchSize', type=int,
                        default=48, help='input batch size')
    parser.add_argument('--patch_radius', type=float, default=[
                        0.05], nargs='+', help='patch radius in multiples of the shape\'s bounding box diagonal, multiple values for multi-scale.')
    parser.add_argument('--patch_center', type=str, default='point', help='center patch at...\n'
                        'point: center point\n'
                        'mean: patch mean')
    parser.add_argument('--patch_point_count_std', type=float, default=0,
                        help='standard deviation of the number of points in a patch')
    parser.add_argument('--patches_per_shape', type=int, default=500,# 800,
                        help='number of patches sampled from each shape in an epoch')
    parser.add_argument('--workers', type=int, default=1,
                        help='number of data loading workers - 0 means same thread as main execution')
    parser.add_argument('--cache_capacity', type=int, default=700,
                        help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')
    parser.add_argument('--seed', type=int,
                        default=3627473, help='manual seed')
    parser.add_argument('--training_order', type=str, default='random', help='order in which the training patches are presented:\n'
                        'random: fully random over the entire dataset (the set of all patches is permuted)\n'
                        'random_shape_consecutive: random over the entire dataset, but patches of a shape remain consecutive (shapes and patches inside a shape are permuted)')
    parser.add_argument('--identical_epochs', type=int, default=False,
                        help='use same patches in each epoch, mainly for debugging')
    parser.add_argument('--lr', type=float, default=0.00000001,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='gradient descent momentum')
    parser.add_argument('--use_pca', type=int, default=False,
                        help='Give both inputs and ground truth in local PCA coordinate frame')
    # model hyperparameters
    parser.add_argument('--outputs', type=str, nargs='+', default=['clean_points'], help='output of the network')
    parser.add_argument('--use_point_stn', type=int,
                        default=False, help='use point spatial transformer')
    parser.add_argument('--use_feat_stn', type=int,
                        default=False, help='use feature spatial transformer')
    parser.add_argument('--sym_op', type=str, default='max',
                        help='symmetry operation')
    parser.add_argument('--point_tuple', type=int, default=1,
                        help='use n-tuples of points as input instead of single points')
    parser.add_argument('--points_per_patch', type=int,
                        default=50, help='max. number of points per patch')# 50

    return parser.parse_args()


def check_path_existance(log_dirname, model_filename, opt):
    if os.path.exists(log_dirname) or os.path.exists(model_filename):
        if os.path.exists(log_dirname):
            shutil.rmtree(os.path.join(opt.logdir, opt.name))

def get_output_format(opt):
    # get indices in targets and predictions corresponding to each output
    target_features = []
    output_target_ind = []
    output_pred_ind = []
    pred_dim = 0
    for o in opt.outputs:
        if o in ['clean_points']:
            target_features.append(o)
            output_target_ind.append(target_features.index(o))
            output_pred_ind.append(pred_dim)
            pred_dim += 3

        else:
            raise ValueError('Unknown output: %s' % (o))
    if pred_dim <= 0:
        raise ValueError('Prediction is empty for the given outputs.')
    return target_features, output_target_ind, output_pred_ind,  pred_dim


def get_data(target_features, opt, train=True):
    # create train and test dataset loaders
    if train:
        shapes_list_file = opt.trainset
    else:
        shapes_list_file = opt.testset

    dataset = PointcloudPatchDataset(
        root=opt.indir,
        shapes_list_file=shapes_list_file,
        patch_radius=opt.patch_radius,
        points_per_patch=opt.points_per_patch,
        patch_features=target_features,
        point_count_std=opt.patch_point_count_std,
        seed=opt.seed,
        identical_epochs=opt.identical_epochs,
        use_pca=opt.use_pca,
        center=opt.patch_center,
        point_tuple=opt.point_tuple,
        cache_capacity=opt.cache_capacity)
    if opt.training_order == 'random':
        datasampler = RandomPointcloudPatchSampler(
            dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    elif opt.training_order == 'random_shape_consecutive':
        datasampler = SequentialShapeRandomPointcloudPatchSampler(
            dataset,
            patches_per_shape=opt.patches_per_shape,
            seed=opt.seed,
            identical_epochs=opt.identical_epochs)
    else:
        raise ValueError('Unknown training order: %s' % (opt.training_order))
    dataloader = torch.utils.data.DataLoader(
        dataset,
        sampler=datasampler,
        batch_size=opt.batchSize,
        num_workers=int(opt.workers))

    return dataloader, datasampler, dataset


def create_model(n_predicted_features, opt):
    # create model
    if len(opt.patch_radius) == 1:
        pcpnet = ResPCPNet(
            num_points=opt.points_per_patch,
            output_dim=n_predicted_features,
            use_point_stn=opt.use_point_stn,
            use_feat_stn=opt.use_feat_stn,
            sym_op=opt.sym_op,
            point_tuple=opt.point_tuple)
    else:
        pcpnet = ResMSPCPNet(
            num_scales=len(opt.patch_radius),
            num_points=opt.points_per_patch,
            output_dim=n_predicted_features,
            use_point_stn=opt.use_point_stn,
            use_feat_stn=opt.use_feat_stn,
            sym_op=opt.sym_op,
            point_tuple=opt.point_tuple)
    return pcpnet


def train_pcpnet(opt):
    # colored console output
    def green(x): return '\033[92m' + x + '\033[0m'
    def blue(x): return '\033[94m' + x + '\033[0m'

    log_dirname = os.path.join(opt.logdir, opt.name)
    params_filename = os.path.join(opt.outdir, '%s_params.pth' % (opt.name))
    model_filename = os.path.join(opt.outdir, '%s_model.pth' % (opt.name))
    desc_filename = os.path.join(opt.outdir, '%s_description.txt' % (opt.name))

    check_path_existance(log_dirname, model_filename, opt)
    target_features, output_target_ind, output_pred_ind,  n_predicted_features = get_output_format(opt)
    pcpnet = create_model(n_predicted_features, opt)
    if opt.refine != '':
        pcpnet.load_state_dict(torch.load(opt.refine))

    if opt.seed < 0:
        opt.seed = random.randint(1, 10000)
        print("Random Seed: %d" % (opt.seed))
    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    # create train and test dataset loaders
    train_dataloader, train_datasampler, train_dataset = get_data(target_features, opt, train=True)
    test_dataloader, test_datasampler, test_dataset = get_data(target_features, opt, train=False)
    # keep the exact training shape names for later reference
    opt.train_shapes = train_dataset.shape_names
    opt.test_shapes = test_dataset.shape_names

    print('training set: %d patches (in %d batches) - test set: %d patches (in %d batches)' %
          (len(train_datasampler), len(train_dataloader), len(test_datasampler), len(test_dataloader)))

    try:
        os.makedirs(opt.outdir)
    except OSError:
        pass

    train_writer = SummaryWriter(os.path.join(log_dirname, 'train'))
    test_writer = SummaryWriter(os.path.join(log_dirname, 'test'))

    optimizer = optim.SGD(pcpnet.parameters(), lr=opt.lr,
                        momentum=opt.momentum)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=0.1)
    pcpnet.cuda()

    total_train_batches = len(train_dataloader)
    total_test_batches = len(test_dataloader)

    # save parameters
    torch.save(opt, params_filename)

    # save description
    with open(desc_filename, 'w+') as text_file:
        print(opt.desc, file=text_file)

    train_loss_vec = []
    test_loss_vec = []
    for epoch in range(opt.nepoch):
        epoch_train_loss = 0
        epoch_test_loss = 0

        current_train_batch_index = -1
        train_completion = 0.0
        train_batches = enumerate(train_dataloader, 0)

        current_test_batch_index = -1
        test_completion = 0.0
        test_batches = enumerate(test_dataloader, 0)
        for current_train_batch_index, data in train_batches:
            # update learning rate
            scheduler.step(epoch * total_train_batches + current_train_batch_index)
            # set to training mode
            pcpnet.train()
            # prepare noisy points batch
            points = data[0]
            points = points.transpose(2, 1)
            points = points.cuda()
            # prepare ground truth points batch
            target = data[1:-1]
            target = tuple(t for t in target)
            target = tuple(t.cuda() for t in target)
            # zero gradients
            optimizer.zero_grad()

            # forward pass
            pred, trans, _, _ = pcpnet(points)
            loss = compute_loss(pred=pred,
                                target=target,
                                outputs=opt.outputs,
                                output_pred_ind=output_pred_ind,
                                output_target_ind=output_target_ind,
                                patch_rot=trans if opt.use_point_stn else None)
            # backpropagate through entire network to compute gradients of loss w.r.t. parameters
            loss.backward()
            # parameter optimization step
            optimizer.step()

            train_completion = (current_train_batch_index + 1) / total_train_batches

            # print info and update log file
            if current_train_batch_index % 500 == 0:
                print('[%s %d/%d: %d/%d] %s loss: %f' % (opt.name, epoch, opt.nepoch, current_train_batch_index,
                                                  total_train_batches - 1, green('train'), loss.item()))
            # print('min normal len: %f' % (pred.data.norm(2,1).min()))
            train_writer.add_scalar('loss', loss.item(),
                                    (epoch + train_completion) * total_train_batches * opt.batchSize)

            epoch_train_loss += loss.item()

            while test_completion <= train_completion and current_test_batch_index + 1 < total_test_batches:

                # set to evaluation mode
                pcpnet.eval()
                current_test_batch_index, data = next(test_batches)

                # get testset batch, convert to variables and upload to GPU
                # volatile means that autograd is turned off for everything that depends on the volatile variable
                # since we dont need autograd for inference (only for training)
                points = data[0]
                points = points.transpose(2, 1)
                points = points.cuda()
                target = data[1:-1]
                target = tuple(t for t in target)
                target = tuple(t.cuda() for t in target)

                # forward pass
                with torch.no_grad():
                    pred, trans, _, _ = pcpnet(points)
                    loss = compute_loss(
                        pred=pred, target=target,
                        outputs=opt.outputs,
                        output_pred_ind=output_pred_ind,
                        output_target_ind=output_target_ind,
                        patch_rot=trans if opt.use_point_stn else None)

                    test_completion = (current_test_batch_index + 1) / total_test_batches

                    if current_train_batch_index % 500 == 0:
                        print('[%s %d: %d/%d] %s loss: %f' % (opt.name, epoch,
                                                             current_train_batch_index, total_train_batches - 1, blue('test'), loss.item()))
                    test_writer.add_scalar(
                        'loss', loss.item(), (epoch + test_completion) *total_train_batches * opt.batchSize)
                epoch_test_loss += loss.item()

        train_loss_vec.append(epoch_train_loss/total_train_batches)
        test_loss_vec.append(epoch_test_loss/total_test_batches)

        # save model, overwriting the old model
        if epoch % opt.saveinterval == 0 or epoch == opt.nepoch - 1:
            torch.save(pcpnet.state_dict(), model_filename)

            plt.figure()
            plt.plot(range(epoch + 1), train_loss_vec, 'b')
            plt.plot(range(epoch + 1), test_loss_vec, 'r')
            plt.ylabel('Loss')
            plt.xlabel('epochs')
            plt.legend(['train loss', 'validation loss'])
            plt.savefig('loss2.png')

        # save model in a separate file in epochs 0,5,10,50,100,500,1000, ...
        if epoch % (5 * 10**math.floor(math.log10(max(2, epoch - 1)))) == 0 or epoch % 100 == 0 or epoch == opt.nepoch - 1:
            torch.save(pcpnet.state_dict(), os.path.join(
                opt.outdir, '%s_model_%d.pth' % (opt.name, epoch)))


def compute_surface_dist(prediction, target):
    # compute dist from target  prediction
    m2 = prediction.expand(target.shape[1], prediction.shape[0], 3).transpose(0, 1)
    m1 = target
    m = (m1 - m2).pow(2).sum(2)
    min_dist = torch.min(m, 1)[0]
    max_dist = torch.max(m, 1)[0]
    alpha = 0.99
    dist = torch.mean((alpha*min_dist) + (1-alpha)*max_dist)
    return dist*100

def compute_clean_point_loss(pred, output_pred_index, current_output_type_index,  target, output_target_index,
    patch_rot):
    o_pred = pred[:, output_pred_index[current_output_type_index]:output_pred_index[current_output_type_index] + 3]
    o_target = target[output_target_index[current_output_type_index]]
    if patch_rot is not None:
        # transform predictions with inverse transform
        # since we know the transform to be a rotation (QSTN), the transpose is the inverse
        o_pred = torch.bmm(o_pred.unsqueeze(1), patch_rot.transpose(2, 1)).squeeze(1)
    return compute_surface_dist(o_pred, o_target)


def compute_loss(pred, target, outputs, output_pred_ind, output_target_ind,  patch_rot):
    loss = 0
    for output_index, output in enumerate(outputs):
        if output in ['clean_points']:
            loss += compute_clean_point_loss(pred, output_pred_ind, output_index,  target, output_target_ind, patch_rot)
        else:
            raise ValueError('Unsupported output type: %s' % (output))
    return loss


if __name__ == '__main__':
    train_opt = parse_arguments()
    train_pcpnet(train_opt)
