import argparse
import numpy as np
import os
import torch.utils.data
from tensorboardX import SummaryWriter
from utils.dataset import load_mat_hsi, sample_gt, HSIDataset
from utils.utils_HSI import seed_worker
from models.get_model import get_model
from loss import make_loss
from train_pipeline import train
# Houston13-Houston18:lr-1e-3,depth-1,re_ratio-3,lambda1-5e-2,e > 500
if __name__ == "__main__":
    # fixed means for all models
    parser = argparse.ArgumentParser(description="Spectral Unmixing-guided Prompt Network for Cross-domain HSI classification")
    parser.add_argument("--model", type=str, default='SUPnet')
    # parser.add_argument('--source_name', type=str, default='Houston13',
    #                     help='the name of the source dir')
    # parser.add_argument('--target_name', type=str, default='Houston18',
    #                     help='the name of the test dir')
    # parser.add_argument("--dataset_dir", type=str, default='/mnt/storage/yzhang/data/Houston/')
    parser.add_argument('--source_name', type=str, default='NW-AreaI',
                        help='the name of the source dir')
    parser.add_argument('--target_name', type=str, default='NW-AreaII',
                        help='the name of the test dir')
    parser.add_argument("--dataset_dir", type=str, default='/mnt/storage/yzhang/data/Natural_Willow/')
    parser.add_argument("--device", type=str, default="1")
    parser.add_argument("--optim", type=str, default="Adam")
    parser.add_argument("--patch_size", type=int, default=13)
    parser.add_argument("--num_run", type=int, default=10) 
    parser.add_argument("--epoch", type=int, default=200)    
    parser.add_argument("--epoch_pgen", type=int, default=10)    
    parser.add_argument("--bs", type=int, default=128)  
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_cls", type=float, default=1e-3)
    parser.add_argument("--lr_prompt", type=float, default=1e-3)
    parser.add_argument("--lr_project", type=float, default=1e-3)
    parser.add_argument("--ratio", type=float, default=0.95)
    parser.add_argument("--num_domains", type=float, default=2)
    parser.add_argument("--dim", type=float, default=64)
    parser.add_argument("--z_dim", type=float, default=64)
    parser.add_argument("--num_em", type=float, default=7)
    parser.add_argument("--num_cls_prompt", type=float, default=4)
    parser.add_argument("--num_em_prompt", type=float, default=4)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--num_tokens", type=int, default=6)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lambda_1", type=float, default=1e-1)
    parser.add_argument("--lambda_2", type=float, default=1e+0)
    parser.add_argument("--loss_type", type=str,
                        default='softmax')
    parser.add_argument("--labelsmooth", type=str,
                        default='off') 
    parser.add_argument("--log_interval", type=float, default=10)
    parser.add_argument('--seed', type=int, default=233,
                        help='random seed ')
    parser.add_argument('--l2_decay', type=float, default=1e-4,
                        help='the L2  weight decay')
    parser.add_argument('--re_ratio', type=int, default=1,
                        help='random seed ')
    opts = parser.parse_args()

    device = torch.device("cuda:{}".format(opts.device))
    torch.cuda.device_count()
    # print parameters
    print("experiments will run on GPU device {}".format(opts.device))
    print("model = {}".format(opts.model))    
    print("source dataset = {}".format(opts.source_name))
    print("target dataset = {}".format(opts.target_name))
    print("dataset folder = {}".format(opts.dataset_dir))
    print("patch size = {}".format(opts.patch_size))
    print("batch size = {}".format(opts.bs))
    print("total epoch = {}".format(opts.epoch))

    # empty list to storing results
    results = []
    seeds = [311, 123, 2333, 202203, 2100, 202205, 1178, 3368, 4567, 1441]
    for run in range(opts.num_run):
        opts.seed = seeds[run]
        seed_worker(opts.seed) 
        print("running an experiment with the {} model".format(opts.model))
        print("run {} / {}".format(run+1, opts.num_run))

        if 'NW' in opts.source_name:
            opts.ratio = [0.0204359673024523, 0.0618769336541767, 0.0449662752935299, 0.161579892280072, 0.0361300682456845, 0.0198150594451783, 0.00547178988326848]
            img_src, gt_src, labels = load_mat_hsi(opts.source_name, opts.dataset_dir, norm='ori')
            img_tar, gt_tar, labels = load_mat_hsi(opts.target_name, opts.dataset_dir, norm='ori')
            train_gt_src, _ = sample_gt(gt_src, opts.ratio, opts.seed, mode='fixed')
            test_gt_tar, _ = sample_gt(gt_tar, 1, opts.seed, mode='random')
        else:
            img_src, gt_src, labels = load_mat_hsi(opts.source_name, opts.dataset_dir, norm='normband')
            img_tar, gt_tar, labels = load_mat_hsi(opts.target_name, opts.dataset_dir, norm='normband')
            train_gt_src, _ = sample_gt(gt_src, opts.ratio, opts.seed, mode='random')
            test_gt_tar, _ = sample_gt(gt_tar, 1, opts.seed, mode='random')

        num_classes = len(labels)
        num_bands = img_src.shape[-1]
        opts.num_bands = num_bands
        opts.num_classes = num_classes

        img_src_con, train_gt_src_con = img_src, train_gt_src
        for i in range(opts.re_ratio-1):
            img_src_con = np.concatenate((img_src_con,img_src))
            train_gt_src_con = np.concatenate((train_gt_src_con,train_gt_src))
        r = int(opts.patch_size / 2)+1

        img_src_con=np.pad(img_src_con,((r,r),(r,r),(0,0)),'symmetric')
        img_tar=np.pad(img_tar,((r,r),(r,r),(0,0)),'symmetric')
        train_gt_src_con=np.pad(train_gt_src_con,((r,r),(r,r)),'constant',constant_values=(-1,-1))
        test_gt_tar=np.pad(test_gt_tar,((r,r),(r,r)),'constant',constant_values=(-1,-1)) 
        
        train_set = HSIDataset(img_src_con, train_gt_src_con, patch_size=opts.patch_size, data_aug=True,
                               flip_augmentation=False, radiation_augmentation=False, mixture_augmentation=False)
        test_dataset = HSIDataset(img_tar, test_gt_tar, patch_size=opts.patch_size, data_aug=False)

        g = torch.Generator()
        g.manual_seed(opts.seed)
        train_loader = torch.utils.data.DataLoader(
            train_set, opts.bs, generator=g, drop_last=False, shuffle=False, num_workers=opts.num_workers)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, opts.bs, generator=g, drop_last=False, shuffle=False, num_workers=opts.num_workers)


        model = get_model(opts.model, opts.source_name, opts.patch_size, opts)
        model = model.to(device)
        criterion, center_criterion = make_loss(opts, num_classes=num_classes)
        
        model_dir = "./checkpoints/" + opts.model + '/' + opts.source_name + '_to_' + opts.target_name + 'train_woVCA_l20' + '/' + str(run+2)
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir, exist_ok=True)
        from datetime import datetime
        now_time = datetime.now()
        time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
        log_dir = os.path.join(model_dir, time_str)

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)   
        try:
            train(model, num_classes, train_loader, test_loader, opts, model_dir, device, writer)
        except KeyboardInterrupt:
            print('"ctrl+c" is pused, the training is over')

