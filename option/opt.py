import torch
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Deep Multiview Learning for HSI classification')
    # general
    parser.add_argument('--backbone', default="vit", type=str, help='network backbone')
    parser.add_argument('--depth', default=4, type=int, help="vit depth")

    # datasets
    parser.add_argument('--dataset_name', default="India Pines", type=str, help='network backbone')
    # parser.add_argument('--path_data', type=str, default="/home/leo/DatasetSMD/")
    parser.add_argument('--path_data', type=str, default="/home/leo/DatasetSMD/")
    parser.add_argument('--print-data-info', action='store_true', default=False)
    parser.add_argument('--data_info_start', default=1, type=int)
    parser.add_argument('--remove_zero_labels', action='store_true', default=True)
    parser.add_argument('--patch_size', default=25, type=int, help='the number of patch size')
    parser.add_argument('--components', default=10, type=int, help='the number of train samples')      # 32 / 10
    parser.add_argument('--train_num', default=100, type=int, help='the number of train samples')
    parser.add_argument('--train_ratio', default=0.1, type=float, help='the ratio of train samples')
    parser.add_argument('--val_num', default=0, type=int, help='the number of validation samples')

    # Multi-crop parameters
    parser.add_argument('--randomCrop', type=int, default=23, help="Crop size")
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
                        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
                        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=2, help="""Number of small
                        local views to generate. Set this parameter to 0 to disable multi-crop training.
                        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                        help="""Scale range of the cropped image before resizing, relatively to the origin image.
                        Used for small local view cropping of multi-crop.""")

    # base 512, fdgc 128, vit 126, vit_gcn 64
    parser.add_argument('--feature_dim', default=126, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')

    # loss ratio
    parser.add_argument('--lambda_contra', default=0.1, type=int, help='Feature dim for latent vector')
    parser.add_argument('--lambda_super', default=1, type=float, help='Temperature used in softmax')

    # train config
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=30, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--tune_epochs', default=10, type=int, help='Number of sweeps over the dataset to train')

    parser.add_argument('--knn_k', default=5, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--knn-t', default=0.1, type=float, help='softmax temperature in kNN monitor; could be different with moco-t')

    # log
    parser.add_argument('--plot_loss_curve', action='store_true', default=False)
    parser.add_argument('--log_interval1', default=1, type=int)
    parser.add_argument('--log_interval2', default=1, type=int)
    parser.add_argument('--resume', default='' , type=str, help='continue training')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str, help='device')

    args = parser.parse_args('')  # running in ipynb
    return args
