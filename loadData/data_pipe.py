import torch
import numpy as np 
import random
import spectral as spy
from loadData import data_reader
from loadData.split_data import sample_gt


# only active in this file
def set_deterministic(seed):
    # seed by default is None 
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 

def get_data(args):
    data, data_gt = data_reader.load_data(args.dataset_name, path_data=args.path_data)
    # data, data_gt = data_reader.load_data(args.dataset_name, path_data=args.path_data, type_data="Houston")
    # split_data.data_info(data_gt, class_num=np.max(data_gt))
    pad_width = args.patch_size // 2
    img = np.pad(data, pad_width=pad_width, mode="constant", constant_values=(0))        # 111104
    img = img[:, :, pad_width:img.shape[2]-pad_width]
    height, width, bands = img.shape
    halving_line = bands // 2
    print("halving_line", halving_line)

    img1 = img[:, :, :halving_line]
    img2 = img[:, :, halving_line:]
    img1, pca = data_reader.apply_PCA(img1, num_components=args.components)
    img2, pca = data_reader.apply_PCA(img2, num_components=args.components)

    gt = np.pad(data_gt, pad_width=pad_width, mode="constant", constant_values=(0))
    print(img.shape, img1.shape, img2.shape, gt.shape)

    train_gt, test_gt = sample_gt(gt, train_num=args.train_num, 
                                  train_ratio=args.train_ratio, mode=args.split_type)
    print(train_gt.shape, test_gt.shape)

    if args.show_gt:
        # data_reader.draw(data_gt, args.result_dir + "/" + args.dataset_name + "data_gt", save_img=True)
        # data_reader.draw(train_gt, args.result_dir + "/" + args.dataset_name + "train_gt", save_img=True)
        # data_reader.draw(test_gt, args.result_dir + "/" + args.dataset_name + "test_gt", save_img=True)
        spy.imshow(classes=data_gt)
        spy.imshow(classes=train_gt)
        spy.imshow(classes=test_gt)

    # obtain label
    train_label, test_label = [], []
    for i in range(pad_width, train_gt.shape[0]-pad_width):
        for j in range(pad_width, train_gt.shape[1]-pad_width):
            if train_gt[i][j]:
                train_label.append(train_gt[i][j])

    for i in range(pad_width, test_gt.shape[0]-pad_width):
        for j in range(pad_width, test_gt.shape[1]-pad_width):
            if test_gt[i][j]:
                test_label.append(test_gt[i][j])
    # len(test_label)
    
    if args.print_data_info:
        data_reader.data_info(train_gt, test_gt, start=args.data_info_start)

    return img1, img2, train_gt, test_gt, data_gt
















