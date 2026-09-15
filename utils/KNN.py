import os
import torch
import torch.nn.functional as F

import numpy as np
from sklearn import metrics
from . import kappa, util

# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_loader, test_loader, class_num, ground_truth, args, visulation=False):
    hight, width = ground_truth.shape
    net.eval()
    total_top1, total_top5, total_num = 0.0, 0.0, 0
    feature_bank = []
    feature_labels = []
    predict_labels = []
    targets = []

    with torch.no_grad():
        # generate feature bank
        # for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
        for data, _, target in memory_loader:
            target = target - 1
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=-1)
            feature_bank.append(feature)
            feature_labels.append(target.cuda(non_blocking=True))
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        feature_labels = torch.cat(feature_labels, dim=0)

        # print(feature_bank.shape, feature_labels.shape)
        # loop test data to predict the label by weighted knn search
        # for data, target in tqdm(test_data_loader):
        for data, _, target in test_loader:
            target = target - 1
            data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
            feature = net(data.cuda(non_blocking=True))
            feature = F.normalize(feature, dim=-1)
            total_num += data.size(0)
            # compute cos similarity between each feature vector and feature bank ---> [B, N]
            sim_matrix = torch.mm(feature, feature_bank)
            # [B, K]  select former k samples
            sim_weight, sim_indices = sim_matrix.topk(k=args.knn_k, dim=-1)
            # [B, K]
            sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
            sim_weight = (sim_weight / args.knn_t).exp()

            # counts for each class
            one_hot_label = torch.zeros(data.size(0) * args.knn_k, class_num, device=sim_labels.device)
            # [B*K, C]
            one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
            # weighted score ---> [B, C]
            pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, class_num) * sim_weight.unsqueeze(dim=-1), dim=1)

            # 预测的标签可能性降序排列
            pred_labels = pred_scores.argsort(dim=-1, descending=True)

            predict_labels.append(pred_labels[:, :1])
            targets.append(target)

        predict_labels = torch.cat(predict_labels, dim=0).cpu().numpy()
        # print(predict_labels.shape) # 有shuffle 这里会出错
        predict_labels = np.squeeze(predict_labels)
        targets = torch.cat(targets, dim=0).cpu().numpy()

        # print("len(targets)", len(targets), "predict_labels", len(predict_labels))
        # print(targets.shape, np.unique(targets), predict_labels.shape)
        matrix = metrics.confusion_matrix(targets, predict_labels)
        OA = np.sum(np.trace(matrix)) / float(len(targets)) * 100
        # print('OA = ', np.sum(np.trace(matrix)) / float(len(targets)) * 100)
        ka, AA, CA  = kappa.kappa(matrix, class_num)


        if visulation:
            predict_labels = predict_labels.reshape(hight, width) + 1

            # print(np.unique(predict_labels))
            util.draw(predict_labels, os.path.join(args.result_dir, str(round(OA, 4)) + "_knn_full"))
            # 背景像元置为 0，因为 pred 预测了所有的像元，但是背景像元并不需要画出来
            for i in range(hight):
                for j in range(width):
                    if ground_truth[i][j] == 0:
                        predict_labels[i][j] = 0

            util.draw(predict_labels, os.path.join(args.result_dir, str(round(OA, 4)) + "_knn_label"))
    return CA, OA, AA, ka
