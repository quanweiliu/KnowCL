import os
import torch
import torch.nn.functional as F

import numpy as  np
from sklearn import svm
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from . import kappa, util

# test for one epoch, use weighted knn to find the most similar images' label to assign the test image
def test(net, memory_loader, test_loader, class_num, ground_truth, args, visulation=False):
    hight, width = ground_truth.shape
    net.eval()
    memory_bank = []
    memory_labels = []
    test_bank = []
    test_labels = []

    with torch.no_grad():
        # generate feature bank
        # for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
        for data,_, target in memory_loader:
            target = target - 1 
            feature, out = net(data.cuda(non_blocking=True))
            feature, out, = F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
            memory_bank.append(feature)
            memory_labels.append(target.cuda(non_blocking=True))
        # [D, N]
        memory_bank = torch.cat(memory_bank, dim=0).contiguous()
        memory_labels = torch.cat(memory_labels, dim=0)

    with torch.no_grad():
        # generate feature bank
        # for data, target in tqdm(memory_data_loader, desc='Feature extracting'):
        for data,_, target in test_loader:
            target = target - 1 
            feature, out = net(data.cuda(non_blocking=True))
            feature, out, = F.normalize(feature, dim=-1), F.normalize(out, dim=-1)
            test_bank.append(feature)
            test_labels.append(target.cuda(non_blocking=True))
        # [D, N]
        test_bank = torch.cat(test_bank, dim=0).contiguous()
        test_labels = torch.cat(test_labels, dim=0)

    memory_bank = memory_bank.cpu().numpy()
    memory_labels = memory_labels.cpu().numpy()
    test_bank = test_bank.cpu().numpy()
    test_labels = test_labels.cpu().numpy()

    # (1260, 512) (1260,) (8989, 512) (8989,)
    # print(memory_bank.shape, memory_labels.shape, test_bank.shape, test_labels.shape)

    # start = time.time()
    C = np.logspace(-2, 8, 11, base=2)  # 2为底数，2的-2次方到2的8次方，一共11个数
    gamma = np.logspace(-2, 8, 11, base=2)

    parameters = {'C': C, 'gamma': gamma}
    clf = GridSearchCV(svm.SVC(kernel='rbf'), parameters, cv=3, refit=True)
    clf.fit(memory_bank, memory_labels)
    

    # start = time.time()
    predict_label = clf.predict(test_bank)  # (42776,)
    
    # (8989,) [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15] (8989,)
    # print(test_labels.shape, np.unique(test_labels), predict_label.shape)
    matrix = metrics.confusion_matrix(test_labels, predict_label)
    OA = np.sum(np.trace(matrix)) / float(len(test_labels)) * 100
    ka, AA, CA = kappa.kappa(matrix, class_num)
    # print('OA = ', OA)
    # print(epoch, "/", epochs)

    if visulation:
        predict_labels = predict_label.reshape(hight, width) + 1

        # print(np.unique(predict_labels))
        util.draw(predict_labels, os.path.join(args.result_dir, str(round(OA, 4)) + "_svm_full"))
        # 背景像元置为 0，因为 pred 预测了所有的像元，但是背景像元并不需要画出来
        for i in range(hight):
            for j in range(width):
                if ground_truth[i][j] == 0:
                    predict_labels[i][j] = 0

        util.draw(predict_labels, os.path.join(args.result_dir, str(round(OA, 4)) + "_svm_label"))
    return CA, OA, AA, ka
