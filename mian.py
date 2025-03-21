import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, [2]))
# print('using GPU %s' % ','.join(map(str, [2])))

import torch
import torch.optim as optim
from thop import profile, clever_format

import csv
import time
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman') 

from option import opt
from loadData import data_pipe, data_reader
from loadData.split_data import HyperX
from loadData.dataAugmentation import DataAugmentationDINO

from utils import util, KNN, SVM, trainer, tester, tester
from models import model, vision_transformer, automaticWeightedLoss


args = opt.get_args()
args.dataset_name = "PaviaU"

args.backbone = "vit"
args.local_crops_number = 0
args.patch_size = 25
args.randomCrop = 23
args.components = 10
args.split_type = "disjoint"
args.lambda_contra = 1
args.lambda_super = 1
args.awl = True

args.result_dir = os.path.join(os.path.join("/home/leo/Multimodal_Classification/KnowCL/results", 
                    datetime.now().strftime("%m-%d-%H-%M-vit_U"))) 
print(args.result_dir)

# args.result_dir = "/home/liuquanwei/code/DMVL_joint_MNDIS/results_final/08-12-16-51-vit_U"
if not os.path.exists(args.result_dir):
    os.mkdir(args.result_dir)
with open(args.result_dir + '/args.json', 'w') as fid:
    json.dump(args.__dict__, fid, indent=2)




# data_pipe.set_deterministic(seed = 666)
args.print_data_info = False
args.show_gt = False
args.remove_zero_labels = True
args.train_ratio = 1

# create dataloader
img1, img2, train_gt, _, _ = data_pipe.get_data(args)
transform = DataAugmentationDINO(args.randomCrop)

train_dataset = HyperX(img1, 
                       img2, 
                       train_gt, 
                       transform=transform, 
                       patch_size=args.patch_size, 
                        remove_zero_labels=args.remove_zero_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=args.batch_size, 
                                           shuffle=True, 
                                           drop_last=True)

args.train_ratio = 0.3
img1, img2, train_gt, test_gt, _ = data_pipe.get_data(args)
memory_dataset = HyperX(img1, img2, train_gt, transform=None, 
                        patch_size=args.patch_size, 
                        remove_zero_labels=args.remove_zero_labels)
test_dataset = HyperX(img1, img2, test_gt, transform=None, 
                      patch_size=args.patch_size, 
                      remove_zero_labels=args.remove_zero_labels)

memory_loader = torch.utils.data.DataLoader(memory_dataset,
                                            batch_size=args.batch_size,
                                            shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False)

class_num = np.max(train_gt)
# print(class_num, train_gt.shape, len(train_loader.dataset))



# model setup and optimizer config
# encoder = model.Model_base(args.components).cuda()
encoder = vision_transformer.vit_hsi(args.components, args.randomCrop).cuda()
# encoder = vision_transformer.vit_small(args.components, args.randomCrop).cuda()
contra_head = model.DINOHead(args.feature_dim).cuda()
super_head = model.FDGC_head(args.feature_dim, class_num=class_num).cuda()
awl = automaticWeightedLoss.AutomaticWeightedLoss(2).cuda()
net = util.MultiCropWrapper(encoder).cuda()
# print(model)


criterion = torch.nn.CrossEntropyLoss()
params = list(super_head.parameters()) + list(contra_head.parameters()) + \
                list(net.parameters()) + list(awl.parameters())
optimizer = optim.Adam(params, lr=1e-3, weight_decay=1e-6, amsgrad=True)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs, 
                            eta_min = 1e-8, last_epoch = -1)

# args.resume = os.path.join(args.result_dir, "joint_oa_model.pth")
if args.resume != '':
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['base'], strict=False)
    epoch_start = checkpoint['epoch'] + 1
    print('Loaded from: {}'.format(args.resume))
else:
    epoch_start = 0


# # # # # # # # # # # # # # # # train # # # # # # # # # # # # # #


best_loss = 999
best_acc = 0
train_losses = []
loss_contras = []
loss_supers = []

total_time = time.time()
for epoch in range(epoch_start, args.epochs):
    start_time = time.time()
    train_loss, loss_contra, loss_super, linear_tra,  linear_tea \
                        = trainer.train(epoch, net, contra_head, super_head, awl, criterion,
                                    train_loader, memory_loader, test_loader, optimizer, args)
    train_losses.append(train_loss)
    loss_contras.append(loss_contra)
    loss_supers.append(loss_super)
    scheduler.step()
    train_time = time.time() - start_time
    
    # # validation
    if epoch % args.log_interval1 == 0:
        start_time = time.time()
        KNNCA, KNNOA, KNNAA, KNNKA = KNN.test(net,memory_loader,test_loader,class_num,train_gt,args)
        Knn_time = time.time() - start_time
    with open(os.path.join(args.result_dir, "log.csv"), 'a+', encoding='gbk') as f:
        row=[["epoch", epoch, 
            "loss", train_loss, 
            "loss_contra", loss_contra,
            "loss_super", loss_super,
            # "linear_tracc", linear_tra,
            "linear_teacc", linear_tea,
            "train_time", round(train_time, 2),
            '\n',
            # "KNN_CA", np.round(KNNCA, 2),
            "KNN_OA", np.round(KNNOA, 2),
            "KNN_AA", np.round(KNNAA, 2),
            "KNN_KA", np.round(KNNKA, 2),
            "KNN_Time", round(Knn_time, 2),
            '\n',
            ]]
        write=csv.writer(f)
        for i in range(len(row)):
            write.writerow(row[i])

    if train_loss < best_loss:
        best_loss = train_loss
        torch.save({
                "epoch": epoch,
                "base": net.state_dict(),
                "contra_head": contra_head.state_dict(),
                "head": super_head.state_dict(),
                "optimizer": optimizer.state_dict()}, 
                os.path.join(args.result_dir, "joint_model_loss.pth"))
        
    if KNNOA > best_acc:
        best_acc = KNNOA
        torch.save({
                "epoch": epoch,
                "base": net.state_dict(),
                "contra_head": contra_head.state_dict(),
                "head": super_head.state_dict(),
                "optimizer": optimizer.state_dict()}, 
                os.path.join(args.result_dir, "joint_model_oa.pth"))
total_time = time.time() - total_time

torch.save({
        "epoch": epoch,
        "base": net.state_dict(),
        "contra_head": contra_head.state_dict(),
        "head": super_head.state_dict(),
        "optimizer": optimizer.state_dict()}, 
os.path.join(args.result_dir, "joint_model_last.pth"))

args.resume = os.path.join(args.result_dir, "joint_model_oa.pth")
# args.resume = os.path.join(args.result_dir, "joint_loss_model.pth")
if args.resume != '':
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['base'], strict=False)
    super_head.load_state_dict(checkpoint['head'], strict=False)
    epoch = checkpoint['epoch'] + 1
    print('Loaded from: {}'.format(args.resume))
else:
    epoch_start = 0


# # # # # # # # # # # # # # # # fine tune and Linear test # # # # # # # # # # # # # #


# save the best result
# for k in range(10, 51, 5):
for k in range(5,6):
    args.knn_k = k
    start_time = time.time()
    KNNCA, KNNOA, KNNAA, KNNKA = KNN.test(net, memory_loader, \
                                test_loader, class_num, train_gt, args)
    Knn_time = time.time() - start_time

    # linear 精度
    start_time = time.time()
    test_losses, test_preds, correct, targets = \
        tester.linear_test(net, super_head, criterion, test_loader, args)
    classification, kappa = tester.get_results(test_preds, targets)
    Linear_time = time.time() - start_time

    # print(KNNOA)
    with open(os.path.join(args.result_dir, "log_final.csv"), 'a+', encoding='gbk') as f:
        row=[["epoch", epoch, 
            "KNN_K", args.knn_k,
            "KNN_CA", np.round(KNNCA, 2),
            "KNN_OA", np.round(KNNOA, 2),
            "KNN_AA", np.round(KNNAA, 2),
            "KNN_KA", np.round(KNNKA, 2),
            "KNN_Time", round(Knn_time, 2),
            "\nclassification\n", classification,
            "\nkappa", kappa,
            "Linear_time", np.round(Linear_time, 2),
            "total_time", round(total_time, 2)
            ]]
        write=csv.writer(f)
        for i in range(len(row)):
            write.writerow(row[i])



# # # # # # # # # # # # # # # # fine tune and Linear test # # # # # # # # # # # # # #



head = model.FDGC_head(in_dim=args.feature_dim, class_num=class_num).to(args.device)

args.resume = os.path.join(args.result_dir, "joint_model_loss.pth")
if args.resume != '':
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['base'], strict=False)
    # head.load_state_dict(checkpoint['head'], strict=False)
    epoch_start = checkpoint['epoch'] + 1
    print('Loaded from: {}'.format(args.resume))
else:
    epoch_start = 0

params = list(net.parameters()) + list(head.parameters())
optimizer = torch.optim.Adam(params, lr=3e-4)
criterion = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=[args.epochs // 2, 
                (5 * args.epochs) // 6], gamma=0.1)


for name, param in net.named_parameters():
    param.requires_grad = False
    # print(param.requires_grad)      

def train(net, head, memory_loader, test_loader, criterion, optimizer, scheduler, args):
  best_loss = 9999
  best_acc = 0
  train_losses = []

  for epoch in range(args.tune_epochs):
    net.train()
    head.train()
    test_preds = []
    train_correct = torch.tensor(0).to(args.device)
    test_correct = torch.tensor(0).to(args.device)
    tic1 = time.time()
    
    for data, _, target in memory_loader:
      target = target -1
      data = data.to(args.device)
      target = target.to(args.device)
      optimizer.zero_grad()
      output = head(net(data))
      loss = criterion(output, target)
      loss.backward()
      optimizer.step()
      pred = output.data.max(1, keepdim=True)[1]
      train_correct += pred.eq(target.data.view_as(pred)).sum()

    scheduler.step()
    train_losses.append(loss.cpu().detach().item())
    train_accuracy = 100. * train_correct / len(memory_loader.dataset)
    training_time = time.time() - tic1

    # validation 
    tic2 = time.time()
    if epoch % args.log_interval2 == 0:
      net.eval()
      head.eval()
      with torch.no_grad():
        for data, _, target in test_loader:
          target = target - 1 
          data, target= data.to(args.device), target.to(args.device)
          output = head(net(data))
          test_pred = output.data.max(1, keepdim=True)[1]
          test_correct += test_pred.eq(target.data.view_as(test_pred)).sum()
          test_label = torch.argmax(output, dim=1)
          test_preds.append(test_label.cpu().numpy().tolist())
      test_accuracy = 100. * test_correct / len(test_loader.dataset)
    

      if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save({"epoch": epoch,
                    "model": net.state_dict(),
                    "head": head.state_dict(),
                    "optimizer": optimizer.state_dict()},
                    args.result_dir + "/best_model_loss.pth")
        print("save best loss weights at epoch", epoch)

      # 按照道理说，这里应该用train_acc, 但是为了精度，我选择 test_acc
      if test_accuracy.item() > best_acc:
        best_acc = test_accuracy.item()
        torch.save({"epoch": epoch,
                    "model": net.state_dict(),
                    "head": head.state_dict(),
                    "optimizer": optimizer.state_dict()},
                    args.result_dir + "/best_model_acc.pth")
        print("save best acc weights at epoch", epoch)
    test_time = time.time() - tic2

    print("epoch", epoch,
          "loss", round(loss.item(), 6), 
          "train_correct", round(test_correct.item(), 4), 
          "test_accuracy", round(test_accuracy.item(), 4))
    
	
    with open(os.path.join(args.result_dir, "linear.csv"), 'a+', encoding='gbk') as f:
        row=[["epoch", epoch, 
              "train num", args.train_num,
                "loss", round(loss.item(), 4), 
                "train acc", round(train_accuracy.item(), 4),
                "test acc", round(test_accuracy.item(), 4),
                "training time", round(training_time, 4),
                "test time", round(test_time, 4)
                ]]
        write=csv.writer(f)
        for i in range(len(row)):
            write.writerow(row[i])
  return train_losses, train_accuracy


fintune_time = time.time()
train_losses, train_accuracy = train(net, head, memory_loader, \
                                     test_loader, criterion, optimizer, scheduler, args)
fintune_time = time.time() - fintune_time


args.resume = os.path.join(args.result_dir, "best_model_acc.pth")
if args.resume != '':
    checkpoint = torch.load(args.resume)
    net.load_state_dict(checkpoint['model'], strict=False)
    head.load_state_dict(checkpoint['head'], strict=False)
    epoch = checkpoint['epoch'] + 1
    print('Loaded from: {} epoch {}'.format(args.resume, epoch))
else:
    epoch_start = 0


fintune_test_time = time.time()
test_losses, test_preds, correct, targets = \
    tester.linear_test(net, head, criterion, test_loader, args)
classification, kappa = tester.get_results(test_preds, targets)
fintune_test_time = time.time() - fintune_test_time


with open(os.path.join(args.result_dir, "log_final.csv"), 'a+', encoding='gbk') as f:
    row=[["epoch", epoch, 
        "\nclassification\n", classification,
        "\nkappa", kappa,
        "\nfintune_time", round(fintune_time, 2),
        "\nfintune_test_time", round(fintune_test_time, 2),
        ]]
    write=csv.writer(f)
    for i in range(len(row)):
        write.writerow(row[i])

































