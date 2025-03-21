from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F

# train for one epoch to learn unique features
def train(epoch, net, contra_head, net_head, awl, criterion, data_loader, \
          memory_loader, test_loader, train_optimizer, args):
    net.train()
    contra_head.train()
    net_head.train()

    total_loss = 0
    total_num = 0
    train_correct = 0
    test_correct = 0
    train_bar = tqdm(enumerate(zip(data_loader, memory_loader)))
    # train_bar = tqdm(enumerate(data_loader))
    
    for step, ((pos_1, pos_2, target), (pos, pos_, label)) in train_bar:
        # torch.Size([128, 100, 28, 28]) torch.Size([128, 100, 28, 28]) torch.Size([128])
        # print(pos_1.shape, pos_2.shape, target.shape, label.shape) 
        pos_1 = [im.cuda(non_blocking=True) for im in pos_1]
        pos_2 = [im.cuda(non_blocking=True) for im in pos_2]

        out_1 = F.normalize(contra_head(net(pos_1)), dim=-1)
        out_1 = out_1.chunk(args.local_crops_number + 2)   # [256, 256]*10

        out_2 = F.normalize(contra_head(net(pos_2)), dim=-1)
        out_2 = out_2.chunk(args.local_crops_number + 2)   # [256, 256]*10


        ########## contra ####################################>>>>>>>>>>>>>>>
        total_contra_loss = 0
        n_loss_terms = 0
        for iq, view1 in enumerate(out_1):
            if iq >= 1:
                continue
            for v in range(iq, len(out_2)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                out = torch.cat([view1, out_2[v]], dim=0)
                sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / args.temperature)
                mask = (torch.ones_like(sim_matrix) - torch.eye(2 * args.batch_size, device=sim_matrix.device)).bool()
                sim_matrix = sim_matrix.masked_select(mask).view(2 * args.batch_size, -1)
                pos_sim = torch.exp(torch.sum(view1 * out_2[v], dim=-1) / args.temperature)
                pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
                total_contra_loss += (- torch.log(pos_sim / sim_matrix.sum(dim=-1))).mean()
                n_loss_terms += 1
        total_contra_loss /= n_loss_terms
        ########## contra ######################################<<<<<<<<<<<<<<


        ########## super ####################################>>>>>>>>>>>>>>>
        label = label - 1
        pos, label = pos.cuda(non_blocking=True), label.cuda(non_blocking=True)
        out_3 = net_head(net(pos))
        loss_super = criterion(out_3, label)
        ########## super ######################################<<<<<<<<<<<<<<


        ########## joint ####################################>>>>>>>>>>>>>>>
        if args.awl:
            loss = awl(args.lambda_contra*total_contra_loss, args.lambda_super*loss_super)
        else:
            # print("not joint")
            loss = args.lambda_contra*total_contra_loss + args.lambda_super*loss_super
        ########## joint ####################################<<<<<<<<<<<<<<<


        pred = out_3.data.max(1, keepdim=True)[1]
        train_correct += pred.eq(label.data.view_as(pred)).sum()
        train_accuracy = 100. * train_correct / len(memory_loader.dataset)
        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += args.batch_size
        total_loss += loss.item() * args.batch_size
        average_loss = total_loss / total_num
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f} loss_contra: {:.4f} loss_super: {:.4f} TRA: {:.4f}'.format(\
                                epoch, args.epochs, round(average_loss, 4), round(args.lambda_contra*total_contra_loss.item(), 4), 
                                round(args.lambda_super*loss_super.item(), 4), round(train_accuracy.item(), 4)))

    if epoch % args.log_interval1 == 0:
        net.eval()
        net_head.eval()
        for t_pos, t_pos_, t_label in test_loader:
            with torch.no_grad():
                t_label = t_label - 1
                t_pos, t_label = t_pos.cuda(non_blocking=True), t_label.cuda(non_blocking=True)
                out_4 = net(t_pos)            # 还需要开发一个从头开始的版本
                out_4 = net_head(out_4)

                t_pred = out_4.data.max(1, keepdim=True)[1]
                test_correct += t_pred.eq(t_label.data.view_as(t_pred)).sum()
                test_accuracy = 100. * test_correct / len(test_loader.dataset)
                test_accuracy = round(test_accuracy.item(), 4)
        # print("linear test_accuracy", round(test_accuracy.item(), 4)) 
    else:
        test_accuracy = 0.0

    return round(average_loss, 4), round(args.lambda_contra*total_contra_loss.item(), 4), round(args.lambda_super*loss_super.item(), 4), round(train_accuracy.item(), 4), test_accuracy
