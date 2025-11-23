import torch
import pandas as pd
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_loss(predict: torch.Tensor, reallabel_onehot: torch.Tensor,
                 reallabel_mask: torch.Tensor):
    real_labels = reallabel_onehot
    we = -torch.mul(real_labels, torch.log(predict + 1e-10))
    we = torch.mul(we, reallabel_mask)
    pool_cross_entropy = torch.sum(we) # 二元交叉熵损失函数
    return pool_cross_entropy

def cal_OA(network_output, train_samples_gt, train_samples_gt_onehot, zeros):

    with torch.no_grad():
        available_label_idx = (train_samples_gt != 0).float()
        available_label_count = available_label_idx.sum()
        correct_prediction = torch.where(
            torch.argmax(network_output, 1) == torch.argmax(train_samples_gt_onehot, 1), available_label_idx,
            zeros).sum()
        OA = correct_prediction.cpu() / available_label_count
        return OA

def train(net, lr, num_epochs, net_input_before, net_input_after, net_input_concat, train_gt, train_onehot, train_mask, val_gt, val_onehot, val_mask):
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    best_loss = 1000000  # 初始化损失
    height = train_onehot.size()[0]
    width = train_onehot.size()[1]
    zeros = torch.zeros([height * width]).to(device).float()
    net.train()
    train_onehot = train_onehot.reshape([-1,2])
    val_onehot = val_onehot.reshape([-1, 2])
    train_gt = train_gt.reshape([-1])
    val_gt = val_gt.reshape([-1])

    # 用于记录训练和验证数据
    train_losses = []
    train_OAs = []
    val_losses = []
    val_OAs = []

    for i in range(num_epochs + 1):
        # 梯度先清零
        optimizer.zero_grad()
        output = net(net_input_before, net_input_after, net_input_concat)
        loss = compute_loss(output, train_onehot, train_mask)
        loss.backward(retain_graph=False)
        optimizer.step()
        # 每训练10轮进行一次验证
        if i % 10 == 0:
            with torch.no_grad():
                net.eval()
                output = net(net_input_before, net_input_after, net_input_concat)
                trainloss = compute_loss(output, train_onehot, train_mask)
                trainOA = cal_OA(output, train_gt, train_onehot, zeros)
                valloss = compute_loss(output, val_onehot, val_mask)
                valOA = cal_OA(output, val_gt, val_onehot, zeros)
                
                # 记录数据
                train_losses.append(trainloss.item())
                train_OAs.append(trainOA.item())
                val_losses.append(valloss.item())
                val_OAs.append(valOA.item())

                print("{}\ttrain loss={}\t train OA={} val loss={}\t val OA={}".format(str(i + 1), trainloss, trainOA,
                                                                                       valloss, valOA))
                # 多次迭代，损失之小于best保存为模型
                if valloss < best_loss:
                    best_loss = valloss
                    torch.save(net.state_dict(), "model\\best_model.pt")
                    print('saving model')
            torch.cuda.empty_cache()
            net.train()
    print("\n-----training ok. start testing-----\n")

    # 保存训练和验证数据到 Excel
    data = {
        'Epoch': list(range(0, num_epochs + 1, 10)),
        'Train Loss': train_losses,
        'Train OA': train_OAs,
        'Val Loss': val_losses,
        'Val OA': val_OAs
    }
    df = pd.DataFrame(data)
    if not os.path.exists('results'):
        os.makedirs('results')
    df.to_excel('results/train_val_data.xlsx', index=False)

    return train_losses, train_OAs, val_losses, val_OAs
