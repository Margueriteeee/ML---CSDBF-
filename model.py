import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# device =torch.device("cpu")

class GraphAttentionLayer(nn.Module):
# 图节点矩阵的参数化
    def __init__(self, A1, A2, A3, A12, A13, A23, A21, A31, A32, nodes1, nodes2, nodes3, in_features, out_features,
                 dropout, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        # 初始化邻接矩阵
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.A12 = A12
        self.A13 = A13
        self.A23 = A23
        self.A21 = A21
        self.A31 = A31
        self.A32 = A32
        self.nodes1 = nodes1
        self.nodes2 = nodes2
        self.nodes3 = nodes3

        # 初始化权重矩阵 W，torch.empty用于创建指定大小的张量，完成特征维度的线性变换
        # nn.Prameter用于将张量转化为可学习的参数，在反向传播是自动更新
        self.W = nn.Parameter(torch.empty(size=(self.in_features, self.out_features))).to(device)
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.empty(size=(2 * self.out_features, 1))).to(device)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # 初始化激活函数
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h):
    # 超像素级分支——实现前向传播，h为输入的节点特征矩阵（节点总数K1+K2+K3,输入特征维度C)
        Wh = torch.mm(h, self.W)  # (nodes, hid)

        e = self._prepare_attentional_mechanism_input(Wh)  # (nodes, nodes)
        zero_vec = -9e15 * torch.ones_like(e)

        # attention 存储经过掩码处理后的注意力系数
        attention = torch.zeros_like(e)

        # 时空联合相关性注入
        attention[0:self.nodes1, 0:self.nodes1] = torch.where(self.A1 > 0, e[0:self.nodes1, 0:self.nodes1],
                                                              zero_vec[0:self.nodes1, 0:self.nodes1])
        attention[self.nodes1:self.nodes1 + self.nodes2, 0:self.nodes1] = torch.where(self.A21 > 0, e[
                                                                                                    self.nodes1:self.nodes1 + self.nodes2,
                                                                                                    0:self.nodes1],
                                                                                      zero_vec[
                                                                                      self.nodes1:self.nodes1 + self.nodes2,
                                                                                      0:self.nodes1])
        attention[0:self.nodes1, self.nodes1:self.nodes1 + self.nodes2] = torch.where(self.A12 > 0, e[0:self.nodes1,
                                                                                                    self.nodes1:self.nodes1 + self.nodes2],
                                                                                      zero_vec[0:self.nodes1,
                                                                                      self.nodes1:self.nodes1 + self.nodes2])
        attention[self.nodes1:self.nodes1 + self.nodes2, self.nodes1:self.nodes1 + self.nodes2] = torch.where(
            self.A2 > 0, e[self.nodes1:self.nodes1 + self.nodes2, self.nodes1:self.nodes1 + self.nodes2],
            zero_vec[self.nodes1:self.nodes1 + self.nodes2, self.nodes1:self.nodes1 + self.nodes2])
        attention[self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3, 0:self.nodes1] = torch.where(
            self.A31 > 0, e[self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3, 0:self.nodes1],
            zero_vec[self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3, 0:self.nodes1])
        attention[0:self.nodes1, self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3] = torch.where(
            self.A13 > 0, e[0:self.nodes1, self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3],
            zero_vec[0:self.nodes1, self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3])
        attention[self.nodes1:self.nodes1 + self.nodes2,
        self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3] = torch.where(self.A23 > 0, e[
                                                                                                       self.nodes1:self.nodes1 + self.nodes2,
                                                                                                       self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3],
                                                                                         zero_vec[
                                                                                         self.nodes1:self.nodes1 + self.nodes2,
                                                                                         self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3])
        attention[self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3,
        self.nodes1:self.nodes1 + self.nodes2] = torch.where(self.A32 > 0, e[
                                                                           self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3,
                                                                           self.nodes1:self.nodes1 + self.nodes2],
                                                             zero_vec[
                                                             self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3,
                                                             self.nodes1:self.nodes1 + self.nodes2])
        attention[self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3,
        self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3] = torch.where(self.A3 > 0, e[
                                                                                                      self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3,
                                                                                                      self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3],
                                                                                         zero_vec[
                                                                                         self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3,
                                                                                         self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3])
        # 注意稀疏矩阵归一化，随机丢弃防止过拟合
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        mid = torch.matmul(attention, Wh) # 加权特征和

        if self.concat:
            return F.elu(mid) # 拼接后输入激活函数 特征传播
        else:
            return mid

    def _prepare_attentional_mechanism_input(self, Wh):
        # 计算注意力系数
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        e = Wh1 + Wh2.T
        return self.leakyrelu(e) # 特征值组合

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

class GAT(nn.Module): 
    # 图注意网络，三层多头注意力
    def __init__(self, A1, A2, A3, A12, A13, A23, A21, A31, A32, nodes1, nodes2, nodes3, nfeat, nhid, nclass, dropout,
                 nheads, alpha):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.dropout = dropout
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        self.A12 = A12
        self.A13 = A13
        self.A23 = A23
        self.A21 = A21
        self.A31 = A31
        self.A32 = A32
        self.nodes1 = nodes1
        self.nodes2 = nodes2
        self.nodes3 = nodes3
        self.attentions3 = [
            GraphAttentionLayer(self.A1, self.A2, self.A3, self.A12, self.A13, self.A23, self.A21, self.A31, self.A32,
                                 self.nodes1, self.nodes2, self.nodes3, nfeat, nhid, dropout=dropout, alpha=alpha,
                                 concat=True) for _ in
            range(nheads)]
        for i, attention in enumerate(self.attentions3):
            self.add_module('attention3_{}'.format(i), attention)

    def forward(self, x):
        x1 = F.dropout(x, self.dropout, training=self.training)

        x2 = torch.cat([att1(x1) for att1 in self.attentions3], dim=1)

        x3 = x2[self.nodes1 + self.nodes2:self.nodes1 + self.nodes2 + self.nodes3, :]
        x4 = x2[0:self.nodes1, :]
        x5 = x2[self.nodes1:self.nodes1 + self.nodes2, :]

        x3 = F.dropout(x3, self.dropout, training=self.training)
        x4 = F.dropout(x4, self.dropout, training=self.training)
        x5 = F.dropout(x5, self.dropout, training=self.training)
        return F.log_softmax(x3, dim=1), F.log_softmax(x4, dim=1), F.log_softmax(x5, dim=1)

class SSConv(nn.Module):

    def __init__(self, in_ch, out_ch, kernel_size=3):
        super(SSConv, self).__init__()
        # nn.Conv2d 定义一个二维卷积层
        # 深度卷积（Depthwise Convolution）为输入特征图的每个通道分配一个独立的卷积核。每个卷积核只对对应的一个通道进行卷积操作，
        # 因此深度卷积不会改变通道数量，但会在空间维度上对每个通道进行特征提取。
        # 提取空间特征（空间由平面的二维坐标表示，是二维的，即2-d卷积）
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,  #输入特征图的通道数设置为 out_ch，这是因为深度可分离卷积中的深度卷积部分，每个通道都有一个独立的卷积核。
            out_channels=out_ch, #输出特征图的通道数同样设置为 out_ch，保持通道数不变。
            kernel_size=kernel_size,  #卷积核的大小由传入的 kernel_size 参数决定
            stride=1,   #卷积的步长设置为 1，表示每次卷积操作移动一个像素
            padding=kernel_size // 2,  #整除运算符“//”，3//2=1，意味着在输入特征图的上下左右各添加 1 个像素的填充，这样，卷积核在进行卷积操作时，能够覆盖到输入特征图的边缘部分
            groups=out_ch  #groups 参数用于指定分组卷积的组数，这里将其设置为 out_ch，表示每个通道都有一个独立的卷积核，实现了深度卷积
        )
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            groups=out_ch
        )
        # 逐点卷积（Pointwise Convolution）对于输入特征图的每个位置，1x1 卷积核会对该位置的所有通道进行线性组合。
        # 提取光谱特征（一维，1-d卷积）
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,    # 输入特征图的通道数为in_ch
            out_channels=out_ch,  # 输出特征图的通道数为 out_ch，用于调整通道数
            kernel_size=1,  # 使用 1x1 的卷积核，在不改变特征图尺寸的情况下，对通道进行线性组合
            stride=1,  # 卷积步长为1
            padding=0, # 不进行填充
            groups=1,  # 只有一组卷积，即普通卷积操作
            bias=False # 无偏置项
        )
        
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN = nn.BatchNorm2d(in_ch) # 批量归一化加速

    def forward(self, input):
        # 前向传播，先光谱卷积后空间卷积
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out

# =================================================================================
# MODIFICATION START: Define the new Cross-Attention Fusion Module
# =================================================================================
class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention Fusion Module.
    使用像素级特征 (P-branch) 作为Query，去关注超像素级上下文特征 (S-branch) 作为Key和Value。
    该模块取代了简单的拼接操作，以实现智能特征融合。
    """
    def __init__(self, in_channels_p, in_channels_s, inter_channels=None):
        super(CrossAttentionFusion, self).__init__()

        self.in_channels_p = in_channels_p
        self.in_channels_s = in_channels_s
        # 如果未指定中间维度，则默认为与P-branch一致
        self.inter_channels = inter_channels if inter_channels is not None else in_channels_p

        # 使用1x1卷积生成Query, Key, Value
        self.g = nn.Conv2d(self.in_channels_p, self.inter_channels, kernel_size=1) # Query (from P-branch)
        self.theta = nn.Conv2d(self.in_channels_s, self.inter_channels, kernel_size=1) # Key (from S-branch)
        self.phi = nn.Conv2d(self.in_channels_s, self.inter_channels, kernel_size=1) # Value (from S-branch)

        # 用于处理注意力加权后的特征
        self.W = nn.Sequential(
            nn.Conv2d(self.inter_channels, self.in_channels_p, kernel_size=1),
            nn.BatchNorm2d(self.in_channels_p)
        )
        # 初始化W中的权重，有助于稳定训练
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


        # 可学习的缩放参数gamma，初始化为0。
        # 这使得模块在训练初期等同于一个恒等映射（只通过P-branch特征），保证了训练的稳定性。
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, p_feat, s_feat):
        """
        Args:
            p_feat (torch.Tensor): P-branch的像素级特征 (B, C_p, H, W).
            s_feat (torch.Tensor): S-branch的超像素级特征 (B, C_s, H, W).
        Returns:
            torch.Tensor: 与p_feat形状相同的融合特征。
        """
        batch_size = p_feat.size(0)
        # N = H * W
        num_pixels = p_feat.size(2) * p_feat.size(3)

        # 生成 Q, K, V
        # q: (B, C_inter, H, W) -> (B, C_inter, N)
        q = self.g(p_feat).view(batch_size, self.inter_channels, -1)
        # k: (B, C_inter, H, W) -> (B, C_inter, N)
        k = self.theta(s_feat).view(batch_size, self.inter_channels, -1)
        # v: (B, C_inter, H, W) -> (B, C_inter, N)
        v = self.phi(s_feat).view(batch_size, self.inter_channels, -1)

        # 计算注意力图
        # energy: (B, N, C_inter) x (B, C_inter, N) -> (B, N, N)
        energy = torch.bmm(q.permute(0, 2, 1), k)
        attention = F.softmax(energy, dim=-1) # 在Key的维度上进行softmax

        # 将注意力应用到Value上
        # attended_v: (B, C_inter, N) x (B, N, N)^T -> (B, C_inter, N)
        attended_v = torch.bmm(v, attention.permute(0, 2, 1))
        
        # 重新塑形回图像格式
        attended_v = attended_v.view(batch_size, self.inter_channels, *p_feat.shape[2:])

        # 处理并缩放注意力加权后的特征
        out = self.W(attended_v)

        # 最终输出是残差连接：fused_feat = p_feat + gamma * out
        fused_feat = p_feat + self.gamma * out

        return fused_feat
# =================================================================================
# MODIFICATION END
# =================================================================================


class CSDBF(nn.Module):
    # height 和 width：分别表示输入数据的高度和宽度；changel：输入数据的维度
    # Q1, Q2, Q3：关联矩阵（Kn×C）
    # Aij：邻接矩阵
    # num 1，2，3：图节点数量
    def __init__(self, height: int, width: int, changel: int, Q1: torch.Tensor, A1: torch.Tensor,
                 Q2: torch.Tensor, A2: torch.Tensor, Q3: torch.Tensor, A3: torch.Tensor, A12: torch.Tensor,
                 A13: torch.Tensor, A23: torch.Tensor, A21: torch.Tensor, A31: torch.Tensor, A32: torch.Tensor,
                 num1: int, num2: int, num3: int):
        super(CSDBF, self).__init__()
        self.channel = changel
        self.height = height
        self.width = width
        self.Q1 = Q1
        self.Q2 = Q2
        self.Q3 = Q3
        # 空间邻接矩阵
        self.A1 = A1
        self.A2 = A2
        self.A3 = A3
        # 时间邻接矩阵
        self.A12 = A12
        self.A13 = A13
        self.A23 = A23
        self.A21 = A21
        self.A31 = A31
        self.A32 = A32
        # 图节点数量
        self.nodes1 = num1
        self.nodes2 = num2
        self.nodes3 = num3
        self.norm_col_Q1 = Q1 / (torch.sum(Q1, 0, keepdim=True))
        self.norm_col_Q2 = Q2 / (torch.sum(Q2, 0, keepdim=True))
        self.norm_col_Q3 = Q3 / (torch.sum(Q3, 0, keepdim=True))
        layers_count = 2 

        self.CNN_denoise = nn.Sequential()
        # 数据预处理：降维（两层）
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i),
                                            nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(128), )
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
        self.CNN_denoise1 = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise1.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(2 * self.channel))
                self.CNN_denoise1.add_module('CNN_denoise_Conv' + str(i),
                                             nn.Conv2d(2 * self.channel, 128, kernel_size=(1, 1)))
                self.CNN_denoise1.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
            else:
                self.CNN_denoise1.add_module('CNN_denoise_BN' + str(i), nn.BatchNorm2d(128), )
                self.CNN_denoise1.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise1.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
        
        self.CNN_Branch = nn.Sequential()
        ## 像素级特征提取分支，两个SSConv组成
        for i in range(2):
            if i < 1:
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(128, 128, kernel_size=1))
            else:
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(128, 64, kernel_size=5))

        # 图注意力网络
        self.myGAT = GAT(self.A1, self.A2, self.A3, self.A12, self.A13, self.A23, self.A21, self.A31, self.A32,
                         self.nodes1, self.nodes2, self.nodes3, nfeat=128, nhid=64, nclass=64, dropout=0.3, nheads=3,
                         alpha=0.2)
        
        # =================================================================================
        # MODIFICATION START: Instantiate the new fusion module and update the final classifier
        # =================================================================================
        
        # P-branch 输出通道数
        p_branch_out_channels = 64
        # S-branch 输出通道数 (nhid * nheads * 3 branches)
        s_branch_out_channels = 64 * 3 * 3
        
        # 实例化跨注意力融合模块
        self.fusion_module = CrossAttentionFusion(
            in_channels_p=p_branch_out_channels,
            in_channels_s=s_branch_out_channels,
            inter_channels=p_branch_out_channels # 设置中间维度与P-branch一致
        )

        # 替换原有的 Softmax_linear。新的分类器接收融合模块的输出，其维度与P-branch一致。
        self.classifier = nn.Sequential(nn.Linear(p_branch_out_channels, 2))
        
        # 注释掉原来的分类器
        # self.Softmax_linear = nn.Sequential(nn.Linear(640, 2))
        
        # =================================================================================
        # MODIFICATION END
        # =================================================================================

# 前向传播，三个张量输入
    def forward(self, x: torch.Tensor, y: torch.Tensor, abs: torch.Tensor):

        (h, w, c) = x.shape
        noise_x = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]),
                                                   0))
        noise_x = torch.squeeze(noise_x, 0).permute([1, 2, 0])
        clean_x = noise_x
        noise_y = self.CNN_denoise(torch.unsqueeze(y.permute([2, 0, 1]),
                                                   0))
        noise_y = torch.squeeze(noise_y, 0).permute([1, 2, 0])
        clean_y = noise_y

        noise_abs = self.CNN_denoise1(torch.unsqueeze(abs.permute([2, 0, 1]), 0))
        noise_abs = torch.squeeze(noise_abs, 0).permute([1, 2, 0])
        clean_abs = noise_abs

        clean_x_flatten = clean_x.reshape([h * w, -1])
        clean_y_flatten = clean_y.reshape([h * w, -1])
        clean_abs_flatten = clean_abs.reshape([h * w, -1])

        # 根据关联矩阵Q将展平后的像素级数据转换为超像素特征，展平并归一化
        superpixels_flatten_x = torch.mm(self.norm_col_Q1.t(),
                                         clean_x_flatten)
        superpixels_flatten_y = torch.mm(self.norm_col_Q2.t(),
                                         clean_y_flatten)
        superpixels_flatten_abs = torch.mm(self.norm_col_Q3.t(), clean_abs_flatten)

        # 像素级分支，SSCONV提取特征
        CNN_result_abs = self.CNN_Branch(torch.unsqueeze(clean_abs.permute([2, 0, 1]), 0))
        CNN_result_abs_reshaped = torch.squeeze(CNN_result_abs, 0).permute([1, 2, 0]).reshape([h * w, -1])

        # 超像素级分支，超像素分割结果编码为图结构输入图注意力网络模块
        GAT_input1 = superpixels_flatten_x
        GAT_input2 = superpixels_flatten_y
        GAT_input3 = superpixels_flatten_abs
        # 特征拼接
        GAT_input_concat = torch.cat([GAT_input1, GAT_input2, GAT_input3], dim=0)

        GAT_result1, GAT_result2, GAT_result3 = self.myGAT(GAT_input_concat)
        #特征解码
        GAT_result1 = torch.matmul(self.Q3, GAT_result1)
        GAT_result2 = torch.matmul(self.Q1, GAT_result2)
        GAT_result3 = torch.matmul(self.Q2, GAT_result3)
        GAT_result_reshaped = torch.cat([GAT_result1, GAT_result2, GAT_result3], dim=1)

        # =================================================================================
        # MODIFICATION START: Apply the new fusion logic
        # =================================================================================

        # 将P-branch和S-branch的扁平化特征重塑为(B, C, H, W)格式以用于融合模块
        # P-branch 特征 (Query)
        p_feat = CNN_result_abs_reshaped.permute(1, 0).view(1, 64, h, w)
        # S-branch 特征 (Key, Value)
        s_feat = GAT_result_reshaped.permute(1, 0).view(1, 576, h, w)
        
        # 调用跨注意力融合模块
        fused_features = self.fusion_module(p_feat, s_feat)
        
        # 将融合后的特征重新变回 (H*W, C) 的扁平格式
        Y = fused_features.squeeze(0).permute(1, 2, 0).reshape(h * w, -1)
        
        # 注释掉原来的融合方式
        # Y = torch.cat([CNN_result_abs,GAT_result],dim=-1)
        
        # 将融合后的特征送入新的分类器
        Y = self.classifier(Y)

        # 注释掉原来的分类器调用
        # Y = self.Softmax_linear(Y)

        # =================================================================================
        # MODIFICATION END
        # =================================================================================

        # 双分支特征融合
        # Y = torch.cat([CNN_result_abs,GAT_result],dim=-1)
        # softmax函数激活
        # Y = self.Softmax_linear(Y)
        
        # 变化检测
        Y = F.softmax(Y, -1)
        return Y
