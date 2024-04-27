# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import numpy as np

class SoftTargetCrossEntropy(nn.Module):

    def __init__(self, weight=None):
        super(SoftTargetCrossEntropy, self).__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.weight is not None:
            loss = torch.sum(-self.weight * target * F.log_softmax(x, dim=-1), dim=-1)
        else:
            loss = torch.sum(-target * F.log_softmax(x, dim=-1), dim=-1)
        return loss.mean()
    

# class centerloss(nn.Module):
#     """While lamda = 0, loss become cross entropy"""
#     def __init__(self):
#         super(centerloss, self).__init__()
#         self.center = nn.Parameter(10 * torch.randn(10, 2))
#         self.lamda = 0.2
#         self.weight = nn.Parameter(torch.Tensor(2, 10))  # (input,output)
#         nn.init.xavier_uniform_(self.weight)

#     def forward(self, feature, label):
#         batch_size = label.size()[0]
#         nCenter = self.center.index_select(dim=0, index=label)
#         distance = feature.dist(nCenter)
#         centerloss = (1 / 2.0 / batch_size) * distance
#         out = feature.mm(self.weight)
#         ceLoss = F.cross_entropy(out, label)
#         return out, ceLoss + self.lamda * centerloss


class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        # print(distmat.size())
        # print(x.size())
        # print(self.centers.size())
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

class SoftmaxCenterLoss(nn.Module):
    def __init__(self, weight, label_smoothing, class_num, features_num, center_lambda=0.01, usg_gpu=True):
        super().__init__()
        self.class_num = class_num
        self.center = CenterLoss(class_num, features_num, usg_gpu)
        self.ce = nn.CrossEntropyLoss(weight, label_smoothing=label_smoothing)
        self.center_lambda = center_lambda
    def forward(self, outputs, features, labels):
        center_loss = self.center(features, labels)
        ce_loss = self.ce(outputs, labels)
        return ce_loss + center_loss * self.center_lambda


# class CenterLoss(nn.Module):
#     def __init__(self, class_num, features_num):
#         super().__init__()
#         self.class_num = class_num
#         self.center = nn.Parameter(torch.randn(class_num, features_num))

#     def forward(self, features, labels):
#         # 根据labels个数将center个数扩充成于lables一样多
#         center_expand = self.center.index_select(dim=0, index=labels)
#         # 获取每种类别的个数
#         count = torch.histc(labels, bins=self.class_num, min=0, max=self.class_num - 1)
#         # 扩充count个数用于对每个类别的欧氏距离总和做除法,因为为整数如果要做除法给转换成float类型
#         count_expand = count.index_select(dim=0, index=labels).float()
#         # 求每一个类的特征值与该类中心点做欧氏距离后的平均值，用于梯度下降
#         loss = torch.sum(torch.sqrt(torch.sum(torch.pow(features - center_expand, 2), dim=1)) / count_expand)
#         return loss


class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(2, 10))  # (input,output)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x, label):
        out = x.mm(self.weight) # x @ self.weight
        loss = F.cross_entropy(out, label)
        return out, loss


class Modified(nn.Module):
    def __init__(self):
        super(Modified, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(2, 10))  # (input,output)
        nn.init.xavier_uniform_(self.weight)
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        # 因为renorm采用的是maxnorm，所以先缩小再放大以防止norm结果小于1

    def forward(self, x, label):
        w = self.weight
        w = w.renorm(2, 1, 1e-5).mul(1e5)
        out = x.mm(w)
        loss = F.cross_entropy(out, label)
        return out, loss


class NormFace(nn.Module):
    def __init__(self):
        super(NormFace, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(2, 10))  # (input,output)
        nn.init.xavier_uniform_(self.weight)
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.s = 16
        # 因为renorm采用的是maxnorm，所以先缩小再放大以防止norm结果小于1

    def forward(self, x, label):
        cosine = F.normalize(x).mm(F.normalize(self.weight, dim=0))
        loss = F.cross_entropy(self.s * cosine, label)
        return cosine, loss


class SphereFace(nn.Module):
    def __init__(self, m=4):
        super(SphereFace, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(2, 10))  # (input,output)
        nn.init.xavier_uniform_(self.weight)
        self.weight.data.renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m
        self.mlambda = [  # calculate cos(mx)
            lambda x: x ** 0,
            lambda x: x ** 1,
            lambda x: 2 * x ** 2 - 1,
            lambda x: 4 * x ** 3 - 3 * x,
            lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
            lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
        ]
        self.it = 0
        self.LambdaMin = 3
        self.LambdaMax = 30000.0
        self.gamma = 0

    def forward(self, input, label):
        # 注意，在原始的A-softmax中是不对x进行标准化的,
        # 标准化可以提升性能，也会增加收敛难度，A-softmax本来就很难收敛

        cos_theta = F.normalize(input).mm(F.normalize(self.weight, dim=0))
        cos_theta = cos_theta.clamp(-1, 1)  # 防止出现异常
        # 以上计算出了传统意义上的cos_theta，但为了cos(m*theta)的单调递减，需要使用phi_theta

        cos_m_theta = self.mlambda[self.m](cos_theta)
        # 计算theta，依据theta的区间把k的取值定下来
        theta = cos_theta.data.acos()
        k = (self.m * theta / 3.1415926).floor()
        phi_theta = ((-1) ** k) * cos_m_theta - 2 * k

        x_norm = input.pow(2).sum(1).pow(0.5)  # 这个地方决定x带不带模长，不带就要乘s
        x_cos_theta = cos_theta * x_norm.view(-1, 1)
        x_phi_theta = phi_theta * x_norm.view(-1, 1)

        ############ 以上计算target logit，下面构造loss，退火训练#####
        self.it += 1  # 用来调整lambda
        target = label.view(-1, 1)  # (B,1)

        onehot = torch.zeros(target.shape[0], 10).cuda().scatter_(1, target, 1)

        lamb = max(self.LambdaMin, self.LambdaMax / (1 + 0.2 * self.it))

        output = x_cos_theta * 1.0  # 如果不乘可能会有数值错误？
        output[onehot.byte()] -= x_cos_theta[onehot.byte()] * (1.0 + 0) / (1 + lamb)
        output[onehot.byte()] += x_phi_theta[onehot.byte()] * (1.0 + 0) / (1 + lamb)
        # 到这一步可以等同于原来的Wx+b=y的输出了，

        # 到这里使用了Focal Loss，如果直接使用cross_Entropy的话似乎效果会减弱许多
        log = F.log_softmax(output, 1)
        log = log.gather(1, target)

        log = log.view(-1)
        pt = log.data.exp()
        loss = -1 * (1 - pt) ** self.gamma * log

        loss = loss.mean()
        # loss = F.cross_entropy(x_cos_theta,target.view(-1))#换成crossEntropy效果会差
        return output, loss


class ArcMarginProduct(nn.Module):
    def __init__(self, s=32, m=0.5):
        super(ArcMarginProduct, self).__init__()
        self.in_feature = 2
        self.out_feature = 10
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.Tensor(2, 10))  # (input,output)
        nn.init.xavier_uniform_(self.weight)
        self.weight.data.renorm_(2, 1, 1e-5).mul_(1e5)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        # 为了保证cos(theta+m)在0-pi单调递减：
        self.th = math.cos(3.1415926 - m)
        self.mm = math.sin(3.1415926 - m) * m

    def forward(self, x, label):
        cosine = F.normalize(x).mm(F.normalize(self.weight, dim=0))
        cosine = cosine.clamp(-1, 1)  # 数值稳定
        sine = torch.sqrt(torch.max(1.0 - torch.pow(cosine, 2), torch.ones(cosine.shape).cuda() * 1e-7))  # 数值稳定

        ##print(self.sin_m)
        phi = cosine * self.cos_m - sine * self.sin_m  # 两角和公式
        # # 为了保证cos(theta+m)在0-pi单调递减：
        # phi = torch.where((cosine - self.th) > 0, phi, cosine - self.mm)#必要性未知
        #
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        output = output * self.s
        loss = F.cross_entropy(output, label)

        return output, loss


class QAMFace(nn.Module):
    def __init__(self, s=32, m=0.4):
        super(QAMFace, self).__init__()
        self.in_feature = 2
        self.out_feature = 10
        self.s = s
        self.m = m
        self.pi = np.pi
        self.weight = nn.Parameter(torch.Tensor(2, 10))  # (input,output)
        nn.init.xavier_uniform_(self.weight)
        self.weight.data.renorm_(2, 1, 1e-5).mul_(1e5)
        self.cnt = 0

    def forward(self, x, label):
        self.cnt += 1
        cosine = F.normalize(x).mm(F.normalize(self.weight, dim=0))
        cosine = cosine.clamp(-1 + 1e-7, 1 - 1e-7)  # 数值稳定
        theta = torch.acos(cosine)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1)

        target = 0.2 * (2 * self.pi - (theta + self.m)) ** 2
        others = 0.2 * (2 * self.pi - theta) ** 2
        output = (one_hot * target) + ((1.0 - one_hot) * others)
        loss = F.cross_entropy(output, label)
        # loss = F.cross_entropy((self.pi-2*(theta+0.1))*160,label)

        return output, loss


class LabelSmoothCrossEntropy(nn.Module):
    def __init__(self, n_class=2, eps=0.05, reduction='mean'):
        super().__init__()
        self.n_class = n_class
        self.eps = eps
        self.reduction = reduction
        self.log_softmax = nn.LogSoftmax(dim=1)
        
    def forward(self, outputs, labels):
        assert outputs.size(0) == labels.size(0)
        one_hot = F.one_hot(labels, self.n_class)
        smooth = one_hot * (1 - self.eps) + (1 - one_hot) * self.eps

        loss = torch.sum(-smooth * self.log_softmax(outputs), dim=1)
        
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
            
        return loss


# class LabelSmoothingCrossEntropy(nn.Module):
#     """
#     NLL loss with label smoothing.
#     """
#     def __init__(self, smoothing=0.1):
#         """
#         Constructor for the LabelSmoothing module.
#         :param smoothing: label smoothing factor
#         """
#         super(LabelSmoothingCrossEntropy, self).__init__()
#         assert smoothing < 1.0
#         self.smoothing = smoothing
#         self.confidence = 1. - smoothing

#     def forward(self, x, target):
#         logprobs = F.log_softmax(x, dim=-1)
#         nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
#         nll_loss = nll_loss.squeeze(1)
#         smooth_loss = -logprobs.mean(dim=-1)
#         loss = self.confidence * nll_loss + self.smoothing * smooth_loss
#         return loss.mean()


class WeightedCrossEntropyLoss(torch.nn.CrossEntropyLoss):
    """
    Network has to have NO NONLINEARITY!
    >>> input = torch.randn(3, 5, requires_grad=False)
    >>> target = torch.empty(3, dtype=torch.long).random_(5)
    >>> weight = torch.FloatTensor([1, 2, 1, 0, 1])
    >>> loss_fn = WeightedCrossEntropyLoss(weight=weight)
    >>> loss = loss_fn(input, target)
    >>> print(loss)
    >>> loss_fn = nn.CrossEntropyLoss(weight=weight)
    >>> loss = loss_fn(input, target)
    >>> print(loss)
    """
    def __init__(self, weight=None):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weight = weight

    def forward(self, inp, target):
        num_classes = inp.size()[1]
        target = target.long()

        i0 = 1
        i1 = 2

        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, num_classes)

        target = target.view(-1,)
        wce_loss = torch.nn.CrossEntropyLoss(weight=self.weight)

        return wce_loss(inp, target)


def binary_cross_entropyloss(prob, target, weight=None):
    loss = -weight * (target * (torch.log(prob)) + (1 - target) * (torch.log(1 - prob)))
    loss = torch.sum(loss) / torch.numel(target)
    return loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, weight=None, reduction='mean'):
        """
        Description:
            - construct label smoothing module with weight
        
        Args:
            - smoothing: (float) smoothing epsilon
            - weight: (Tensor) input class weight
            - reduction: (str) how the loss return
        """
        super().__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing
        self.reduction = reduction
        self.weight = weight
    
    def forward(self, out, target):
        log_probs = F.log_softmax(out, dim=-1)
        nll_loss = F.nll_loss(log_probs, target, weight=self.weight, reduction='mean')
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == 'none':
            pass
            
        return loss

    def __str__(self):
        return f'{self.__class__.__name__}(smoothing={self.smoothing}, weight={self.weight}, reduction={self.reduction})'
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += f'smoothing={self.smoothing}, '
        format_string += f'weight={self.weight}, '
        format_string += f'reduction={self.reduction}'
        format_string += ')'
        return format_string


from torch.nn.modules.loss import _Loss
import torch.distributed as dist
class SoftmaxEQLLoss(_Loss):
    def __init__(self, num_classes, indicator='pos', loss_weight=1.0, tau=1.0, eps=1e-4):
        super(SoftmaxEQLLoss, self).__init__()
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.tau = tau
        self.eps = eps

        assert indicator in ['pos', 'neg', 'pos_and_neg'], 'Wrong indicator type!'
        self.indicator = indicator

        # initial variables
        self.register_buffer('pos_grad', torch.zeros(num_classes))
        self.register_buffer('neg_grad', torch.zeros(num_classes))
        self.register_buffer('pos_neg', torch.ones(num_classes))
        # self.register_backward_hook(self.collect_grad)

    def forward(self, input, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        if self.indicator == 'pos':
            indicator = self.pos_grad.detach()
        elif self.indicator == 'neg':
            indicator = self.neg_grad.detach()
        elif self.indicator == 'pos_and_neg':
            indicator = self.pos_neg.detach() + self.neg_grad.detach()
        else:
            raise NotImplementedError
        
        # print(f'indicator: {indicator}')
        self.device = input.device
        label = label.to(self.device)

        one_hot = F.one_hot(label, self.num_classes)
        self.targets = one_hot.detach()
    
        matrix = (indicator[None, :].clamp(min=self.eps) / indicator[:, None].clamp(min=self.eps)).to(self.device)
        factor = matrix[label.long(), :].pow(self.tau)

        cls_score = input + (factor.log() * (1 - one_hot.detach()))
        loss = F.cross_entropy(cls_score, label)
        return loss * self.loss_weight

    def collect_grad(self, grad):
        # print(f'grad: {grad}')
        # print(f'targets: {self.targets}')
        grad = torch.abs(grad)
        pos_grad = torch.sum(grad * self.targets, dim=0)
        neg_grad = torch.sum(grad * (1 - self.targets), dim=0)

        # dist.all_reduce(pos_grad)
        # dist.all_reduce(neg_grad)

        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += f'num_classes={self.num_classes}, '
        format_string += f'indicator={self.indicator}, '
        format_string += f'loss_weight={self.loss_weight}, '
        format_string += f'tau={self.tau}, '
        format_string += f'eps={self.eps}'
        format_string += ')'
        return format_string


class SoftmaxEQV2LLoss(_Loss):
    def __init__(self, num_classes, indicator='pos', loss_weight=1.0, tau=1.0, eps=1e-4):
        super().__init__()
        self.loss_weight = loss_weight
        self.num_classes = num_classes
        self.tau = tau
        self.eps = eps

        assert indicator in ['pos', 'neg', 'pos_and_neg'], 'Wrong indicator type!'
        self.indicator = indicator

        # initial variables
        self.register_buffer('pos_grad', torch.zeros(num_classes))
        self.register_buffer('neg_grad', torch.zeros(num_classes))
        self.register_buffer('pos_neg', torch.ones(num_classes))

    def forward(self, input, label, weight=None, avg_factor=None, reduction_override=None, **kwargs):
        if self.indicator == 'pos':
            indicator = self.pos_grad.detach()
        elif self.indicator == 'neg':
            indicator = self.neg_grad.detach()
        elif self.indicator == 'pos_and_neg':
            indicator = self.pos_neg.detach() + self.neg_grad.detach()
        else:
            raise NotImplementedError
        
        if label.dim() == 1:
            one_hot = F.one_hot(label, self.num_classes)
        else:
            one_hot = label.clone()
        self.targets = one_hot.detach()

        indicator = indicator / (indicator.sum() + self.eps)
        indicator = (indicator ** self.tau + 1e-9).log()
        cls_score = input + indicator[None, :]
        loss = F.cross_entropy(cls_score, label)
        return loss * self.loss_weight

    def collect_grad(self, grad):
        # print(f'grad: {grad}')
        # print(f'targets: {self.targets}')
        grad = torch.abs(grad)
        pos_grad = torch.sum(grad * self.targets, dim=0)
        neg_grad = torch.sum(grad * (1 - self.targets), dim=0)

        # dist.all_reduce(pos_grad)
        # dist.all_reduce(neg_grad)

        self.pos_grad += pos_grad
        self.neg_grad += neg_grad
        self.pos_neg = self.pos_grad / (self.neg_grad + 1e-10)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += f'num_classes={self.num_classes}, '
        format_string += f'indicator={self.indicator}, '
        format_string += f'loss_weight={self.loss_weight}, '
        format_string += f'tau={self.tau}, '
        format_string += f'eps={self.eps}'
        format_string += ')'
        return format_string


class EQFocalLoss(_Loss):
    def __init__(self, reduction='none', eps=1e-9, maximum=1.0):
        super().__init__()
        self.reduction = reduction
        self.eps = eps
        self.maximum = maximum

    def forward(self, inputs, targets, gamma=2.0, scale_factor=8.0):
        ce_loss = F.cross_entropy(inputs, targets, reduce='none')
        ce_out = F.cross_entropy(inputs, targets)
        log_pt = -ce_loss
        pt = torch.exp(log_pt)

        targets = targets.view(-1, 1)
        grad_i = torch.autograd.grad(outputs=-ce_out, inputs=inputs)[0]
        grad_i = grad_i.gather(1, targets)
        pos_grad_i = F.relu(grad_i).sum()
        neg_grad_i = F.relu(-grad_i).sum()
        grad_i = pos_grad_i / (neg_grad_i + self.eps)
        grad_i = grad_i.clamp(0, max=self.maximum)

        dy_gamma = gamma + scale_factor * (1 - grad_i)
        dy_gamma = dy_gamma.view(-1)
        wf = dy_gamma / gamma
        weights = wf * (1 - pt) ** dy_gamma
        efl = weights * ce_loss
        return efl
        

class LDAMLoss(nn.Module):
    """
    Args:
        - cls_num_list: count_samples_per_class
    """
    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, smoothing=0.0):
        super(LDAMLoss, self).__init__()
        self.cls_num_list = cls_num_list
        self.max_m = max_m
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = weight
        self.smoothing = smoothing
        
    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)
        
        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0,1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m
    
        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s*output, target, weight=self.weight, label_smoothing=self.smoothing)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += f'cls_num_list={self.cls_num_list}, '
        format_string += f'max_m={self.max_m}, '
        format_string += f'weight={self.weight}, '
        format_string += f's={self.s}, '
        format_string += f'smoothing={self.smoothing}'
        format_string += ')'
        return format_string


class OrderFocalLoss(nn.Module):
    """
    Focal loss implementation
    
    Argus:
        - alpha: (float or tensor), weights for imbalance samples
        - gamma: (float), weight for difficult classified sample
        - n: (int), order of p
    """

    def __init__(self, alpha=1, gamma=2, n=1, reduction='mean'):
        super(OrderFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.n = n
        self.reduction = reduction.lower()

    def forward(self, inputs, targets):
        """
        - inputs: logits [batch_size, num_classes]
        - targets: targets [batch_size]
        """
        pt = torch.softmax(inputs, dim=-1)
        pt = pt.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        pn = pt ** self.n
        logpn = pn.log()

        loss = -self.alpha * (1 - pn)**self.gamma * logpn
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()


class WeightedEqualizedFocalLoss(nn.Module):
    """
    CrossEntropy Loss with weighted gamma
    """

    def __init__(self, alpha=1, gamma=2, theta=0.5, n=1, weight=None, smoothing=None, reduction='mean', ):
        super(WeightedEqualizedFocalLoss, self).__init__()
        assert 0<= theta <= 1.0, f'theta must be in range [0, 1.0], but got {theta}.'
        self.alpha = alpha
        self.theta = theta
        self.gamma = gamma
        self.n = n
        self.weight = weight
        self.reduction = reduction
        self.smoothing = smoothing
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none', label_smoothing=smoothing)

    def forward(self, inputs, target):
        """
        - inputs: logits [batch_size, num_classes]
        - targets: targets [batch_size]
        """
        logpt = self.ce(inputs, target)
        pt = torch.exp(-logpt)
        
        # order efl
        pn = pt**self.n
        logpn = pn.log()
        efl = -self.alpha * (1 - pn)**self.gamma * logpn
        
        loss = self.theta * logpt + (1 - self.theta) * efl
        # print(f'ce: {logpt}, efl: {efl}, loss: {loss}')
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        elif self.reduction == 'none':
            return loss
    
    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += f'alpha={self.alpha}, '
        format_string += f'gamma={self.gamma}, '
        format_string += f'theta={self.theta}, '
        format_string += f'n={self.n}, '
        format_string += f'weight={self.weight}, '
        format_string += f'smoothing={self.smoothing}, '
        format_string += f'reduction={self.reduction}'
        format_string += ')'
        return format_string


class CBLoss(nn.Module):
    """
    Class Balance Loss between `logits` and the ground truth `targets`
    
    Arguments:
        - logits: float tensor, output of model [batch, num_classes]
        - targets: int tensor, ground truth labels [batch]
        - samples_per_cls: a python list of size [num_classes], example 2 classes [100, 10]
        - num_classes: int, number of classification
        - loss_type: string, 'sigmoid', 'focal', 'softmax'
        - beta: float, hyperparemeter for Class Blanced Loss
    """
    def __init__(self, num_per_cls, num_classes, beta, reduction='none', eps=1e-8):
        super().__init__()
        self.num_per_cls = num_per_cls
        self.num_classes = num_classes
        self.beta = beta
        self.reduction = reduction.lower()
        
        effective_num = 1.0 - np.power(beta, num_per_cls)
        weights = (1.0 - beta) / (effective_num + eps)
        
        # normalizd
        weights = weights / np.sum(weights) * num_classes 
        
        self.weights = torch.from_numpy(weights).float()
    
    def forward(self, logits, targets, ):
        
        logpt = F.log_softmax(logits, dim=-1)
        logpt_weighted = self.weights.to(targets.device) * logpt
        loss = - logpt_weighted.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'none':
            return loss
        elif self.reduction == 'sum':
            return loss.sum()

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += f'num_per_cls={self.num_per_cls}, '
        format_string += f'num_classes={self.num_classes}, '
        format_string += f'beta={self.beta}, '
        format_string += f'reduction={self.reduction}'
        format_string += ')'
        return format_string


if __name__ == '__main__':
    torch.manual_seed(0)
    input = torch.randn(3, 2, requires_grad=True)#.cuda()
    target = torch.empty(3, dtype=torch.long).random_(2)#.cuda()
    weight = torch.FloatTensor([1, 1.5])
    # loss_fn = WeightedCrossEntropyLoss(weight=weight)
    # loss = loss_fn(input, target)
    # print(loss)
    # loss_fn = nn.CrossEntropyLoss(weight=weight)
    # loss = loss_fn(input, target)
    # print(loss)
    # loss_fn = nn.CrossEntropyLoss()
    # loss = loss_fn(input, target)
    # print(loss)
    # smoothing = 0.05
    # loss_fn = LabelSmoothingCrossEntropy(smoothing)
    # loss = loss_fn(input, target)
    # print(loss)
    # loss_fn = LabelSmoothingCrossEntropy(smoothing, weight)
    # loss = loss_fn(input, target)
    # print(loss)
    # print(torch.argmax(input, dim=1))
    # print(target)
    # loss_fn = LabelSmoothCrossEntropy(eps=smoothing)
    # loss = loss_fn(input, target)
    # print(loss)

    criterion = SoftmaxEQLLoss(2)
    for i in range(10):
        input = torch.randn(3, 2, requires_grad=True)#.cuda()
        target = torch.empty(3, dtype=torch.long).random_(2)#.cuda()
        loss = criterion(input, target)
        loss.backward()
        criterion.collect_grad(input.grad)
        print(loss)
    print(criterion)