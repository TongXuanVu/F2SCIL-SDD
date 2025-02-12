import torch
import torch.nn.functional as F

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    if torch.cuda.device_count() > 1:
        device = torch.device('cuda:0')
    else:
        device = torch.device('cuda')
else:
    device = torch.device('cpu')


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


class CE(torch.nn.Module):
    def __init__(self, num_classes=60):
        super(CE, self).__init__()
        self.device = device

    def forward(self, output, labels, t_out, s_out):
        _, predicted = t_out.max(1)
        out_logit = torch.cat((output, s_out), dim=0)
        label_s = torch.cat((labels, predicted.detach()), dim=0)
        loss = F.cross_entropy(out_logit, label_s)
        return loss


class SCE(torch.nn.Module):
    def __init__(self, alpha, beta, num_classes=10):
        super(SCE, self).__init__()
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(labels, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


class CEandSCE(torch.nn.Module):
    def __init__(self, alpha=1, beta=1,lam=1, num_classes=60):
        super(CEandSCE, self).__init__()
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.num_classes = num_classes

    def forward(self, output, labels, t_out, s_out):
        _, predicted = t_out.max(1)
        # OCE
        old_ce = F.cross_entropy(output, labels)
        # NCE
        new_ce = F.cross_entropy(s_out[:, :self.num_classes], predicted)
        # RCE
        pred = F.softmax(s_out[:, :self.num_classes], dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = torch.nn.functional.one_hot(predicted, self.num_classes).float().to(self.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))

        loss_sce = self.alpha * new_ce + self.beta * rce.mean()
        loss = old_ce + self.lam * loss_sce
        return loss


class CEandKD(torch.nn.Module):
    def __init__(self, alpha=1, num_classes=60):
        super(CEandKD, self).__init__()
        self.device = device
        self.alpha = alpha
        self.num_classes = num_classes

    def forward(self, output, labels, t_out, s_out):
        fake_targets = labels - self.num_classes
        loss_ce = F.cross_entropy(output[:, self.num_classes:], fake_targets)
        loss_old = _KD_loss(
            s_out[:, :self.num_classes],
            t_out.detach(), 2, )
        loss = loss_ce + self.alpha * loss_old
        return loss