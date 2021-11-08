import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    learner import Learner
from    copy import deepcopy

import pdb

class Meta(nn.Module):
    """
    Meta Learner
    """
    def __init__(self, args, config):
        """

        :param args:
        """
        super(Meta, self).__init__()

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.m_coef = args.m_coef
        self.n_way = args.n_way
        self.k_spt = args.k_spt
        self.k_qry = args.k_qry
        self.mu = args.mu
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test


        self.net = Learner(config, args.imgc, args.imgsz)
        self.momentum_weight = None
        if self.mu > 0:
            self.meta_optim = optim.SGD(self.net.parameters(), lr=self.meta_lr, momentum=self.mu)
        else:
            self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def clip_grad_by_norm_(self, grad, max_norm):
        """
        in-place gradient clipping.
        :param grad: list of gradients
        :param max_norm: maximum norm allowable
        :return:
        """

        total_norm = 0
        counter = 0
        for g in grad:
            param_norm = g.data.norm(2)
            total_norm += param_norm.item() ** 2
            counter += 1
        total_norm = total_norm ** (1. / 2)

        clip_coef = max_norm / (total_norm + 1e-6)
        if clip_coef < 1:
            for g in grad:
                g.data.mul_(clip_coef)

        return total_norm/counter


    def forward(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        task_num, setsz, c_, h, w = x_spt.size()
        querysz = x_qry.size(1)

        losses_q = [0 for _ in range(2)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(2)]

        tmp_weight = [torch.zeros_like(p) for p in self.net.parameters()]
        tmp_grad = [torch.zeros_like(p) for p in self.net.parameters()]
        for i in range(task_num):
            # pdb.set_trace()

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()
                corrects[0] = corrects[0] + correct


            fast_weights = list(map(lambda p: p, self.net.parameters()))

            for _ in range(self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.cross_entropy(logits, y_spt[i])
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            if self.momentum_weight is None:
                u_state = [u.detach().clone().requires_grad_() for u in fast_weights]
            else:
                u_state = list(map(lambda p: (1 - self.m_coef) * p[0] + self.m_coef * p[1].detach().clone(), \
                    zip(self.momentum_weight, fast_weights)))
                u_state = [u.detach().clone().requires_grad_() for u in u_state]

            for tmp_w, cur_w in zip(tmp_weight, u_state):
                tmp_w += cur_w/task_num

            logits_q = self.net(x_qry[i], u_state, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry[i]); losses_q[1] += loss_q
            grad_q = torch.autograd.grad(loss_q, u_state)
            grad = torch.autograd.grad(fast_weights, self.net.parameters(), grad_outputs=grad_q)
            for tmp_g, cur_g in zip(tmp_grad, grad):
                tmp_g += cur_g/task_num 
                 
            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry[i]).sum().item()  # convert to numpy
                corrects[1] += correct

        

        # optimize theta parameters
        self.momentum_weight = [u.detach().clone() for u in tmp_weight]
        self.meta_optim.zero_grad()
        for p, g in zip(self.net.parameters(), tmp_grad):
            p.grad = g.clone()
        # loss_q.backward()
        self.meta_optim.step()


        accs = np.array(corrects) / (querysz * task_num)
        losses = np.array([l.data.cpu().numpy().item() for l in losses_q]) / (task_num)
        return accs, losses


    def finetunning(self, x_spt, y_spt, x_qry, y_qry):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        assert len(x_spt.shape) == 4

        querysz = x_qry.size(0)

        losses_q = [0 for _ in range(self.update_step_test + 1)] 
        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            loss_q = F.cross_entropy(logits_q, y_qry)
            losses_q[0] += loss_q
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            loss_q = F.cross_entropy(logits_q, y_qry)
            losses_q[1] += loss_q
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.cross_entropy(logits_q, y_qry)
            losses_q[k + 1] += loss_q

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct


        del net

        accs = np.array(corrects) / querysz
        losses = np.array([l.data.cpu().numpy().item() for l in losses_q])

        return accs, losses




def main():
    pass


if __name__ == '__main__':
    main()
