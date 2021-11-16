import  torch
from    torch import nn
from    torch import optim
from    torch.nn import functional as F
from    torch.utils.data import TensorDataset, DataLoader
from    torch import optim
import  numpy as np

from    copy import deepcopy

import pdb

class Learner(nn.Module):
    """

    """

    def __init__(self, config):
        """

        :param config: network config file, type:list of (string, list)
        """
        super(Learner, self).__init__()


        self.config = config

        # this dict contains all tensors needed to be optimized
        self.vars = nn.ParameterList()

        for i, (name, param) in enumerate(self.config):
            if name is 'conv2d':
                # [ch_out, ch_in, kernelsz, kernelsz]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'convt2d':
                # [ch_in, ch_out, kernelsz, kernelsz, stride, padding]
                w = nn.Parameter(torch.ones(*param[:4]))
                # gain=1 according to cbfin's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_in, ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[1])))

            elif name is 'linear':
                # [ch_out, ch_in]
                w = nn.Parameter(torch.ones(*param))
                # gain=1 according to cbfinn's implementation
                torch.nn.init.kaiming_normal_(w)
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

            elif name is 'bn':
                # [ch_out]
                w = nn.Parameter(torch.ones(param[0]))
                self.vars.append(w)
                # [ch_out]
                self.vars.append(nn.Parameter(torch.zeros(param[0])))

                # must set requires_grad=False
                running_mean = nn.Parameter(torch.zeros(param[0]), requires_grad=False)
                running_var = nn.Parameter(torch.ones(param[0]), requires_grad=False)
                self.vars_bn.extend([running_mean, running_var])


            elif name in ['tanh', 'relu', 'upsample', 'avg_pool2d', 'max_pool2d',
                          'flatten', 'reshape', 'leakyrelu', 'sigmoid']:
                continue
            else:
                raise NotImplementedError






    def extra_repr(self):
        info = ''

        for name, param in self.config:
            if name is 'conv2d':
                tmp = 'conv2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[1], param[0], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'convt2d':
                tmp = 'convTranspose2d:(ch_in:%d, ch_out:%d, k:%dx%d, stride:%d, padding:%d)'\
                      %(param[0], param[1], param[2], param[3], param[4], param[5],)
                info += tmp + '\n'

            elif name is 'linear':
                tmp = 'linear:(in:%d, out:%d)'%(param[1], param[0])
                info += tmp + '\n'

            elif name is 'leakyrelu':
                tmp = 'leakyrelu:(slope:%f)'%(param[0])
                info += tmp + '\n'


            elif name is 'avg_pool2d':
                tmp = 'avg_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name is 'max_pool2d':
                tmp = 'max_pool2d:(k:%d, stride:%d, padding:%d)'%(param[0], param[1], param[2])
                info += tmp + '\n'
            elif name in ['flatten', 'tanh', 'relu', 'upsample', 'reshape', 'sigmoid', 'use_logits', 'bn']:
                tmp = name + ':' + str(tuple(param))
                info += tmp + '\n'
            else:
                raise NotImplementedError

        return info



    def forward(self, x, vars=None, bn_training=True):
        """
        This function can be called by finetunning, however, in finetunning, we dont wish to update
        running_mean/running_var. Thought weights/bias of bn is updated, it has been separated by fast_weights.
        Indeed, to not update running_mean/running_var, we need set update_bn_statistics=False
        but weight/bias will be updated and not dirty initial theta parameters via fast_weiths.
        :param x: [b, 1, 28, 28]
        :param vars:
        :param bn_training: set False to not update
        :return: x, loss, likelihood, kld
        """

        if vars is None:
            vars = self.vars

        idx = 0
        bn_idx = 0

        for name, param in self.config:
            if name is 'conv2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'convt2d':
                w, b = vars[idx], vars[idx + 1]
                # remember to keep synchrozied of forward_encoder and forward_decoder!
                x = F.conv_transpose2d(x, w, b, stride=param[4], padding=param[5])
                idx += 2
                # print(name, param, '\tout:', x.shape)
            elif name is 'linear':
                w, b = vars[idx], vars[idx + 1]
                x = F.linear(x, w, b)
                idx += 2
                # print('forward:', idx, x.norm().item())
            elif name is 'bn':
                w, b = vars[idx], vars[idx + 1]
                running_mean, running_var = self.vars_bn[bn_idx], self.vars_bn[bn_idx+1]
                x = F.batch_norm(x, running_mean, running_var, weight=w, bias=b, training=bn_training)
                idx += 2
                bn_idx += 2

            elif name is 'flatten':
                # print(x.shape)
                x = x.view(x.size(0), -1)
            elif name is 'reshape':
                # [b, 8] => [b, 2, 2, 2]
                x = x.view(x.size(0), *param)
            elif name is 'relu':
                x = F.relu(x, inplace=param[0])
            elif name is 'leakyrelu':
                x = F.leaky_relu(x, negative_slope=param[0], inplace=param[1])
            elif name is 'tanh':
                x = F.tanh(x)
            elif name is 'sigmoid':
                x = torch.sigmoid(x)
            elif name is 'upsample':
                x = F.upsample_nearest(x, scale_factor=param[0])
            elif name is 'max_pool2d':
                x = F.max_pool2d(x, param[0], param[1], param[2])
            elif name is 'avg_pool2d':
                x = F.avg_pool2d(x, param[0], param[1], param[2])

            else:
                raise NotImplementedError

        # make sure variable is used properly
        assert idx == len(vars)


        return x


    def zero_grad(self, vars=None):
        """

        :param vars:
        :return:
        """
        with torch.no_grad():
            if vars is None:
                for p in self.vars:
                    if p.grad is not None:
                        p.grad.zero_()
            else:
                for p in vars:
                    if p.grad is not None:
                        p.grad.zero_()

    def parameters(self):
        """
        override this function since initial parameters will return with a generator.
        :return:
        """
        return self.vars


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
        self.k_spt = args.k_spt
        self.mu = args.mu
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.opt = args.opt
        self.mult_state = args.mult_state


        self.net = Learner(config)
        if self.mult_state:
            self.momentum_weight = [None] * 25
        else:
            self.momentum_weight = None
        if self.opt ==  'sgd':
            self.meta_optim = optim.SGD(self.net.parameters(), lr=self.meta_lr)
        elif self.opt == 'momentum':
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


    def forward(self, x_spt, y_spt, x_qry, y_qry, task_code=None):
        """

        :param x_spt:   [b, setsz, c_, h, w]
        :param y_spt:   [b, setsz]
        :param x_qry:   [b, querysz, c_, h, w]
        :param y_qry:   [b, querysz]
        :return:
        """
        # pdb.set_trace()
        task_num, setsz, _ = x_spt.size()
        querysz = x_qry.size(1)
        assert np.max(task_code) <= 24
        losses_q = [0 for _ in range(2)]  # losses_q[i] is the loss on step i


        # this is the loss and accuracy before first update
        # tmp_weights = [torch.zeros_like(p) for p in self.net.parameters()]
        if self.mult_state:
            tmp_state = [[torch.zeros_like(p) for p in self.net.parameters()] ]* 25
            tmp_count = [0]*25
        else:
            tmp_state = [torch.zeros_like(p) for p in self.net.parameters()]

        tmp_grad = [torch.zeros_like(p) for p in self.net.parameters()]
        for i in range(task_num):
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_qry[i], self.net.parameters(), bn_training=True)
                loss_q = F.mse_loss(logits_q, y_qry[i])/querysz
                losses_q[0] += loss_q


            fast_weights = list(map(lambda p: p, self.net.parameters()))
            # pdb.set_trace()
            for k in range(self.update_step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x_spt[i], fast_weights, bn_training=True)
                loss = F.mse_loss(logits, y_spt[i])/setsz
                # print(k,loss)
                # 2. compute grad on theta_pi
                
                # if k == self.update_step - 1:
                #     total_weight = torch.sum(torch.cat([torch.norm((f_p - p.detach().clone())**2).view(1,-1) for f_p, p in zip(fast_weights, self.net.parameters())]))
                #     # print(total_weight)
                #     loss = loss + 1e-2 * total_weight

                grad = torch.autograd.grad(loss, fast_weights, create_graph=True)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            
            # tmp_weights = [tmp_w + fast_w/task_num for tmp_w, fast_w in zip(tmp_weights, fast_weights)]
            # pdb.set_trace()
            if self.mult_state:
                if self.momentum_weight[task_code[i]] is None:
                    u_state = [u.detach().clone().requires_grad_() for u in fast_weights]
                else:
                    u_state = list(map(lambda p: (1 - self.m_coef) * p[0] + self.m_coef * p[1].detach().clone(), \
                        zip(self.momentum_weight[task_code[i]], fast_weights)))
                    u_state = [u.detach().clone().requires_grad_() for u in u_state]
            else:
                if self.momentum_weight is None:
                    u_state = [u.detach().clone().requires_grad_() for u in fast_weights]
                else:
                    u_state = list(map(lambda p: (1 - self.m_coef) * p[0] + self.m_coef * p[1].detach().clone(), \
                        zip(self.momentum_weight, fast_weights)))
                    u_state = [u.detach().clone().requires_grad_() for u in u_state]

            logits_q = self.net(x_qry[i], u_state, bn_training=True)
            loss_q = F.mse_loss(logits_q, y_qry[i])/querysz; losses_q[1] += loss_q.detach().clone()
            grad_q = torch.autograd.grad(loss_q, u_state)
            
            grad = torch.autograd.grad(fast_weights, self.net.parameters(), grad_outputs=grad_q)

            tmp_grad = [tmp_g + fast_g/task_num for tmp_g, fast_g in zip(tmp_grad, grad)]

            if self.mult_state:
                tmp_state[task_code[i]] = [tmp_st + state_cur for tmp_st, state_cur in zip(tmp_state[task_code[i]], u_state)]
                tmp_count[task_code[i]] += 1
            else:
                tmp_state = [tmp_st + state_cur/task_num for tmp_st, state_cur in zip(tmp_state, u_state)]

        # tmp_grad = [torch.zeros_like(p) for p in self.net.parameters()]
        # for i in range(task_num):
        #     logits_q = self.net(x_qry[i], tmp_state, bn_training=True)
        #     loss_q = F.mse_loss(logits_q, y_qry[i]); losses_q[1] += loss_q.detach().clone()
        #     grad_q = torch.autograd.grad(loss_q, tmp_state)

        #     tmp_grad = [tmp_g + fast_g/task_num for tmp_g, fast_g in zip(tmp_grad, grad_q)]

        
        # grad = torch.autograd.grad(tmp_weights, self.net.parameters(), grad_outputs=tmp_grad)
        # optimize theta parameters
        # print(grad[-1])
        # self.momentum_weight = [u.detach().clone() for u in tmp_state]
        if self.mult_state:
            for code in task_code:
                self.momentum_weight[code] = [u.detach().clone()/tmp_count[code] for u in tmp_state[code]]
        else:
            self.momentum_weight = [u.detach().clone() for u in tmp_state]
        
        self.meta_optim.zero_grad()
        for p, g in zip(self.net.parameters(), tmp_grad):
            p.grad = g.clone()
        # loss_q.backward()
        self.meta_optim.step()

        losses = np.array([l.data.cpu().numpy().item() for l in losses_q]) / task_num
        return losses


    def finetunning(self, x_spt, y_spt, x_qry, y_qry, useLogits=False):
        """

        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """
        querysz = x_qry.size(0)

        losses_q = [0 for _ in range(self.update_step_test + 1)] 
        corrects = [0 for _ in range(self.update_step_test + 1)]

        # in order to not ruin the state of running_mean/variance and bn_weight/bias
        # we finetunning on the copied model instead of self.net
        net = deepcopy(self.net)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x_spt)
        loss = F.mse_loss(logits, y_spt)/x_spt.size(0)
        grad = torch.autograd.grad(loss, net.parameters())
        fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, net.parameters())))

        # this is the loss and accuracy before first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, net.parameters(), bn_training=True)
            loss_q = F.mse_loss(logits_q, y_qry)/querysz
            losses_q[0] += loss_q
            # [setsz]

        # this is the loss and accuracy after the first update
        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_qry, fast_weights, bn_training=True)
            loss_q = F.mse_loss(logits_q, y_qry)/querysz
            losses_q[1] += loss_q
            # [setsz]

        best_loss = 1e8
        best_logits = None
        for k in range(1, self.update_step_test):
            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x_spt, fast_weights, bn_training=True)
            loss = F.mse_loss(logits, y_spt)/x_spt.size(0)
            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))

            logits_q = net(x_qry, fast_weights, bn_training=True)
            # loss_q will be overwritten and just keep the loss_q on last update step.
            loss_q = F.mse_loss(logits_q, y_qry)/querysz
            if loss_q < best_loss:
                best_loss = loss_q
                best_logits = logits_q
            losses_q[k + 1] += loss_q


        del net

        losses = np.array([l.data.cpu().numpy().item() for l in losses_q])

        if useLogits:
            return best_logits
        else:
            return losses




def main():
    pass


if __name__ == '__main__':
    main()
