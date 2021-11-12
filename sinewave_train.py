import  torch, os
import  numpy as np
from    SineWave import SineWave
import  argparse
import pdb
from   meta import Meta

def main(args):

    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)
    print(args)

    def add_info(name, value):
        return '_' + name + '_' + str(value)


    save_dir = os.environ['PROJECT']
    model_path = save_dir + '/MAML/models/'
    result_path = save_dir + '/MAML/results/'
    
    info = 'sinewave_'
    if args.dimi_m_coef:
        info += 'dim'
    else:
        info += 'const'

    info +=  add_info('beta', args.m_coef) + add_info('shot', args.k_spt)\
         + add_info('task_num', args.task_num) + add_info('inner_step', args.update_step) + add_info('qry_size', args.k_qry)

    prefix = result_path + info
    save_path = model_path + info

    config = [
        ('linear', [40, 1]),
        ('relu', [True]),
        ('linear', [1, 40])
    ]

    device = torch.device('cuda')
    maml = Meta(args, config).to(device)
    
    if args.restore:
        maml.net.load_state_dict(torch.load(save_path))
    
    tmp = filter(lambda x: x.requires_grad, maml.parameters())
    num = sum(map(lambda x: np.prod(x.shape), tmp))
    print(maml)
    print('Total trainable tensors:', num)
    
    if args.restore:
        train_loss = list(np.load(prefix +'train_loss.npy'))
        train_acc = list(np.load(prefix +'train_acc.npy'))
        val_loss = list(np.load(prefix +'val_loss.npy'))
        val_acc = list(np.load(prefix +'val_acc.npy'))
    else:
        train_acc =[]; val_acc = []; train_loss = []; val_loss = []


    db_train = SineWave('SineWave',
                       batchsz=args.task_num,
                       k_shot=args.k_spt)

    step = len(train_acc)
    # for step in range(len(train_acc), args.epoch):
    for batch in db_train.dataloader:
        pdb.set_trace()
        x_spt, y_spt = batch['train']
        x_qry, y_qry = batch['test']

        # pdb.set_trace()
        # x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
        #                              torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)
        x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), \
                                     x_qry.to(device), y_qry.to(device)

        # set traning=True to update running_mean, running_variance, bn_weights, bn_bias
        if args.dimi_m_coef: maml.m_coef =  max(1/(step // 5000 + 1) ** 0.5, 0.2)
        accs, losses = maml(x_spt, y_spt, x_qry, y_qry)
        train_acc.append(accs[-1]); train_loss.append(losses[-1])

        if step % 50 == 0:
            print('step:', step, '\ttraining acc:', accs, '\ttraining loss:', losses)
            
            np.save(prefix +'train_loss.npy', train_loss)
            np.save(prefix +'train_acc.npy', train_acc)
            
            torch.save(maml.net.state_dict(), save_path)
            
            
            
        if step % 500 == 0:
            accs = []; losses = []
            test_step = 0
            # for _ in range(600//args.task_num):
            for test_batch in db_train.dataloader_val:
                # test
                # x_spt, y_spt, x_qry, y_qry = db_train.next('test')
                # x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt).to(device), torch.from_numpy(y_spt).to(device), \
                #                              torch.from_numpy(x_qry).to(device), torch.from_numpy(y_qry).to(device)

                x_spt, y_spt = test_batch['train']
                x_qry, y_qry = test_batch['test']
                x_spt, y_spt, x_qry, y_qry = x_spt.to(device), y_spt.to(device), x_qry.to(device), y_qry.to(device)

                # split to single task each time
                for x_spt_one, y_spt_one, x_qry_one, y_qry_one in zip(x_spt, y_spt, x_qry, y_qry):
                    test_acc, test_loss = maml.finetunning(x_spt_one, y_spt_one, x_qry_one, y_qry_one)
                    accs.append(test_acc); losses.append(test_loss)
                
                test_step += args.task_num
                if test_step > 100:
                    break

            # [b, update_step+1]
            accs = np.array(accs).mean(axis=0).astype(np.float16)
            losses = np.array(losses).mean(axis=0).astype(np.float16)
            print('Test acc:', accs, '\tTest loss:', losses)
            
            val_acc.append(accs[-1]); val_loss.append(losses[-1])
            
            np.save(prefix +'val_loss.npy', val_loss)
            np.save(prefix +'val_acc.npy', val_acc)

        step += 1
        if step > args.epoch:
            break

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--epoch', type=int, help='epoch number', default=500000)
    argparser.add_argument('--k_spt', type=int, help='k shot for support set', default=1)
    argparser.add_argument('--k_qry', type=int, help='k shot for query set', default=15)
    argparser.add_argument('--task_num', type=int, help='meta batch size, namely task num', default=3)
    argparser.add_argument('--meta_lr', type=float, help='meta-level outer learning rate', default=1e-3)
    argparser.add_argument('--m_coef', type=float, help='momentum coefficient for SCGD', default=1)
    argparser.add_argument('--mu', type=float, help='momentum coefficient for SCGD outer update', default=0)
    argparser.add_argument('--update_lr', type=float, help='task-level inner update learning rate', default=0.4)
    argparser.add_argument('--update_step', type=int, help='task-level inner update steps', default=1)
    argparser.add_argument('--update_step_test', type=int, help='update steps for finetunning', default=5)
    argparser.add_argument('--dimi_m_coef', dest='dimi_m_coef', action='store_true')
    argparser.add_argument('--restore', dest='restore', action='store_true')


    args = argparser.parse_args()

    main(args)
