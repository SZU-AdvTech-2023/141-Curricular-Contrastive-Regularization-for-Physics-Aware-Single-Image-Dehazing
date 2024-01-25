import time
import warnings

from tensorboardX import SummaryWriter
from torch import optim
from torch.backends import cudnn

from C2R import C2R
from data_utils import *
from models.C2PNet import *
from option import model_name, log_dir

warnings.filterwarnings('ignore')

print('log_dir :', log_dir)
print('model_name:', model_name)

models_ = {
    'C2PNet': C2PNet(gps=opt.gps, blocks=opt.blocks),
}
loaders_ = {
    #'its_train': ITS_train_loader_lmdb,
    'its_test': ITS_test_loader,
    'ots_test': OTS_test_loader,
     'ots_train': OTS_train_loader_all,
}

start_time = time.time()
#设置训练总步数 T 和课程学习策略中使用的一个参数 cl_lambda。

T = opt.steps
cl_lambda = opt.cl_lambda

#定义了一个余弦衰减学习率调度函数。这个函数会根据当前步数 t 和总步数 T 调整学习率。随着训练的进行，学习率会先减小后增大
def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr

#定义一个函数，根据图像的难易程度来调整它们在训练过程中的权重。这是课程学习策略的一部分
def curriculum_weight(difficulty):
    #根据测试集的不同，为难度设置不同的阈值列表。
    if opt.testset == 'OTS_test':
        diff_list = [17.43, 18.12, 30.86, 31.98, 32.98, 33.57]  # OTS
    else:
        diff_list = [17, 19, 31, 32, 36, 40]
        #计算了每个难度级别的权重，如果图像的难度大于某个阈值，它的权重会增加；如果小于阈值，权重则减小。这些权重随后被标准化，以便它们的和为1。
    weights = [(1 + cl_lambda) if difficulty > x else (1 - cl_lambda) for x in diff_list]
    weights.append(len(diff_list))
    new_weights = [i / sum(weights) for i in weights]
    return new_weights
#整体来看，这段代码配置了模型、数据加载器以及学习率和课程学习的权重调整策略。

#这个 clcr_train 函数是一个典型的深度学习训练过程，包括数据的加载、前向传播、损失计算、反向传播、参数更新、定期评估和模型保存。特别地，它还集成了课程学习策略，通过调整训练数据的权重来引导模型更有效地学习。
def clcr_train(train_model, train_loader, test_loader, optim, criterion):
    losses = []
    start_step = 0
    max_ssim = 0
    max_psnr = 0
    best_psnr = 0
    ssims = []
    psnrs = []
    initial_loss_weight = opt.loss_weight
    if opt.resume and os.path.exists(opt.model_dir):
        #print(f'resume from {opt.latest_model_dir}')
        print(f'resume from {opt.model_dir}')
        ckp = torch.load(opt.model_dir)
        print(opt.best_model_dir)
        losses = ckp['losses']
        start_step = ckp['step']
        max_ssim = ckp['max_ssim']
        max_psnr = ckp['max_psnr']
        psnrs = ckp['psnrs']
        ssims = ckp['ssims']
        weights = curriculum_weight(best_psnr)
        print(f'start_step:{start_step} start training ---')
    else:
        weights = curriculum_weight(0)
        print('train from scratch *** ')
        print(
            f'n1_weight:{weights[0]}| n2_weight:{weights[1]}| n3_weight:{weights[2]}| n4_weight:{weights[3]}| n5_weight:{weights[4]}|n6_weight:{weights[5]}|inp_weight:{weights[6]}')
    for step in range(start_step + 1, opt.steps + 1):
        train_model.train()
        #调整学习率，使用余弦衰减调度或保持初始设定。
        lr = opt.lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr
                #从训练加载器中获取一批数据，包括输入图像和负样本。
        x, y, n1, n2, n3, n4, n5, n6, inp = next(iter(train_loader))

        x = x.to(opt.device)
        y = y.to(opt.device)
        n1 = n1.to(opt.device)
        n2 = n2.to(opt.device)
        n3 = n3.to(opt.device)
        n4 = n4.to(opt.device)
        n5 = n5.to(opt.device)
        n6 = n6.to(opt.device)
        out = train_model(x)
        #使用模型进行前向传播并计算损失。
        pixel_loss = criterion[0](out, y)
        loss2 = 0
        loss = pixel_loss
        if opt.clcrloss:
            loss2 = criterion[1](out, y, n1, n2, n3, n4, n5, n6, x, weights)
        loss = pixel_loss + opt.loss_weight * loss2
        loss.backward()
        if opt.clip:
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), 0.2)
        optim.step()
        optim.zero_grad()
        for param in train_model.parameters():
            param.grad = None
        losses.append(loss.item())
        print(
            f'\rpixel loss : {pixel_loss.item():.5f}| cr loss : {opt.loss_weight * loss2.item():.5f}| step :{step}/{opt.steps}|lr :{lr :.7f} |time_used :{(time.time() - start_time) / 60 :.1f}',
            end='', flush=True)


         #每隔一定步数，在测试集上评估模型的性能。
        if step % opt.eval_step == 0:
            opt.loss_weight = initial_loss_weight
            with torch.no_grad():
                ssim_eval, psnr_eval = test(train_model, test_loader)
                print(f'\nstep :{step} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')

            with SummaryWriter(logdir=log_dir, comment=log_dir) as writer:
                writer.add_scalar('data/ssim', ssim_eval, step)
                writer.add_scalar('data/psnr', psnr_eval, step)
                writer.add_scalars('group', {
                    'ssim': ssim_eval,
                    'psnr': psnr_eval,
                    'loss': loss
                }, step)
            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)
            torch.save({
                'step': step,
                'ssims': ssims,
                'psnrs': psnrs,
                'losses': losses,
                'model': train_model.state_dict(),
                'weight': weights
            }, opt.latest_model_dir)
            #如果达到新的最佳性能，则更新记录并保存模型。
            if ssim_eval > max_ssim and psnr_eval > max_psnr:
                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)
                weights = curriculum_weight(max_psnr)
                torch.save({
                    'step': step,
                    'max_psnr': max_psnr,
                    'max_ssim': max_ssim,
                    'ssims': ssims,
                    'psnrs': psnrs,
                    'losses': losses,
                    'model': train_model.state_dict(),
                    'weight': weights
                }, opt.model_dir)
                print(f'\n model saved at step :{step}| max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}')
            # print(
            #     f'n1_weight:{weights[0]}| n2_weight:{weights[1]}| n3_weight:{weights[2]}| n4_weight:{weights[3]}| n5_weight:{weights[4]}|n6_weight:{weights[5]}|inp_weight:{weights[6]}')
            np.save(f'./numpy_files/{model_name}_{opt.steps}_losses.npy', losses)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_ssims.npy', ssims)
    np.save(f'./numpy_files/{model_name}_{opt.steps}_psnrs.npy', psnrs)


#整体而言，这段代码展示了从设置数据加载器、模型、优化器和损失函数，到执行模型训练和评估的整个过程。这是深度学习项目中典型的工作流程。
def test(test_model, loader_test):
    test_model.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []
    for i, (inputs, targets) in enumerate(loader_test):
        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)
        pred = test_model(inputs)
        ssim1 = ssim(pred, targets).item()
        psnr1 = psnr(pred, targets)
        ssims.append(ssim1)
        psnrs.append(psnr1)
        return np.mean(ssims), np.mean(psnrs)


if __name__ == "__main__":
    loader_train = loaders_[opt.trainset]
    loader_test = loaders_[opt.testset]
    net = models_[opt.net]
    net = net.to(opt.device)
    pytorch_total_params = sum(p.nelement() for p in net.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(pytorch_total_params / 1e6))
    if opt.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True
    criterion = [nn.L1Loss().to(opt.device)]
    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.lr, betas=(0.9, 0.999),
                           eps=1e-08)
    optimizer.zero_grad()
    if opt.clcrloss:
        criterion.append(C2R().to(opt.device))
    clcr_train(net, loader_train, loader_test, optimizer, criterion)
