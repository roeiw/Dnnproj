import os
import math
from decimal import Decimal
import matplotlib.pyplot as plt
import utility

import torch
from torch.autograd import Variable
from tqdm import tqdm


class Trainer():
    def __init__(self, args, test_loader,train_loader, my_model, my_loss, ckp, path):
        self.args = args
        self.noise_g = args.noise_g

        self.ckp = ckp
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args,self.model)
        # self.scheduler = utility.make_scheduler(args,self.optimizer)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,1,0.5)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.path = path

        # if self.args.load != '.':
        #     self.optimizer.load_state_dict(
        #         torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
        #     )
        #     for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        self.model.train()
        for epoch in range(3):
            timer_data, timer_model = utility.timer(), utility.timer()
            lr = 1e-4
            running_loss = 0
            self.ckp.write_log(
                '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
            for batch, data in enumerate(self.train_loader):
                gt_image = data['GT_image']
                noisy_image = data['NOISY_image']
                name = data['image_name']
                self.optimizer.zero_grad()

                # plt.imshow(lr.permute(1,2,0))
                timer_data.hold()
                timer_model.tic()
                pred_image = self.model(noisy_image)
                loss = self.loss(pred_image, gt_image)
                # if loss.item() < self.args.skip_threshold * self.error_last:
                loss.backward()
                self.optimizer.step()
                # else:
                #     print('Skip this batch {}! (Loss: {})'.format(
                #         batch + 1, loss.item()
                #     ))
                running_loss += loss.item()*gt_image.size(0)
                timer_model.hold()
                if batch % 2000 == 0:
                    print("Loss on batch: ", batch, " is: ", loss.item())


            self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                (epoch + 1) ,
                len(self.train_loader.dataset),
                running_loss/len(self.train_loader.dataset),
                timer_model.release(),
                timer_data.release()))

            timer_data.tic()
            self.scheduler.step()
        torch.save(self.model.state_dict(), self.path)

def test(self):

    epoch = self.scheduler.last_epoch + 1
    self.ckp.write_log('\nEvaluation:')

    self.model.eval()

    timer_test = utility.timer()
    with torch.no_grad():
        # running_loss=0
        eval_acc = 0
        tqdm_test = tqdm(self.loader_test, ncols=80)
        for i,(gt_image,noisy_image,name) in enumerate(tqdm_test):
            # no_eval = (hr.nelement() == 1)
            # if not no_eval:
            #     lr, hr = self.prepare([lr, hr])
            # else:
            #     lr = self.prepare([lr])[0]

            pred_image = self.model(noisy_image)
            # sr = utility.quantize(sr, self.args.rgb_range)

            save_list = [pred_image]
            eval_acc += utility.calc_psnr(
                pred_image, gt_image,  self.args.rgb_range,
            )
            save_list.extend([noisy_image, gt_image])

        # if self.args.save_results:
        #     self.ckp.save_results(filename, save_list, scale)

    # self.ckp.log[-1, idx_scale] = eval_acc / len(self.loader_test)
    # best = self.ckp.log.max(0)
    # self.ckp.write_log(
    #     '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
    #         self.args.data_test,                  )
    # )

#
# self.ckp.write_log(
#     'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
# )
# if not self.args.test_only:
#     self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

def prepare(self, l, volatile=False):
    device = torch.device('cpu' if self.args.cpu else 'cuda')

    def _prepare(tensor):
        if self.args.precision == 'half': tensor = tensor.half()
        return tensor.to(device)

    return [_prepare(_l) for _l in l]

def terminate(self):
    if self.args.test_only:
        self.test()
        return True
    else:
        epoch = self.scheduler.last_epoch + 1
        return epoch >= self.args.epochs

