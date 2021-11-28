import os
import math
from decimal import Decimal
import matplotlib.pyplot as plt
import utility
import torch.nn as nn
import torch
from torch.autograd import Variable
from tqdm import tqdm
import logging


class Trainer():
    def __init__(self, args, val_loader,train_loader, my_model, my_loss, ckp,save_path):
        self.args = args

        self.ckp = ckp
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = my_model
        self.loss = my_loss
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=1e-4)
        # self.scheduler = utility.make_scheduler(args,self.optimizer)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,5000,0.6)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.path = save_path
        self.epochs = 3
        self.val_loss = 1000000
        print(self.path)
        logging.basicConfig(filename = "./../logs/"+ self.path.split('.pt')[0]+"_train.log", level=logging.INFO)

        # if self.args.load != '.':
        #     self.optimizer.load_state_dict(
        #         torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
        #     )
        #     for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def train(self):
        running_loss = 0
        number_of_batches = len(self.train_loader)
        for epoch in range(self.epochs):
            timer_data, timer_model = utility.timer(), utility.timer()

            self.model.train()

            for batch, data in enumerate(self.train_loader):

                gt_image = data['GT_image']
                noisy_image = data['NOISY_image']
                name = data['image_name']
                self.optimizer.zero_grad()

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
                self.scheduler.step()

                if (batch % 2000 == 0) and batch != 0 :
                    self.model.eval()
                    current_loss = 0
                    with torch.no_grad():
                        for val_batch, data in enumerate(self.val_loader):
                            gt_image = data['GT_image']
                            noisy_image = data['NOISY_image']
                            name = data['image_name']
                            val_pred  = self.model(noisy_image)

                            current_loss += self.loss(val_pred, gt_image).item()*gt_image.size(0)
                        self.model.train()

                        print("Val_Loss is: ", current_loss)

                        if current_loss<self.val_loss:
                            self.val_loss = current_loss
                            torch.save(self.model.state_dict(), self.path)
                            print("resaved")

                    print("Loss on batch: ", batch, " is: ", loss.item())

                if (batch % 100 == 0):
                    logging.info(str(batch)+" loss: " + str(running_loss/((number_of_batches*epoch)+(batch+1))))
            self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                (epoch + 1) ,
                len(self.train_loader.dataset),
                running_loss/len(self.train_loader.dataset),
                timer_model.release(),
                timer_data.release()))

            timer_data.tic()
        path_split = self.path.split('.pt')
        torch.save(self.model.state_dict(), path_split[0]+"_final.pt")


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

