from random import shuffle
import numpy as np
from connect_loss import connect_loss,Bilateral_voting
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.nn import CosineSimilarity
from lr_update import get_lr
from SLITNet_utils import calculate_hausdorff_metric
from metrics.cldice import clDice
import os
from medpy.metric import binary
# from apex import amp
import sklearn
import torchvision.utils as utils
from sklearn.metrics import precision_score
from skimage.io import imread, imsave



class Solver(object):
    def __init__(self, args,optim=torch.optim.Adam):
        self.args = args
        self.optim = optim
        self.NumClass = self.args.num_class
        self.lr = self.args.lr
        H,W = args.resize

        ########## check
        self.hori_translation = torch.zeros([1,self.NumClass,W,W])
        for i in range(W-1):
            self.hori_translation[:,:,i,i+1] = torch.tensor(1.0)
        self.verti_translation = torch.zeros([1,self.NumClass,H,H])
        for j in range(H-1):
            self.verti_translation[:,:,j,j+1] = torch.tensor(1.0)
        self.hori_translation = self.hori_translation.float()
        self.verti_translation = self.verti_translation.float()

    def create_exp_directory(self,exp_id):
        if not os.path.exists('models/' + str(exp_id)):
            os.makedirs('models/' + str(exp_id))

        csv = 'results_'+str(exp_id)+'.csv'
        with open(os.path.join(self.args.save, csv), 'w') as f:
            f.write('epoch, dice, reflex, scar, thinning, pupil, SI, inflammation, hypopyon \n')



    def get_density(self, pos_cnt,bins = 50):

        val_in_bin_ = [[],[],[]]
        density_ = [[],[],[]]
        bin_wide_ = []

        ### check
        for n in range(3):
            density = []
            val_in_bin = []
            c1 = [i for i in pos_cnt[n] if i != 0]
            c1_t = torch.tensor(c1)
            bin_wide = (c1_t.max()+50)/bins
            bin_wide_.append(bin_wide)

            edges = torch.arange(bins + 1).float()*bin_wide
            for i in range(bins):
                val = [c1[j] for j in range(len(c1)) if ((c1[j] >= edges[i]) & (c1[j] < edges[i + 1]))]
                # print(val)
                val_in_bin.append(val)
                inds = (c1_t >= edges[i]) & (c1_t < edges[i + 1]) #& valid
                num_in_bin = inds.sum().item()
                # print(num_in_bin)
                density.append(num_in_bin)

            denominator = torch.tensor(density).sum()
            # print(val_in_bin)

            #### get density ####
            density = torch.tensor(density)/denominator
            density_[n]=density
            val_in_bin_[n] = val_in_bin
        print(density_)

        return density_, val_in_bin_,bin_wide_



    def train(self, model, train_loader, val_loader,exp_id, num_epochs=10):

        #### lr update schedule
        # gamma = 0.5
        # step_size = 10
        optim = self.optim(model.parameters(), lr=self.lr)
        # scheduler = lr_scheduler.MultiStepLR(optim, milestones=[12,24,35],
        #                                 gamma=gamma)  # decay LR by a factor of 0.5 every 5 epochs
        ####

        print('START TRAIN.')

        
        self.create_exp_directory(exp_id)


        #### check
        if self.args.use_SDL:
            pos_cnt = np.load(self.args.weights+'training_positive_pixel_'+str(exp_id)+'.npy',allow_pickle=True)
            density, val_in_bin,bin_wide = self.get_density(pos_cnt)
            self.loss_func=connect_loss(self.args,self.hori_translation,self.verti_translation, density,bin_wide)
        else:
            self.loss_func=connect_loss(self.args,self.hori_translation,self.verti_translation)

        # net, optimizer = amp.initialize(model, optim, opt_level='O2')
        

        best_p = 0
        best_epo, best_SI_epo, best_WBC_epo, best_hypopyon_epo = 0, 0, 0, 0
        best_SI = 0
        best_WBC = 0
        best_hypopyon = 0

        scheduled = ['CosineAnnealingWarmRestarts']
        if self.args.lr_update in scheduled:
            scheduled = True
            if self.args.lr_update == 'CosineAnnealingWarmRestarts':
                scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min = 0.00001)
        else:
            scheduled = False
        # self.test_epoch(model,val_loader,0,exp_id)
        cos_loss = CosineSimilarity()
        for epoch in range(self.args.epochs):
            model.train()

            if scheduled:
                scheduler.step()
            else:
                curr_lr = get_lr(self.lr,self.args.lr_update, epoch, num_epochs, gamma=self.args.gamma,step=self.args.lr_step)
                for param_group in optim.param_groups:
                    param_group['lr'] = curr_lr
            

            for i_batch, sample_batched in enumerate(train_loader):
                loss_skd = 0
                X = Variable(sample_batched[0])
                y = Variable(sample_batched[1])
                # limbus_mask = Variable(sample_batched[2])

                X= X.cuda()
                # limbus_mask = limbus_mask.cuda()
                y = y.float().cuda()
                # print(X.shape,y.shape,limbus_mask.shape)

                # imsave('img.png',X[0].permute(1,2,0).cpu().data.numpy())
                # imsave('y.png',y[0,0].cpu().data.numpy())
                # print(k)
                optim.zero_grad()
                output, aux_out, students, teachers = model(X)

                
                loss_main = self.loss_func(output, y)
                loss_aux = self.loss_func(aux_out, y)

                for stu, tea in zip(students, teachers):
                    # print(stu.shape, tea.shape)
                    #stu, tea = F.normalize(stu), F.normalize(tea)
                    #D_stu = smat(stu, stu)
                    #D_tea = smat(tea, tea)
                    loss_skd += (1-cos_loss(stu, tea.detach()).mean())
                # print(loss_skd)
                loss =loss_main+0.3*loss_aux + 0.5*loss_skd
                # print(loss)
                loss.backward()
                # with amp.scale_loss(loss, optimizer) as scale_loss:
                #     scale_loss.backward()

                optim.step()
                
                print('[epoch:'+str(epoch)+'][Iteration : ' + str(i_batch) + '/' + str(len(train_loader)) + '] Total:%.3f' %(
                    loss.item()))



            dice_p, SI_dice, WBC_dice, hypopyon_dice = self.test_epoch(model,val_loader,epoch,exp_id)

            if best_p<dice_p:
                best_p = dice_p
                best_epo = epoch
                torch.save(model.state_dict(), 'models/' + str(exp_id) + '/best_model.pth')

            if best_SI<SI_dice:
                best_SI = SI_dice
                best_SI_epo = epoch
                torch.save(model.state_dict(), 'models/' + str(exp_id) + '/best_SI_model.pth')

            if best_WBC<WBC_dice:
                best_WBC = WBC_dice
                best_WBC_epo = epoch
                torch.save(model.state_dict(), 'models/' + str(exp_id) + '/best_WBC_model.pth')

            if best_hypopyon<hypopyon_dice:
                best_hypopyon = hypopyon_dice
                best_hypopyon_epo = epoch
                torch.save(model.state_dict(), 'models/' + str(exp_id) + '/best_hypop_model.pth')

            if (epoch+1) % self.args.save_per_epochs == 0:
                torch.save(model.state_dict(), 'models/' + str(exp_id) + '/'+str(epoch+1)+'_model.pth')
            print('[Epoch :%d] total loss:%.3f ' %(epoch,loss.item()))

            # if epoch%self.args.save_per_epochs==0:
            #     torch.save(model.state_dict(), 'models/' + str(exp_id) + '/epoch' + str(epoch + 1)+'.pth')
        csv = 'results_'+str(exp_id)+'.csv'
        with open(os.path.join(self.args.save, csv), 'a') as f:
            f.write('%03d,%03d,%03d,%03d,%0.6f \n' % (
                best_epo,
                best_SI_epo,
                best_WBC_epo,
                best_hypopyon_epo,
                best_p
            ))
        # writer.close()
        print('FINISH.')
        
    def test_epoch(self,model,loader,epoch,exp_id):
        model.eval()
        self.dice_ls = []
        self.Jac_ls=[]
        self.cldc_ls = []

        self.scar_dsc_ls = []
        self.thin_dsc_ls = []
        hd_ls = []
        with torch.no_grad(): 
            for j_batch, test_data in enumerate(loader):
                curr_dice = []
                X_test = Variable(test_data[0])
                y_test = Variable(test_data[1])
                # limbus_mask = test_data[2]

                X_test= X_test.cuda()
                # limbus_mask = limbus_mask.cuda()
                y_test = y_test.long().cuda()

                # if torch.max(limbus_mask) == 0:
                #     continue
                output_test,_, _, _ = model(X_test)
                batch,channel,H,W = X_test.shape

                #### check ###
                hori_translation = self.hori_translation.repeat(batch,1,1,1).cuda()
                verti_translation = self.verti_translation.repeat(batch,1,1,1).cuda()

                

                # if self.args.num_class == 1: ## check
                output_test = F.sigmoid(output_test)
                class_pred = output_test.view([batch,-1,8,H,W])
                # print(class_pred.shape)
                pred = torch.where(class_pred>0.5,1,0)
                pred,_ = Bilateral_voting(pred.float(),hori_translation,verti_translation)
                # pred = F.interpolate(pred, [512,768])
                # else:
                #     class_pred = output_test.view([batch,-1,8,H,W])
                #     final_pred,_ = Bilateral_voting(class_pred,hori_translation,verti_translation)
                #     pred = get_mask(final_pred)
                #     pred = self.one_hot(pred, X_test.shape)
                


                scar_pred = pred[:,1].unsqueeze(1)
                scar_gt = y_test[:,1].unsqueeze(1)

                thin_pred = pred[:,2].unsqueeze(1)
                thin_gt = y_test[:,2].unsqueeze(1)

                index = [0,3,4,5,6]
                balanced_pred = pred[:,index]
                balanced_gt = y_test[:,index]

                dice,Jac = self.per_class_dice(balanced_pred,balanced_gt)

                class_hd = calculate_hausdorff_metric(pred=balanced_pred,
                                                           truth=balanced_gt,
                                                           num_classes=5)
                hd_ls.append(class_hd)
                if scar_gt.max() != 0:
                    dice_scar,_ = self.per_class_dice(scar_pred,scar_gt)
                    self.scar_dsc_ls.append(dice_scar[0,0].cpu().data.numpy())
                if thin_gt.max() != 0:
                    dice_thin,_ = self.per_class_dice(thin_pred,thin_gt)
                    self.thin_dsc_ls.append(dice_thin[0,0].cpu().data.numpy())

                if self.args.num_class == 1:
                    pred_np = pred.squeeze().cpu().numpy()
                    target_np = y_test.squeeze().cpu().numpy()
                    # cldc = clDice(pred_np,target_np)
                    # self.cldc_ls.append(cldc)

                ###### notice: for multi-class segmentation, the self.dice_ls calculated following exclude the background (BG) class

                if self.args.num_class>1:
                    self.dice_ls += dice.tolist() ## use self.dice_ls += torch.mean(dice,1).tolist() if you want to include BG

                    # self.Jac_ls += torch.mean(Jac[:,1:],1).tolist() ## same as above
                else:
                    self.dice_ls += dice[:,0].tolist()
                    # self.Jac_ls += Jac[:,0].tolist()

                if j_batch%(max(1,int(len(loader)/5)))==0:
                    print('[Iteration : ' + str(j_batch) + '/' + str(len(loader)) + '] Total DSC:%.3f ' %(
                        np.mean(self.dice_ls)))

            # print(len(self.Jac_ls))
            # Jac_ls =np.array(self.Jac_ls)
            # print(self.dice_ls)
            # print(self.scar_dsc_ls)
            # print(self.thin_dsc_ls)
            # dice_ls = np.array(self.dice_ls)
            class_dice = np.mean(self.dice_ls,0).tolist() 
            # print(np.mean(self.scar_dsc_ls))
            class_dice.append(np.mean(self.scar_dsc_ls))
            class_dice.append(np.mean(self.thin_dsc_ls))
            

            # print(class_dice)
            hd_ls = np.array(hd_ls)
            class_hd = np.nanmean(hd_ls,1)
            total_dice = np.mean(class_dice[:5])
            csv = 'results_'+str(exp_id)+'.csv'
            with open(os.path.join(self.args.save, csv), 'a') as f:
                f.write('%03d,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f,%0.6f \n' % (
                    (epoch + 1),
                    total_dice,
                    class_dice[0],
                    class_dice[5],
                    class_dice[6],
                    class_dice[1],
                    class_dice[2],
                    class_dice[3],
                    class_dice[4],
                    class_hd[0],
                    class_hd[1],
                    class_hd[2],
                    class_hd[3],
                    class_hd[4],


                ))

            return total_dice, class_dice[2], class_dice[3], class_dice[4]

    # def evaluate(self,pred,gt):
    #     scar_pred = pred[:,1]
    #     scar_gt = gt[:,1]

    #     thin_pred = pred[:,2]
    #     thin_gt = gt[:,2]

    #     index = [0,3,4,5,6]
    #     dsc_pred = pred[:,index]
    #     dsc_gt = gt[:,index]
    #     print(dsc_gt.shape,pos_dsc_gt.shape)
    #     # dice,_ = self.per_class_dice(pred,gt)

    def per_class_dice(self,y_pred, y_true):
        eps = 0.0001

        FN = torch.sum((1-y_pred)*y_true,dim=(2,3)) 
        FP = torch.sum((1-y_true)*y_pred,dim=(2,3)) 
        Pred = y_pred
        GT = y_true
        inter = torch.sum(GT* Pred,dim=(2,3)) 


        union = torch.sum(GT,dim=(2,3)) + torch.sum(Pred,dim=(2,3)) 
        dice = (2*inter+eps)/(union+eps)
        Jac = (inter+eps)/(inter+FP+FN+eps)

        return dice, Jac

    def one_hot(self,target,shape):

        one_hot_mat = torch.zeros([shape[0],self.args.num_class,shape[2],shape[3]]).cuda()
        target = target.cuda()
        one_hot_mat.scatter_(1, target, 1)
        return one_hot_mat

def get_mask(output):
    output = F.softmax(output,dim=1)
    _,pred = output.topk(1, dim=1)
    #pred = pred.squeeze()
    
    return pred




