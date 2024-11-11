import os
import sys
import torch
import torch.optim as optim
import logging
import numpy as np
import argparse
from tqdm import tqdm
from tensorboardX import SummaryWriter
from .eval_second import meta_test
sys.path.append('..')



def get_logger(filename):

    formatter = logging.Formatter(
        "[%(asctime)s] %(message)s",datefmt='%m/%d %I:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(filename,"w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def train_parser():

    parser = argparse.ArgumentParser()

    ## general hyper-parameters
    parser.add_argument("--opt",help="optimizer",choices=['adam','sgd'])
    parser.add_argument("--lr",help="initial learning rate",type=float)
    parser.add_argument("--gamma",help="learning rate cut scalar",type=float,default=0.1)
    parser.add_argument("--epoch",help="number of epochs before lr is cut by gamma",type=int)
    parser.add_argument("--stage",help="number lr stages",type=int)
    parser.add_argument("--weight_decay",help="weight decay for optimizer",type=float)
    parser.add_argument("--gpu",help="gpu device",type=int,default=0)
    parser.add_argument("--seed",help="random seed",type=int,default=42)
    parser.add_argument("--val_epoch",help="number of epochs before eval on val",type=int,default=20)
    parser.add_argument("--resnet", help="whether use resnet12 as backbone or not",action="store_true")
    parser.add_argument("--nesterov",help="nesterov for sgd",action="store_true")
    parser.add_argument("--batch_size",help="batch size used during pre-training",type=int)
    parser.add_argument('--decay_epoch',nargs='+',help='epochs that cut lr',type=int)
    parser.add_argument("--pre", help="whether use pre-resized 84x84 images for val and test",action="store_true")
    parser.add_argument("--no_val", help="don't use validation set, just save model at final timestep",action="store_true")
    parser.add_argument("--train_way",help="training way",type=int)
    parser.add_argument("--test_way",help="test way",type=int,default=5)
    parser.add_argument("--train_shot",help="number of support images per class for meta-training and meta-testing during validation",type=int)
    parser.add_argument("--test_shot",nargs='+',help="number of support images per class for meta-testing during final test",type=int)
    parser.add_argument("--train_query_shot",help="number of query images per class during meta-training",type=int,default=15)
    parser.add_argument("--test_query_shot",help="number of query images per class during meta-testing",type=int,default=16)
    parser.add_argument("--train_transform_type",help="size transformation type during training",type=int)
    parser.add_argument("--test_transform_type",help="size transformation type during inference",type=int)
    parser.add_argument("--val_trial",help="number of meta-testing episodes during validation",type=int,default=1000)
    parser.add_argument("--detailed_name", help="whether include training details in the name",action="store_true")

    args = parser.parse_args()

    return args

def get_opt(model,args):

    if args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(model.parameters(),lr=args.lr,momentum=0.9,weight_decay=args.weight_decay,nesterov=args.nesterov)

    if args.decay_epoch is not None:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer,milestones=args.decay_epoch,gamma=args.gamma)

    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=args.epoch,gamma=args.gamma)

    return optimizer,scheduler


class Path_Manager:

    def __init__(self,fewshot_path,args):

        self.train = os.path.join(fewshot_path,'train')
        self.eigen_train = os.path.join(fewshot_path,'eigen_train')

        if args.pre:
            self.test = os.path.join(fewshot_path,'test_pre')
            self.val = os.path.join(fewshot_path,'val_pre') if not args.no_val else self.test

        else:
            self.test = os.path.join(fewshot_path,'test')
            self.val = os.path.join(fewshot_path,'val') if not args.no_val else self.test


class Parm_Train_Auto_Manager:

    def __init__(self,args,path_manager,train_func):

        seed = args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.set_device(args.gpu)

        if args.resnet:
            name = 'AutoParm-auto-ResNet-12'
        else:
            name = 'AutoParm-auto-Conv-4'

        if args.detailed_name:
            if args.decay_epoch is not None:
                temp = ''
                for i in args.decay_epoch:
                    temp += ('_'+str(i))

                suffix = '%s-lr_%.0e-gamma_%.0e-epoch_%d-drop%s-decay_%.0e-way_%d' % (args.opt,
                    args.lr,args.gamma,args.epoch,temp,args.weight_decay,args.train_way)
            else:
                suffix = '%s-lr_%.0e-gamma_%.0e-epoch_%d-stage_%d-decay_%.0e-way_%d' % (args.opt,
                    args.lr,args.gamma,args.epoch,args.stage,args.weight_decay,args.train_way)

            name = "%s-%s"%(name,suffix)

        self.logger = get_logger('%s.log' % (name))
        self.save_path = 'model_%s.pth' % (name)
        self.writer = SummaryWriter('log_%s' % (name))

        self.logger.info('display all the hyper-parameters in args:')
        for arg in vars(args):
            value = getattr(args,arg)
            if value is not None:
                self.logger.info('%s: %s' % (str(arg),str(value)))
        self.logger.info('------------------------')
        self.args = args
        self.train_func = train_func
        self.pm = path_manager

    def train(self,model):

        args = self.args
        train_func = self.train_func
        writer = self.writer
        save_path = self.save_path
        logger = self.logger
        optimizer,scheduler = get_opt(model,args)

        model.train()
        model.cuda()

        iter_counter = 0

        if args.decay_epoch is not None:
            total_epoch = args.epoch
        else:
            total_epoch = args.epoch*args.stage

        logger.info("start training!")

        combinations1 = [(5, 1), (5, 5), (5, 10), (5, 15), (5, 20), (5, 25), (5, 30),
                        (10, 5), (15, 5),(20,5),(25,5),(30,5)]


        best_loss_dict = {scenario: 2 for scenario in combinations1}



        for e in tqdm(range(total_epoch)):
            iter_counter,avg_acc_dict,avg_loss_dict = train_func(model=model,
                                                optimizer=optimizer,
                                                writer=writer,
                                                iter_counter=iter_counter)

            if (e+1)%args.val_epoch==0:
                logger.info("")
                logger.info("epoch %d/%d, iter %d:" % (e+1,total_epoch,iter_counter))
                model.eval()
                with torch.no_grad():
                    all_better_or_equal = True
                    print("avg_loss_dict:  ",avg_loss_dict)
                    print("best_loss_dict: ",best_loss_dict)
                    for (way, shot) in avg_loss_dict:

                        avg_loss = avg_loss_dict[(way, shot)]
                        best_loss = best_loss_dict[(way, shot)]


                        if avg_loss!=0 and avg_loss > best_loss + 0.05:
                            all_better_or_equal = False
                            break

                    if all_better_or_equal:
                        best_loss_dict = avg_loss_dict
                        best_epoch = e + 1
                        if not args.no_val:
                            torch.save(model.state_dict(), save_path)
                        logger.info('BEST!')

                model.train()
            scheduler.step()

        logger.info('training finished!')
        logger.info('------------------------')


