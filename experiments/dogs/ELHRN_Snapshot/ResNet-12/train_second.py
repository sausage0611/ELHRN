import os
import sys
import torch
import yaml
from functools import partial
sys.path.append('../../../../')
from trainers import trainer_snapshot_second, elhrn_train
from datasets import dataloaders
from models.ELHRN_Snapshot import ELHRN_Snapshot

args = trainer_snapshot_second.train_parser()
with open('../../../../config.yml', 'r') as f:
    temp = yaml.safe_load(f)
data_path = os.path.abspath(temp['data_path'])
fewshot_path = os.path.join(data_path,'dogs')


model_path = './2000_model_Conv-4.pth'


pm = trainer_snapshot_second.Path_Manager(fewshot_path=fewshot_path,args=args)


gpu = 0
torch.cuda.set_device(gpu)
autoparm_model = ELHRN_Snapshot(resnet=False)
autoparm_model.cuda()
autoparm_model.load_state_dict(torch.load(model_path,map_location=util.get_device_map(gpu)),strict=True)

for name, param in autoparm_model.named_parameters():
    if 'autoParmFive' not in name:
        param.requires_grad = False
for name, param in autoparm_model.named_parameters():
    if 'scale' in name:
        param.requires_grad = True

for name, param in autoparm_model.named_parameters():
    print(name, param.requires_grad)



way1=5
shot1=[1,5,10,15,20,25,30]
way2=[10,15,20,25,30]
shot2=5


train_loader = dataloaders.meta_autoparm_train_dataloader(data_path=pm.train,
                                                way1=way1,
                                                shot1=shot1,
                                                way2=way2,
                                                shot2=shot2,
                                                transform_type=args.train_transform_type)

train_func = partial(elhrn_train.auto_train,train_loader=train_loader)


auto_tm = trainer_snapshot_second.Parm_Train_Auto_Manager(args,path_manager=pm,train_func=train_func)


auto_tm.train(autoparm_model)




