import sys
import os
import copy
import torch
import yaml
sys.path.append('../../../../')
from models.ELHRN_Snapshot import ELHRN_Snapshot
from utils import util
from trainers.eval_snapshot_first import meta_test


with open('../../../../config.yml', 'r') as f:
    temp = yaml.safe_load(f)
data_path = os.path.abspath(temp['data_path'])

test_path = os.path.join(data_path,'cars_196/test_pre')


gpu = 0
torch.cuda.set_device(gpu)


stage = 5
epoch = 400


model = ELHRN_Snapshot(resnet=False)
model.cuda()
models = [copy.deepcopy(model) for i in range(stage)]

[models[i].load_state_dict(torch.load('model_'+str(epoch*(i+1))+'_AutoParm-auto-Conv-4.pth')) for i in range(stage)]

[m.eval() for m in models]



with torch.no_grad():
    way = 5
    for shot in [1,5]:
        mean,interval = meta_test(data_path=test_path,
                                models=models,
                                way=way,
                                shot=shot,
                                pre=True,
                                transform_type=None,
                                trial=10000)
        print('%d-way-%d-shot acc: %.2f\t%.2f'%(way,shot,mean,interval))