import sys
import os
import torch
import yaml
sys.path.append('../../../../')
from models.ELHRN import ELHRN
from utils import util
from trainers.eval_first import meta_test


with open('../../../../config.yml', 'r') as f:
    temp = yaml.safe_load(f)
data_path = os.path.abspath(temp['data_path'])

test_path = os.path.join(data_path,'tiered_meta_iNat/test')
model_path = './model_AutoParm-auto-Conv-4.pth'

gpu = 0
torch.cuda.set_device(gpu)

model = ELHRN(resnet=False)
model.cuda()
model.load_state_dict(torch.load(model_path,map_location=util.get_device_map(gpu)),strict=True)
model.eval()


with torch.no_grad():
    way = 5
    for shot in [1, 5]:
        mean,interval = meta_test(data_path=test_path,
                                model=model,
                                way=way,
                                shot=shot,
                                pre=True,
                                transform_type=None,
                                trial=10000)
        print('%d-way-%d-shot acc: %.2f\t%.2f'%(way,shot,mean,interval))