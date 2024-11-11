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

test_path = os.path.join(data_path,'dogs/test_pre')

model_path = './model_AutoParm-auto-Conv-4.pth'

gpu = 0
torch.cuda.set_device(gpu)
model = ELHRN(resnet=False)
model.cuda()
model.load_state_dict(torch.load(model_path,map_location=util.get_device_map(gpu)),strict=True)
model.eval()

with torch.no_grad():
    way1 = 5
    for shot1 in [1, 5, 10, 15, 20, 25, 30]:
        mean,interval = meta_test(data_path=test_path,
                                model=model,
                                way=way1,
                                shot=shot1,
                                pre=True,
                                transform_type=None,
                                trial=10000)
        print('%d-way-%d-shot acc: %.2f\t%.2f'%(way1,shot1,mean,interval))
    shot2 = 5
    for way2 in [10, 15, 20, 25, 30]:
        mean,interval = meta_test(data_path=test_path,
                                model=model,
                                way=way2,
                                shot=shot2,
                                pre=True,
                                transform_type=None,
                                trial=10000)
        print('%d-way-%d-shot acc: %.2f\t%.2f'%(way2,shot2,mean,interval))

