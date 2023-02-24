#%%
import torch
import numpy as np
import matplotlib.pyplot as plt

import mmcv
from mmseg.apis import init_model, inference_model, show_result_pyplot
from mmseg.utils import register_all_modules
register_all_modules()

# Cityscapes数据集
# Cityscapes语义分割数据集：https://www.cityscapes-dataset.com

from PIL import Image

img_path = '../data/street_uk.jpeg'
img_pil = Image.open(img_path)

#%%

# 模型 config 配置文件
config_file = "../configs/pspnet/pspnet_r50-d8_4xb2-40k_cityscapes-512x1024.py "
# config_file = '../configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py'

# 模型 checkpoint 权重文件
checkpoint_file = '../checkpoints/pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
# checkpoint_file = '../checkpoints/mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024_20221202_141901-dc2c2ddd.pth'

from mmseg.apis import init_model
model = init_model(config_file, checkpoint_file, device='cuda:0')

#%%
# 运行语义分割预测
from mmseg.apis import inference_model
from mmengine.model.utils import revert_sync_batchnorm
if not torch.cuda.is_available():
    model = revert_sync_batchnorm(model)

result = inference_model(model, img_path)

# 类别：0-18，共 19 个 类别
print(result.pred_sem_seg.data.shape)
print(np.unique(result.pred_sem_seg.data.cpu()))

class_map = result.pred_sem_seg.data[0].detach().cpu().numpy()
plt.imshow(class_map)
plt.show()

# 置信度
print(result.seg_logits.data.shape)

#%%
# 可视化语义分割预测结果-方法一
# from mmseg.apis import show_result_pyplot
# visualization = show_result_pyplot(model, img_path, result, opacity=0.8, title='MMSeg', out_file='outputs/B2.jpg')
# plt.imshow(mmcv.bgr2rgb(visualization))
# plt.show()

#%%
## 可视化语义分割预测结果-方法二
from mmseg.datasets import cityscapes
import numpy as np
import mmcv

# 获取类别名和调色板
classes = cityscapes.CityscapesDataset.METAINFO['classes']
palette = cityscapes.CityscapesDataset.METAINFO['palette']
opacity = 0.15 # 透明度，越大越接近原图

# 将分割图按调色板染色
# seg_map = result[0].astype('uint8')
seg_map = class_map.astype('uint8')
seg_img = Image.fromarray(seg_map).convert('P')
seg_img.putpalette(np.array(palette, dtype=np.uint8))

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
plt.figure(figsize=(14, 8))
im = plt.imshow(((np.array(seg_img.convert('RGB')))*(1-opacity) + mmcv.imread(img_path)*opacity) / 255)

# 为每一种颜色创建一个图例
patches = [mpatches.Patch(color=np.array(palette[i])/255., label=classes[i]) for i in range(18)]
plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')

plt.show()








