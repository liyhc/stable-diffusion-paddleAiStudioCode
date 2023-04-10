# stable-diffusion-paddleAiStudioCode

## 使用LDM图片放大
```
import paddle
from ppdiffusers import LDMSuperResolutionPipeline
from ppdiffusers.utils import load_image

pipe = LDMSuperResolutionPipeline.from_pretrained("CompVis/ldm-super-resolution-4x-openimages")
#原图
img_path = '/home/aistudio/2.png'
display(load_image(img_path))

#放大
from ldm_scale_super import ldm_scale
display(ldm_scale(pipe,img_path,num_inference_steps=10,patch_wh=128,save_path='222.png'))
```
