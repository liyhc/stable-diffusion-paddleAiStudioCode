from ppdiffusers.utils import load_image
from PIL import Image
import numpy as np
import einops
def img2np(img):
    # h w c -> c h w
    return einops.rearrange(np.asarray(img), "h w c -> c h w")
def np2img(array):
    # c h w  -> h w c
    return Image.fromarray(einops.rearrange(array, "c h w  -> h w c"))

def ldm_scale(pipe,img_path,num_inference_steps=10,patch_wh=128,save_path=None):
    image = load_image(img_path)
    # image = image.resize((int(image.size[0]/2),int(image.size[1]/2)))

    image = img2np(image)
    c, h, w = image.shape #c h w
    patch_wh = patch_wh
    h_list_len = int(h/patch_wh) if h % patch_wh==0 else (int(h/patch_wh) +1)
    w_list_len = int(w/patch_wh) if w % patch_wh==0 else (int(w/patch_wh) +1)
    # print(h_list_len,w_list_len)
    img_c = np.ones((c,h_list_len*patch_wh,w_list_len*patch_wh),dtype= np.uint8) 
    img_c[:,:h,:w] = image
    h = int(h/patch_wh)*patch_wh if h % patch_wh<64 else (int(h/patch_wh) +1)*patch_wh
    w = int(w/patch_wh)*patch_wh if w % patch_wh<64 else (int(w/patch_wh) +1)*patch_wh
    img_c  = img_c[:,:h,:w]
    new_image  = np.ones((c,h_list_len*patch_wh*4,w_list_len*patch_wh*4),dtype= np.uint8)


    patch_h_list = [i*patch_wh for i in range(int(h/patch_wh)+1)]
    patch_w_list = [i*patch_wh for i in range(int(w/patch_wh)+1)]

    # print(patch_h_list,patch_w_list)
    
    for h_i in range(len(patch_h_list)-1):
        for w_i in range(len(patch_w_list)-1):
            h_up =  patch_h_list[h_i]
            h_down = patch_h_list[h_i+1]
            w_up =  patch_w_list[w_i]
            w_down = patch_w_list[w_i+1]
            patch = img_c[:,h_up:h_down,w_up:w_down]
            new_patch = img2np(pipe(image =np2img(patch) ,num_inference_steps=num_inference_steps).images[0])
            # print(patch.shape,new_patch.shape)
            new_image[:,h_up*4:h_down*4 ,w_up*4:w_down*4] = new_patch
        #     break
        # break
    super_img = np2img(new_image[:,:h*4,:w*4])
    if(save_path is not None):
        super_img.save(save_path)
    print("new_image-shape",new_image.shape)
    return super_img
