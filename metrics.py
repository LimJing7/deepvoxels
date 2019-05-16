import scipy.misc
from PIL import Image
import math
import numpy as np
import cv2 as cv
import glob
import re
from skimage.measure import compare_ssim
from skimage.measure import compare_psnr

true_img_path = "/Users/gaoxian/Downloads/CS231nProject/22654a0cc1ba382c4eace6243432632/rgb/*.png"
# gen_img_path = "/Users/gaoxian/Downloads/CS231nProject/test_traj_mod/05_15/08-28-48_14-46-49_skip_2_200.00_l1_weight_2_trgt__22654a0cc1ba382c4eace6243432632_model-epoch_119_iter_10000.pth_22654a0cc1ba382c4eace6243432632/*.png"
gen_img_path = "/Users/gaoxian/Downloads/CS231nProject/test_traj/05_16/06-02-00_14-46-49_skip_2_200.00_l1_weight_2_trgt__22654a0cc1ba382c4eace6243432632_model-epoch_238_iter_20000.pth_22654a0cc1ba382c4eace6243432632/*.png"


true_img_list = glob.glob(true_img_path)
gen_img_list = glob.glob(gen_img_path)

psnr_val = 0.0
ssim_val = 0.0

i = 0
for img in gen_img_list:
    if ((i % 3) == 0):
        gen_img = Image.open(img)
        gen_img = gen_img.resize((512, 512), Image.BICUBIC)
        gen_img = np.asarray(gen_img)
        # img_name = re.search("08-28-48_14-46-49_skip_2_200.00_l1_weight_2_trgt__22654a0cc1ba382c4eace6243432632_model-epoch_119_iter_10000.pth_22654a0cc1ba382c4eace6243432632/(.*).png", img, re.IGNORECASE)
        img_name = re.search("/Users/gaoxian/Downloads/CS231nProject/test_traj/05_16/06-02-00_14-46-49_skip_2_200.00_l1_weight_2_trgt__22654a0cc1ba382c4eace6243432632_model-epoch_238_iter_20000.pth_22654a0cc1ba382c4eace6243432632/(.*).png", img, re.IGNORECASE) 
        if (img_name):
            img_name = img_name.group(1)
            img_name = "0" + img_name[4:]
            original_img = scipy.misc.imread("/Users/gaoxian/Downloads/CS231nProject/22654a0cc1ba382c4eace6243432632/rgb/" + str(img_name) + ".png")
            psnr_val += compare_psnr(original_img[:, :, :3], gen_img)
            ssim_val += compare_ssim(original_img[:, :, :3], gen_img, multichannel=True)
    print(i)
    i += 1

print("mean ssim_score: ", ssim_val / len(true_img_list))
print("mean psnr_score: ", psnr_val / len(true_img_list))