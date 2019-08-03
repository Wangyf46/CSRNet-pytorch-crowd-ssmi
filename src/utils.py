import h5py
import torch
import shutil
import cv2
import math
import numpy as np
import scipy.io as io
import torch.nn.functional as F
import matplotlib.pylab as plt
from torch.autograd import Variable
from torchvision import transforms
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from scipy.signal import argrelextrema
from torch.nn.modules.loss import _Loss
from torch.nn.parameter import Parameter


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

def gaussian_kernel(size, sigma):
    x, y = np.mgrid[-size:size + 1, -size:size + 1] # todo
    kernel = np.exp(-0.5 * (x * x + y * y) / (sigma * sigma))
    kernel /= kernel.sum() # todo
    return kernel


class SSIM_Loss(_Loss):
    def __init__(self, in_channels, size = 5, sigma = 1.5, size_average = True):
        super(SSIM_Loss, self).__init__(size_average)
        self.in_channels = in_channels
        self.size = size
        self.sigma = sigma
        self.size_average = size_average

        kernel = gaussian_kernel(self.size, self.sigma) # todo
        self.kernel_size = kernel.shape # 11*11
        weight = np.tile(kernel, (in_channels, 1, 1, 1))   # todo
        self.weight = Parameter(torch.from_numpy(weight).float(), requires_grad = False)   # todo

    def forward(self, img1, img2):
        mean1 = F.conv2d(img1, self.weight, padding = self.size, groups = self.in_channels)
        mean2 = F.conv2d(img2, self.weight, padding = self.size, groups = self.in_channels)
        mean1_sq = mean1 * mean1
        mean2_sq = mean2 * mean2
        mean_12 = mean1 * mean2

        sigma1_sq = F.conv2d(img1 * img1, self.weight, padding = self.size, groups = self.in_channels) - mean1_sq # padding = 0
        sigma2_sq = F.conv2d(img2 * img2, self.weight, padding = self.size, groups = self.in_channels) - mean2_sq
        sigma_12 = F.conv2d(img1 * img2, self.weight, padding = self.size, groups = self.in_channels) - mean_12

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        ssim = ((2 * mean_12 + C1) * (2 * sigma_12 + C2)) / ((mean1_sq + mean2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        if self.size_average:
            out = 1 - ssim.mean()
        else:
            out = 1 - ssim.view(ssim.size(0), -1).mean(1)
        return out

class SANetLoss(torch.nn.Module):
    def __init__(self, in_channels, size = 5, sigma = 1.5, size_average = True):
        super(SANetLoss, self).__init__()
        self.ssim_loss = SSIM_Loss(in_channels, size, sigma, size_average)

    def forward(self, img1, img2):
        #print(img1.shape)
        height = img1.shape[2] # TODO
        width = img1.shape[3]  # TODO
        loss_c = self.ssim_loss(img1, img2)
        loss_e = torch.mean((img1 - img2) ** 2, dim = (0, 1, 2, 3))  # TODO dim?? MSE
        loss = torch.mul(torch.add(torch.mul(loss_c, 0.001), loss_e), height * width) #todo toch.mul ???
        return loss

class Frame_label:
    def __init__(self, num, frame, frame_diff):
        self.num = num
        self.frame = frame
        self.frame_diff = frame_diff

def gt_label(path):
    mat = io.loadmat(path.replace(".jpg", ".mat").replace("images", "ground_truth").replace("IMG_", "GT_IMG_"))
    gt = mat["image_info"][0, 0][0, 0][0]
    label_count = gt.shape[0]
    gt_file = h5py.File(path.replace(".jpg", ".h5").replace("images", "ground_truth"), "r")
    gt_density_map = np.asarray(gt_file["density"])
    return label_count, gt_density_map

def et_data(img, model):
    img = transform(img)
    img = Variable(img)
    img = img.cuda().unsqueeze(0)                        # torch.Size([1, 3, H, W])
    et_density_map = model(img)                          # torch.Size([1, 1, H, W])
    et_density_map = et_density_map.detach().cpu().numpy()
    et_density_map = et_density_map.squeeze(0).squeeze(0)
    return et_density_map

def visualization_save(path, img, gt_density_map, et_density_map, label_count, gt_count, et_count, PSNR, SSIM):
    label1 = "Label: " + str(label_count)
    label2 = "Count: " + str(round(gt_count, 2))
    label3 = "Estimated: " + str(round(et_count, 2))
    label4 = "PSNR: " + str(round(PSNR, 2))
    label5 = "SSIM: " + str(round(SSIM, 2))

    '''
    ## ndarray to PIL Image
    gt_density_map = Image.fromarray(gt_density_map).convert("RGB")
    et_density_map = Image.fromarray(et_density_map).convert("RGB")

    p1 = (30,30)
    p2 = (30,100)
    p3 = (30,170)
    label_visualization(label1, img, 50, p1)
    label_visualization(label2, gt_density_map, 50, p1)
    label_visualization(label3, et_density_map, 50, p1)
    label_visualization(label4, et_density_map, 50, p2)
    label_visualization(label5, et_density_map, 50, p3)

    ## PIL Image to ndarray
    gt_density_map = np.array(gt_density_map.convert('F'))
    et_density_map = np.array(et_density_map.convert('F'))
    '''

    ## save and display
    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.axis("off")
    plt.subplot(1,3,2)
    plt.imshow(gt_density_map, cmap = plt.cm.jet)
    plt.axis("off")
    plt.subplot(1,3,3)
    plt.imshow(et_density_map, cmap = plt.cm.jet)
    plt.axis("off")
    plt.savefig(path, bbox_inches = "tight")  # remove blank, only can save and cannot display
    #plt.show()
    plt.close()

    ## TODO debug
    dst = Image.open(path)
    plt.imshow(dst)
    plt.text(50, 140, label1, size = 6, alpha = 0.9)
    plt.text(225, 140, label2, size = 6, alpha = 0.9)
    plt.text(400, 140, label3, size = 6, alpha = 0.9)
    plt.text(400, 155, label4, size = 6, alpha = 0.9)
    plt.text(400, 170, label5, size = 6, alpha = 0.9)
    plt.axis("off")
    plt.savefig(path, bbox_inches = "tight")
    #plt.show()
    plt.close()


def get_cov(img1, img2, u1, u2):
    cov = np.sum((img1 - u1) * (img2 - u2))
    return cov / img1.size

def get_interpolation(et_density_map):
        shape1 = int(et_density_map.shape[1] * 8)           # w
        shape0 = int(et_density_map.shape[0] * 8)           # h
        et_density_map_interpolation = cv2.resize(et_density_map, (shape1, shape0), interpolation = cv2.INTER_LINEAR) / 64.0
        #et_density_map_interpolation = cv2.resize(et_density_map, (shape1, shape0)) / 64.0
        et_density_map_interpolation = 255 * et_density_map_interpolation / np.max(et_density_map_interpolation)
        return et_density_map_interpolation

def get_quality_psnr(gt_density_map, et_density_map):
    if gt_density_map.shape[1] != et_density_map.shape[1]:
       et_density_map = get_interpolation(et_density_map)
    diff = np.sum((abs(gt_density_map - et_density_map)) ** 2)
    mse = diff / gt_density_map.size
    PSNR = 20 * (math.log(10, 255 / np.sqrt(mse)))
    return PSNR, et_density_map

def get_quality_ssim(gt_density_map, et_density_map_interpolation):
    '''ssim: 0-1'''
    K1 = 0.01
    K2 = 0.03
    L = 255
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    C3 = C2 / 2
    u_gt = np.mean(gt_density_map)  # gt mean value
    u_et = np.mean(et_density_map_interpolation)
    sd_gt = np.std(gt_density_map)                # gt standard deviation
    sd_et = np.std(et_density_map_interpolation)
    c_gt_et = get_cov(gt_density_map, et_density_map_interpolation, u_gt, u_et)            # gt and et covariance
    SSIM = ((2 * u_gt * u_et + C1) * (c_gt_et + C2)) / ((u_gt ** 2 + u_et ** 2 + C1) * (sd_et ** 2 + sd_gt ** 2 + C2))
    return SSIM

def label_visualization(label, img, font_size, locations):
    font = ImageFont.truetype("/home/wangyf/testfont.ttf", size = font_size)
    draw = ImageDraw.Draw(img)
    draw.text(locations, label, font = font, fill = "white")

def smooth(frames_diff, window_len, window = "hanning"):
    diff = np.array(frames_diff)
    s = np.r_[2 * diff[0] - diff[window_len:1:-1],
              diff, 2 * diff[-1] - diff[-1:-window_len:-1]]
    #print(len(s))
    w = getattr(np, window)(window_len)
    y = np.convolve(w / w.sum(), s, mode = "same")
    return y[window_len - 1:-window_len + 1]

def get_keyframe(len_window, cap):
    cur_frame = None
    pre_frame = None
    frames = []
    frames_diff = []
    ret, frame = cap.read()
    print(cap.isOpened())
    num = 0
    while(ret):
        cur_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LUV)
        if cur_frame is not None and pre_frame is not None:
            frame_diff = np.sum(cv2.absdiff(cur_frame, pre_frame))
            frames_diff.append(frame_diff)
            frame = Frame_label(num, frame, frame_diff)  # from 1 to last, not 0
            frames.append(frame)
        pre_frame = cur_frame
        ret, frame = cap.read()
        if ret != False:
           num = num + 1
        if num == 6000:
            break
    print("The total FrameNum is %d \n" % (num + 1))
    sm_diff_array = smooth(frames_diff, len_window)
    frame_indexes = np.asarray(argrelextrema(sm_diff_array, np.greater))[0]
    cap.release()
    return frame_indexes, frames

def get_frame(cap):
    ret, frame = cap.read()
    print(cap.isOpened())
    frames = []
    frame_indexes = []
    num = 0
    while(ret):
        if num == 1000:
            break
        elif num % 25 ==0:
             frame = Frame_label(num, frame, 0)
             frames.append(frame)
             frame_indexes.append(num - 1)
        num = num + 1
        ret, frame = cap.read()
    return frame_indexes, frames

def save_checkpoint(state, is_best,task_id, filename = 'checkpoint.pth.tar'):
    torch.save(state, task_id + filename)
    if is_best:
        shutil.copyfile(task_id + filename, task_id + 'model_best.pth.tar')

def cross(x1,y1, x2,y2, x3,y3):
    var1 = x2 - x1
    var2 = y2 - y1

    var3 = x3 - x1
    var4 = y3 - y1
    var = var1 * var4 - var3 * var2
    return var

def intersect(l1, p1, p2, flags = False):
    x1, y1 = l1[0], l1[1]
    x2, y2 = l1[2], l1[3]

    x3, y3 = p1
    x4, y4 = p2

    ## l1
    if x1 >= x2:
        l1_x_max = x1
        l1_x_min = x2
    else:
        l1_x_max = x2
        l1_x_min = x1

    if y1 >= y2:
        l1_y_max = y1
        l1_y_min = y2
    else:
        l1_y_max = y2
        l1_y_min = y1

    ## l2
    if x3 >= x4:
        l2_x_max = x3
        l2_x_min = x4
    else:
        l2_x_max = x4
        l2_x_min = x3

    if y3 >= y4:
        l2_y_max = y3
        l2_y_min = y4
    else:
        l2_y_max = y4
        l2_y_min = y3

    ##
    if l1_x_max >= l2_x_min and l2_x_max >= l1_x_min and l1_y_max >= l2_y_min and l2_y_max >= l1_y_min:
        if ((cross(x1,y1, x2,y2, x3,y3) * cross(x1,y1, x2,y2, x4,y4)) <= 0 and
            (cross(x3,y3, x4,y4, x1,y1) * cross(x3,y3, x4,y4, x2,y2) <=0)):
            flags = True
    return flags
