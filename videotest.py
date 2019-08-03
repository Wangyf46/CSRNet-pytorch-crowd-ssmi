import cv2
import numpy as np
import os
import torch
import time
from PIL import Image
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from matplotlib import pyplot as plt
from src.model import CSRNet
from src import utils


start = time.time()
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
model = CSRNet()
model = model.cuda()
checkpoint = torch.load('1model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])
model.cuda().eval()

len_window = 100
root = "/data2/crowd_task/video/"
#video_path = root + "1.mp4"
video_path = "/data2/aiit/2019_0109/03_2h.avi"
output_path = root + str(len_window) + '/'
exp_results = output_path + "exp_results.txt"
fd = open(exp_results, 'w')

def main():
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("fps {0}, size {1}".format(fps, size))

    #index, frames = utils.get_keyframe(len_window, cap)

    index, frames = utils.get_frame(cap)

    start = time.time()
    FrameNum = 0
    for i in index:
        for j in range(len(frames)):
            if (i + 1) == frames[j].num:
                save_path = output_path + "img_et_" + str(frames[j].num) + ".jpg"
                img = Image.fromarray(frames[j].frame).convert("RGB") #TODO RGB
                et_density_map = utils.et_data(img, model)
                et_count = et_density_map.sum()
                et_density_map = et_density_map * 255 / np.max(et_density_map)
                et_density_map_interpolation = utils.get_interpolation(et_density_map) ## TODO

                label = "Estimated: " + str(round(et_count, 2))

                '''
                save_path1 = output_path + "et_" + str(frames[j].num) + ".png"
                plt.figure()
                plt.imshow(et_density_map_interpolation, cmap = plt.cm.jet)
                plt.axis("off")
                plt.savefig(save_path1, bbox_inches = "tight", dpi = 80)
                #plt.show()

        
                p1 = (30, 40)
                font = ImageFont.truetype("/home/wangyf/testfont.ttf", size = 50)
                draw = ImageDraw.Draw(Image.fromarray(img).convert("RGB"))
                draw.text(p1, label, font = font, fill = "white")
                img = np.array(img.convert("L"))
                dst = cv2.imread(save_path1)
                output_img = np.vstack((img, dst))
                cv2.imwrite(save_path, output_img)
                cv2.imshow(save_path, output_img)
                cv2.waitKey(0)
                '''

                ## visualization and save
                '''
                label1 = "Estimated: " + str(round(et_count, 2))
                et_density_map_interpolation = Image.fromarray(et_density_map_interpolation).convert("RGB")
                p1 = (30, 40)
                utils.label_visualization(label1, et_density_map_interpolation, 50, p1)
                et_density_map_interpolation = np.array(et_density_map_interpolation.convert('F'))
                plt.subplot(1, 2, 1)
                #plt.plot([0, 500], [0, 500])
                plt.imshow(img)
                plt.axis("off")
                plt.subplot(1, 2, 2)
                #plt.plot([0, 500], [0, 500])
                plt.imshow(et_density_map_interpolation, cmap=plt.cm.jet)
                plt.axis("off")
                plt.savefig(save_path, bbox_inches = "tight", dpi=200)
                plt.show()
                plt.close()
                '''

                '''
                ##  TODO debug
                plt.subplot(1, 2, 1)
                plt.imshow(img)
                plt.axis("off")
                plt.subplot(1, 2, 2)
                plt.imshow(et_density_map, cmap = plt.cm.jet)
                plt.axis("off")
                plt.text(6, 130, label1, size = 8, alpha = 0.9)
                plt.savefig(save_path, bbox_inches = "tight")
                #plt.show()
                plt.close()
                '''

                fd.write("IMG_{name};"
                         "et_count:{et_count:.2f};\n"
                         .format(name = str(frames[i - 1].num), et_count = et_count))
                break
        FrameNum += 1
    end = time.time()
    seconds = end - start
    fps = FrameNum / seconds
    print("elapsed time:{elapsed_time:.4f};  "
          "FrameNum:{FrameNum:d};      "
          "fps:{fps:.4f};\n  "
          .format(elapsed_time = seconds, FrameNum = FrameNum, fps = fps))
    fd.write("elapsed time:{elapsed_time:.4f};  "
             "FrameNum:{FrameNum:d};      "
             "fps:{fps:.4f};\n  "
             .format(elapsed_time = seconds, FrameNum = FrameNum, fps = fps))
    fd.close()

if __name__ == '__main__':
    main()
