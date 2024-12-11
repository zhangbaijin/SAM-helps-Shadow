import cv2
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel
sys.path.append("..")
from segment_anything.segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import os
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def image_composer(masks, image):
    mask = np.zeros((*masks.shape, 3))
    mask[:,:,0] = masks
    mask[:,:,1] = masks
    mask[:,:,2] = masks
    masked_img = image * mask
    return masked_img

# def need_to_show(masks, image, text, target_tag, processor, clip_model):
#     max_similarity = -1
#     max_similarity_index = -1
#     plot = np.zeros(len(masks), dtype=bool)
#     for i in range(len(masks)):
#         image_index = image_composer(masks[i]['segmentation'], image)

#         inputs = processor(text=text, images=image_index, return_tensors="pt", padding=True)
#         outputs = clip_model(**inputs)
#         logits_per_image = outputs.logits_per_image
#         probs = logits_per_image.softmax(dim=1)
#         max_prob_index = probs.argmax(1)
#         if text[max_prob_index] == target_tag and probs[0, max_prob_index] > max_similarity:
#             max_similarity = probs[0, max_prob_index]
#             max_similarity_index = i

#     plot[max_similarity_index] = True
#     return plot

def need_to_show(masks, image, text, target_tag, processor, clip_model):
    plot = np.zeros(len(masks), dtype=bool)
    for i in range(len(masks)):
        image_index = image_composer(masks[i]['segmentation'], image)

        inputs = processor(text=text, images=image_index, return_tensors="pt", padding=True)
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        if text[probs.argmax(1)] == target_tag:
            plot[i] = True
            plt.figure(figsize=(10,10))
            plt.imshow(image)
            show_mask(masks[i]['segmentation'], plt.gca())
            plt.axis('off')
            plt.show()
    return plot


def show_anns_select(anns, plot):
    if len(anns) == 0:
        return

    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    sorted_plot = [plot[i] for i in np.argsort([-ann['area'] for ann in anns])]
    for i, ann in enumerate(sorted_anns):
        if sorted_plot[i]:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
    ax.imshow(img)


def show_masks_select(masks, plot):
    for i in range(len(masks)):
        if plot[i] == True:
            plt.figure(figsize=(5,5))
            show_mask(masks[i]['segmentation'], plt.gca())
            plt.axis('off')
            plt.show()
# def show_masks_select(masks, plot):
#     for i in range(len(masks)):
#         if plot[i]:
#             plt.figure(figsize=(5, 5))
#             mask = masks[i]['segmentation']
#             mask_bw = (mask * 255).astype(np.uint8)  # 将蒙版转换为黑白单通道图像
#             plt.imshow(mask_bw, cmap='gray')  # 使用灰度色彩映射显示黑白图像
#             plt.axis('off')
#             plt.savefig(f'{masks_select_save_dir}/mask_select_{i}.png')  # 保存每个被选中的蒙版
#             plt.close()




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sam = sam_model_registry["vit_h"]("./pretrained_models/sam_vit_h_4b8939.pth").to(device)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("Load Model successfully！")
mask_generator = SamAutomaticMaskGenerator(sam)
image = cv2.imread('./imgs/ISTD-256/95-4.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
masks = mask_generator.generate(image)
# ... 你之前的代码 ...
# 创建一个用于保存图片的文件夹
save_dir = './results/masks'
os.makedirs(save_dir, exist_ok=True)
for i, mask_dict in enumerate(masks):
    mask = mask_dict['segmentation']

    # convert the boolean mask to an image mask
    mask_img = (mask.astype(np.uint8) * 255)

    # save the image mask
    cv2.imwrite(os.path.join(save_dir, f'mask_{i}.png'), mask_img)


# 创建一个保存annotations图片的文件夹
anns_save_dir = './results/anns'
os.makedirs(anns_save_dir, exist_ok=True)



# 创建一个保存selected masks图片的文件夹
masks_select_save_dir = './results/masks_select'
os.makedirs(masks_select_save_dir, exist_ok=True)

# 展示并保存每个遮罩的annotation
for i, mask in enumerate(masks):
    plt.figure(figsize=(5, 5))
    plt.imshow(image)
    show_anns([mask])
    plt.axis('off')
    plt.savefig(f'{anns_save_dir}/ann_{i}.png')  # 保存每个遮罩的annotation
    plt.close()

# 使用CLIP进行遮罩选择
text = ['shadow', 'background', 'road', 'cars door', 'door', 'walkway', 'human', 'plastic', 'canvas', 'smoke', 'building', 'mountain', 'snow', 'animal', 'sport car', 'beach', 'sand']
target_tag = 'shadow'
plot = need_to_show(masks, image, text, target_tag, processor, clip_model)

# 展示并保存每个被选中的遮罩
max_similarity_index = np.argmax(plot)  # 获取最好的mask的索引
max_similarity_index = np.argmax(plot)  # 获取最好的蒙版的索引
plt.figure(figsize=(2.5, 2.5))
show_masks_select(masks, plot)  # 显示所有与目标标签相似的蒙版
plt.axis('off')
plt.savefig(f'{masks_select_save_dir}/mask_select.png')  # 保存蒙版
plt.close()

# import cv2

# # 加载黑白图像
# mask_select_path = f'{masks_select_save_dir}/mask_select.png'
# mask_select_image = cv2.imread(mask_select_path, cv2.IMREAD_GRAYSCALE)

# # 将黑白图像转换为彩色图像
# mask_select_image_color = cv2.cvtColor(mask_select_image, cv2.COLOR_GRAY2RGB)

# # 保存彩色图像
# cv2.imwrite(mask_select_path, mask_select_image_color)
