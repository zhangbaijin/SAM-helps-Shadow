import torch
import sys
import argparse
from pathlib import Path
from matplotlib import pyplot as plt
from sam_segment import predict_masks_with_sam
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point
import cv2
import numpy as np
from torch.autograd import Variable
import torchvision.transforms as transforms
from IterModel3 import Deshadow_netS4
from PIL import Image
"""python remove_anything_ZXF.py --input_img ./imgs/256/95-4.png --coords_type key_in --point_coords 100 100 --point_labels 1 --dilate_kernel_size 15 --output_dir ./results --sam_model_type "vit_h" --sam_ckpt ./pretrained_models/sam_vit_h_4b8939.pth 
"""
def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--coords_type", type=str, required=True,
        default="key_in", choices=["click", "key_in"], 
        help="The way to select coords",
    )
    parser.add_argument(
        "--point_coords", type=float, nargs='+', required=True,
        help="The coordinate of the point prompt, [coord_W coord_H].",
    )
    parser.add_argument(
        "--point_labels", type=int, nargs='+', required=True,
        help="The labels of the point prompt, 1 or 0.",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=None,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--sam_model_type", type=str,
        default="vit_h", choices=['vit_h', 'vit_l', 'vit_b'],
        help="The type of sam model to load. Default: 'vit_h"
    )
    parser.add_argument(
        "--sam_ckpt", type=str, required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
    )


def test_single_image(input_img_path, mask_img_path, model_weights_path, save_path):
    trans_eval = transforms.Compose(
        [transforms.Resize([256, 256]),
         transforms.ToTensor()
         ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read input image and mask
    input_img = cv2.imread(input_img_path)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)

    # convert images to PIL images
    input_img = Image.fromarray(input_img)
    mask_img = Image.fromarray(mask_img)
    # Apply transforms
    input_img=trans_eval(input_img)
    mask_img=trans_eval(mask_img)

    # Load the model
    net = Deshadow_netS4(ex1=6, ex2=4).to(device)
    net.load_state_dict(torch.load(model_weights_path))

    # Convert images to tensors and send them to the device
    inputs = Variable(input_img.unsqueeze(0)).to(device)
    masks = Variable(mask_img.unsqueeze(0)).to(device)

    # Set model to evaluation mode
    net.eval()

    # Inference
    with torch.no_grad():
        _, _, _, out_eval = net(inputs, masks)
        out_eval = torch.clamp(out_eval, 0., 1.)
        out_eval_np = np.squeeze(out_eval.cpu().numpy())
        out_eval_np_ = out_eval_np.transpose((1, 2, 0))

        # Save the output image
        #cv2.imwrite(save_path, np.uint8(out_eval_np_ * 255.))
        result_img=Image.fromarray(np.uint8(out_eval_np_*255))
        result_img.save(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.coords_type == "click":
        latest_coords = get_clicked_point(args.input_img)
    elif args.coords_type == "key_in":
        latest_coords = args.point_coords
    img = load_img_to_array(args.input_img)
    #print(img.shape)  (256,256,3)

    masks, _, _ = predict_masks_with_sam(
        img,
        [latest_coords],
        args.point_labels,
        model_type=args.sam_model_type,
        ckpt_p=args.sam_ckpt,
        device=device,
    )
    #print(masks.shape) (3,256,256)
    masks = masks.astype(np.uint8) * 255
    #print(masks.shape) (3,256,256)

    # dilate mask to avoid unmasked edge effect
    # if args.dilate_kernel_size is not None:
    #     masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]

    # visualize the segmentation results
    img_stem = Path(args.input_img).stem
    out_dir = Path(args.output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, mask in enumerate(masks):
        # path to the results
        mask_p = out_dir / f"mask_{idx}.png"
        img_points_p = out_dir / f"with_points.png"
        img_mask_p = out_dir / f"with_{Path(mask_p).name}"

        # save the mask
        save_array_to_img(mask, mask_p)
        ##mask_p is path

        # save the pointed and masked image
        dpi = plt.rcParams['figure.dpi']
        height, width = img.shape[:2]
        plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
        plt.imshow(img)
        plt.axis('off')
        show_points(plt.gca(), [latest_coords], args.point_labels,
                    size=(width*0.04)**2)
        plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
        show_mask(plt.gca(), mask, random_color=False)
        plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
        plt.close()
    #print(mask.shape)  256x256
    # inpaint the masked image

    mask_p = out_dir / f"mask_{idx}.png"
    # img_inpainted_p = out_dir / f"inpainted_with_{Path(mask_p).name}"
    # img_inpainted = inpaint_img_with_lama(
    #     img, mask, args.lama_ckpt, device=device)
    # save_array_to_img(img_inpainted, img_inpainted_p)

    input_img_path="./imgs/ISTD-256/134-6.png"
    mask_img_path="./results/134-6/mask_1.png"
    models_weight_path="./AAAI_zyr/ckpt/models/best.pth"
    save_path="./results/134-6/134-6.png"
    test_single_image(input_img_path,mask_img_path,models_weight_path,save_path)


