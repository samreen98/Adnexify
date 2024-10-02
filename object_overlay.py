import PIL
from PIL import Image, ImageDraw
import numpy as np
import torch
import os
import base64
import requests
from transformers import pipeline

from lama_inpaint import inpaint_img_with_lama
from utils import load_img_to_array, save_array_to_img, dilate_mask,show_mask

para_dict = {
    "coords_type": "bbox",
    "point_labels": [1],
    "dilate_kernel_size": None,
    # "lama_config": "/mnt/nfshome3/FRACTAL/parul.chaudhary/creative_copy/lama/configs/prediction/default.yaml",
    # "lama_ckpt": "/mnt/nfshome3/FRACTAL/parul.chaudhary/Inpaint-Anything/pretrained_models/big-lama"
    #"lama_config": "lama/configs/prediction/default.yaml",
    #"lama_ckpt": "../Inpaint-Anything/pretrained_models/big-lama"
    #"lama_config": "/home/FRACTAL/samreen.khan/samreen/models/lama_config/default.yaml",
    "lama_config": "lama/lama_config/default.yaml",
    "lama_ckpt": "/home/FRACTAL/samreen.khan/samreen/models/pretrained_models/big-lama"
}

device: str = "cuda" if torch.cuda.is_available() else "cpu"

def inpaint(img, bbox_coords):
    # img = load_img_to_array(img_path)
    img = np.array(img)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Initialize a blank mask
    combined_mask = np.zeros_like(img)

    # for bbox_coords in bbox_list:
    x_min, y_min, x_max, y_max = bbox_coords
    mask = np.zeros_like(img)
    mask[int(y_min):int(y_max), int(x_min):int(x_max)] = para_dict["point_labels"]
    combined_mask += mask

    # dilate mask to avoid unmasked edge effect
    if para_dict["dilate_kernel_size"] is not None:
        combined_mask = dilate_mask(combined_mask, para_dict["dilate_kernel_size"])

    num_iterations = 1
    inpainted_img = img.copy()
    combined_mask_vis = np.mean(combined_mask, axis=-1)

    for i in range(num_iterations):
        if i % 10 == 0:
            print(i)
        inpainted_img = inpaint_img_with_lama(
            inpainted_img, combined_mask_vis, para_dict["lama_config"], para_dict["lama_ckpt"], device=device)
 
   # Convert the NumPy array to an image
    inpainted_image = Image.fromarray(inpainted_img)

    return inpainted_image

def detect_object(image, object_name):
    
    checkpoint = "google/owlv2-base-patch16-ensemble"
    detector = pipeline(model=checkpoint, task="zero-shot-object-detection")

    image_copy = image.copy()
    predictions = detector(
        image_copy,
        candidate_labels=[object_name],
    )

    # Initialize variables to track the bbox with the highest score
    max_score = 0
    max_bbox = None

    draw = ImageDraw.Draw(image_copy)

    if predictions:
        for prediction in predictions:
            box = prediction["box"]
            label = prediction["label"]
            score = prediction["score"]

            xmin, ymin, xmax, ymax = box.values()
            draw.rectangle((xmin, ymin, xmax, ymax), outline="red", width=1)
            draw.text((xmin, ymin), f"{label}: {round(score,2)}", fill="black")
            # Check if this prediction has the highest score so far
            if score > max_score:
                max_score = score
                max_bbox = box
    else:
        return None
 
    image_copy.save("outputs/detected_object.png")
    return (max_bbox.values())

def pad_bbox(bbox, image, pad_value=10):
    # Unpack the bounding box
    x1, y1, x2, y2 = bbox
    
    # Get image dimensions (width, height)
    img_width, img_height = image.size
    
    # Apply padding
    padded_x1 = x1 - pad_value
    padded_y1 = y1 - pad_value
    padded_x2 = x2 + pad_value
    padded_y2 = y2 + pad_value
    
    # Ensure the padded bbox is within the image boundaries
    padded_x1 = max(0, padded_x1)  # Ensure x1 is not less than 0
    padded_y1 = max(0, padded_y1)  # Ensure y1 is not less than 0
    padded_x2 = min(img_width, padded_x2)  # Ensure x2 is not more than image width
    padded_y2 = min(img_height, padded_y2)  # Ensure y2 is not more than image height
    
    # Return the padded bounding box
    padded_bbox = (padded_x1, padded_y1, padded_x2, padded_y2)
    
    return padded_bbox

def resize_img(object_img, bbox):
    # Unpack the bounding box (x1, y1, x2, y2)
    x1, y1, x2, y2 = bbox
    
    # Calculate the maximum allowable width and height from the bbox
    bbox_width = x2 - x1
    bbox_height = y2 - y1
    
    # Get the original dimensions of the object image
    obj_width, obj_height = object_img.size
    
    # Calculate the aspect ratios for the object image and the bounding box
    width_ratio = bbox_width / obj_width
    height_ratio = bbox_height / obj_height
    
    # Choose the smaller ratio to maintain the aspect ratio while fitting within the bbox
    scale_ratio = min(width_ratio, height_ratio)
    
    # Compute the new size of the object image while maintaining aspect ratio
    new_width = int(obj_width * scale_ratio)
    new_height = int(obj_height * scale_ratio)
    
    # Resize the image using the new dimensions
    resized_img = object_img.resize((new_width, new_height), PIL.Image.LANCZOS)
    
    return resized_img

def get_object(alpha_img):
    # Ensure the image has an alpha channel
    if alpha_img.mode != 'RGBA':
        alpha_img = alpha_img.convert("RGBA")

    # Convert the image to a NumPy array for easier manipulation
    alpha_np = np.array(alpha_img)

    # Extract the alpha channel (4th channel in RGBA)
    alpha_channel = alpha_np[:, :, 3]

    # Find non-transparent pixels (alpha > 0)
    non_transparent_pixels = np.where(alpha_channel > 0)

    if len(non_transparent_pixels[0]) == 0 or len(non_transparent_pixels[1]) == 0:
        # No non-transparent pixels, return the original image (or handle as needed)
        return alpha_img

    # Get the bounding box around non-transparent pixels
    top_left_x = np.min(non_transparent_pixels[1])
    top_left_y = np.min(non_transparent_pixels[0])
    bottom_right_x = np.max(non_transparent_pixels[1])
    bottom_right_y = np.max(non_transparent_pixels[0])

    # Crop the original image using the bounding box
    cropped_img = alpha_img.crop((top_left_x, top_left_y, bottom_right_x + 1, bottom_right_y + 1))

    return cropped_img

def overlay_alpha_img(inpainted_img, resized_object_img, padded_bbox):
    # Calculate the center of the padded bounding box
    padded_bbox_x1, padded_bbox_y1, padded_bbox_x2, padded_bbox_y2 = padded_bbox
    padded_bbox_center_x = (padded_bbox_x1 + padded_bbox_x2) // 2
    padded_bbox_center_y = (padded_bbox_y1 + padded_bbox_y2) // 2

    # Get the dimensions of the resized object image
    resized_width, resized_height = resized_object_img.size

    # Calculate the center of the resized object image
    object_center_x = resized_width // 2
    object_center_y = resized_height // 2

    # Calculate the top-left position to paste the object such that the centers align
    top_left_x = padded_bbox_center_x - object_center_x
    top_left_y = padded_bbox_center_y - object_center_y

    # Ensure the coordinates are within the image boundaries
    top_left_x = max(0, top_left_x)
    top_left_y = max(0, top_left_y)

    # Create a copy of the inpainted image to work on
    final_img = inpainted_img.copy()

    # Paste the object image onto the inpainted image at the calculated position
    # Use the alpha channel from resized_object_img to handle transparency
    final_img.paste(resized_object_img, (top_left_x, top_left_y), resized_object_img)

    return final_img

def overlay_object_old(img, object_alpha, padding, object_name):

    bbox = detect_object(img, object_name)
    print(bbox)
    if (bbox==None):
        return img
    padded_bbox = pad_bbox(bbox, img, padding)
    inpainted_img = inpaint(img, padded_bbox)
    inpainted_img.save("outputs/inpainted_img.png")
    object_img = get_object(object_alpha)
    
    resized_object_img = resize_img(object_img, padded_bbox)

    final_image =  overlay_alpha_img(inpainted_img, resized_object_img, padded_bbox)

    final_image.save("outputs/overlayed.png")

    return final_image


def overlay_object(img, object_alpha, padding, object_name):
 
    bbox = detect_object(img, object_name)
    print(bbox)
    if (bbox==None):
        return img
    padded_bbox = pad_bbox(bbox, img, padding)
    shrinked_bbox = shrink_bbox(bbox, img, padding)
    inpainted_img = inpaint(img, padded_bbox)
    inpainted_img.save("outputs/inpainted_img.png")
    object_img = get_object(object_alpha)
   
    resized_object_img = resize_img(object_img, shrinked_bbox)
 
    final_image =  overlay_alpha_img(inpainted_img, resized_object_img, shrinked_bbox)
 
    final_image.save("outputs/overlayed.png")
 
    return final_image



# img = Image.open("/home/FRACTAL/parul.chaudhary/ad_hackathon/outputs/layout_1_variation_0.png")
# object_alpha =Image.open("/home/FRACTAL/parul.chaudhary/ad_hackathon/outputs/object_alpha_img.png")

# # bbox = (551, 262, 776, 670)
# overlay_object(img, object_alpha, 10, "bottle")



def shrink_bbox(bbox, image, pad_value=10):
    # Unpack the bounding box
    x1, y1, x2, y2 = bbox
   
    # Get image dimensions (width, height)
    img_width, img_height = image.size
   
    # Apply padding
    padded_x1 = x1 + pad_value
    padded_y1 = y1 + pad_value
    padded_x2 = x2 - pad_value
    padded_y2 = y2 - pad_value
   
    # # Ensure the padded bbox is within the image boundaries
    # padded_x1 = max(0, padded_x1)  # Ensure x1 is not less than 0
    # padded_y1 = max(0, padded_y1)  # Ensure y1 is not less than 0
    # padded_x2 = min(img_width, padded_x2)  # Ensure x2 is not more than image width
    # padded_y2 = min(img_height, padded_y2)  # Ensure y2 is not more than image height
   
    # Return the padded bounding box
    shrinked_bbox = (padded_x1, padded_y1, padded_x2, padded_y2)
   
    return shrinked_bbox