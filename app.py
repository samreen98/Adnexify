import os
import random
import gradio as gr
import torch
import diffusers
import numpy as np
from diffusers import FluxInpaintPipeline
from object_overlay import overlay_object
# from layout_gen import layout_gen_pipeline
from ad_creation import layout_gen_pipeline, upload_product, get_ad_heading
from ad_creation import get_theme_based_color_palettes

def get_heading_and_color_palette(img, user_theme):
    output_directory = f"outputs"
    os.makedirs(output_directory, exist_ok=True)

    print(img.mode)
    # Ensure the image is in RGBA mode
    if img.mode != 'RGBA':
        print("Mode not correct")
        img = img.convert('RGBA')

    # Save the image in RGBA format
    image_path = f"{output_directory}/object_alpha_img.png"
    img.save(image_path, format='PNG')  # PNG format retains RGBA information

    product_name, product_class, product_alignment = upload_product(image_path)
    poster_heading = get_ad_heading(user_theme, product_name, temperature=0.5 )

    color_palettes = get_theme_based_color_palettes(user_theme, product_name, product_alignment, temperature=0.5 )
    return poster_heading, color_palettes


def ad_pipeline(object_alpha_img, user_theme='', user_heading='', promo='', color_palettes=''):
    output_directory = f"outputs"
    os.makedirs(output_directory, exist_ok=True)

    # generated_layout, class_name = layout_gen_pipeline(object_alpha_img, user_theme, user_heading, promo)
    generated_layout, class_name= layout_gen_pipeline(object_alpha_img, user_theme, user_heading, promo, color_palettes)
    print("class_name.lower()", class_name.lower())
    if class_name.lower() not in ['fruit', 'vegetable']:
        print("111111111111111111111111111111111111111")
        final_ad = overlay_object(generated_layout, object_alpha_img, 10, class_name)
    else:
        print(222222222222222222222222222222222222222222)
        # Skip overlay_object and use generated_layout as final output
        final_ad = generated_layout
    # final_ad = overlay_object(generated_layout, object_alpha_img, 10 , class_name_temp)

    return final_ad


with gr.Blocks() as generate_ad:
    with gr.Row():
        with gr.Column(scale=1):
            prod_img = gr.Image(type="pil", label="Product Image", image_mode='RGBA', sources="upload", elem_id="prod_image")
        with gr.Column(scale=1):
            canvas_prod = gr.Image(label="Generated Poster", elem_id="output_poster")

    with gr.Row():
        with gr.Column(scale=1):
            theme = gr.Textbox(label="Provide a theme for Banner", placeholder="E.g., Summer Sale, Diwali offer", value="", interactive=True)
            generate_heading_btn = gr.Button(value="Generate Heading and Color Palette", elem_id="generate_heading_button")
           
            heading_text = gr.Textbox(label="Heading for Banner", placeholder="E.g., Special Offer", value="", interactive=True)
            color_palette = gr.Textbox(label="Color Palette for Banner", placeholder="E.g., red,brown,pink or Rosy flamingo", value="", interactive=True)
            promo_text = gr.Textbox(label="Provide Promotional offer for Banner", placeholder="E.g., 50% OFF", value="")
            #image_res = gr.Dropdown(label="Select Image Resolution", choices=["1360x800", "1080x1080", "800x1360"], value="1360x800", interactive=True)
            image_res = gr.Dropdown(label="Select Image Resolution", choices=["1360x800"], value="1360x800", interactive=True)
            run_btn = gr.Button(value="Generate Banner", elem_id="generate_button")
        with gr.Column(scale=1):
            print()

    # Trigger return_heading function only when "Generate Heading" button is clicked
    generate_heading_btn.click(get_heading_and_color_palette, inputs=[prod_img, theme], outputs=[heading_text, color_palette])
    run_btn.click(ad_pipeline, inputs=[prod_img, theme, heading_text, promo_text, color_palette], outputs=[canvas_prod])

        
    # Custom CSS for styling
    generate_ad.css = """
        #prod_image, #output_poster {
            width: 100%;  /* Ensure both images are the same width */
            height: 100%; /* Increase image height */
            border: 1px solid #000000; /* Add border if needed */ 
        }

        #generate_button, #generate_heading_button {
            color: black;
            border: none;
            padding: 15px 32px;
            text-align: center;
            font-size: 18px !important;;
            font-family: 'Arial', sans-serif;  /* Better font choice */
            font-weight: bold;
            margin: 10px 2px;
            cursor: pointer;
            border-radius: 8px;
        }

        #generate_button:hover, #generate_heading_button:hover {
            background-color: #333333;  /* Lighter black on hover */
            color: red;
        }

        .gr-button {
            border-radius: 8px;
        }
    """

if __name__ == "__main__":
    generate_ad.launch(share=False, server_port=6040)

