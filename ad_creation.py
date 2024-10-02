import os
from openai import OpenAI, AzureOpenAI
import openai
import ast

import torch
from diffusers import FluxPipeline
from PIL import Image
import matplotlib.pyplot as plt
import base64
from openai import OpenAI
import cv2

import re
import numpy as np
import textwrap

os.environ['OPENAI_API_KEY'] = '<your openai key>'
openai.api_key = os.environ['OPENAI_API_KEY']
openai_api_key = os.environ['OPENAI_API_KEY']
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")

def generate_and_display_image(pipe, prompt, seed=10, save=False, show=True, filename="flux-schnell.png") -> Image:

    image = pipe(
        prompt,
        height=800,
        width=1360,
        guidance_scale=0.0,   #0.0
        num_inference_steps=4,
        max_sequence_length=256,
        #generator=torch.Generator("cpu").manual_seed(seed)
    ).images[0]

    if save:
        image.save(filename)
        print(f"Image saved as: {filename}")
    
    if show:
        plt.figure(figsize=(6,6))
        plt.imshow(image)
        plt.axis('off')  # Turn off axis labels and ticks
        plt.show()

    return(image)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def encode_numpy_image(np_image):
    rgb_image = cv2.cvtColor(np_image, cv2.COLOR_BGR2RGB)
    # Convert the NumPy image array to a byte buffer (e.g., PNG format)
    _, buffer = cv2.imencode('.png', rgb_image)
    # Encode the buffer to base64
    return base64.b64encode(buffer).decode('utf-8')

def image_query(question, system_msg="Helpful assistant", image_path=None, numpy_img=None):

    base64_image=None
    if image_path:
        base64_image = encode_image(image_path)

    if numpy_img is not None:
        base64_image = encode_numpy_image(numpy_img)

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            { "role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                                },
                    },
                ],
            }
        ],
        max_tokens=300,
    )

    raw_output = response.choices[0].message.content
    return( raw_output )

def process_upload_product_response_old(response):

    # Default values in case the LLM does not find the product
    default_product_name = "suqare bottle"
    default_alignment = "vertical"
    default_description = "No description available."

    # Check if the response contains the expected product information
    if "Product Name:" in response and "Alignment:" in response:
        try:
            # Extracting the Product Name
            product_name = response.split("Product Name: ")[1].split("\n")[0]
            # Extracting the Alignment (if present)
            alignment = response.split("Alignment: ")[1].split("\n")[0] if "Alignment: " in response else "None"
            # Extracting the Description
            description = response.split("Description: ")[1].strip()
        except IndexError:
            # If something is missing or there's an error, use default values
            product_name = default_product_name
            alignment = default_alignment
            description = default_description
    else:
        # If no product information is found, use default values
        product_name = default_product_name
        alignment = default_alignment
        description = default_description

    # Return the processed information
    return (product_name, alignment )

def process_upload_product_response(response):
    # Default values in case the LLM does not find the product
    default_product_name = "unknow"
    default_product_class = "Unknown"
    default_alignment = "vertical"
    default_description = "No description available."

    # Check if the response contains the expected product information
    if "Product name:" in response and "Product class:" in response and "Alignment:" in response:
        try:
            # Extracting the Product Name
            product_name = response.split("Product name: ")[1].split("\n")[0]
            # Extracting the Product Class
            product_class = response.split("Product class: ")[1].split("\n")[0]
            # Extracting the Alignment
            alignment = response.split("Alignment: ")[1].split("\n")[0]
            # Extracting the Description
            description = response.split("Description: ")[1].strip()
        except IndexError:
            # If something is missing or there's an error, use default values
            product_name = default_product_name
            product_class = default_product_class
            alignment = default_alignment
            description = default_description
    else:
        # If no product information is found, use default values
        product_name = default_product_name
        product_class = default_product_class
        alignment = default_alignment
        description = default_description

    # Return the processed information
    return (product_name, product_class, alignment)



def upload_product(image_path):
    #image_path = "/home/FRACTAL/samreen.khan/samreen/code/Big_basket/inputs/products_alpha/119709.png"

    system_msg = """
    Extract and return the full product name even if it is vegetable or fruits or grocery item exactly as it appears, including relevant descriptors
    (e.g., variant, shape, type, and packaging). **Include the packaging type (e.g., bottle, can, packet) as 
    part of the product name if it is present**.
    Additionally, identify the alignment of the product (e.g., vertical, lying, tilted) and include it in the description.

    Follow the format strictly:
    Product name: [full product name, including packaging type]
    Product class: [Give class name this Product belongs to. Normally classes on which image object detectors are trained on ]
    Alignment: [alignment of the product]
    Description: [brief description of the product, emphasizing packaging type].
    Do not shorten or omit any part of the product name.
    """

    question = "Identify the product"

    image_ans = image_query(question, system_msg=system_msg, image_path=image_path)
    print( image_ans )
    product_name, product_class, product_alignment  = process_upload_product_response(image_ans)

    #product_name = image_ans.split("Product Name:")[1].split("\n")[0].strip()
    #product_alignment = image_ans.split("Alignment:")[1].split("\n")[0].strip()

    return(product_name, product_class, product_alignment)


def ask_gpt4(system_msg, user_msg, temperature=0.5):
    llm = OpenAI()
    
    raw_list = llm.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        temperature=temperature
    )
    
    raw_output = raw_list.choices[0].message.content
    return raw_output

def get_theme_based_color_palettes(theme, product_name, product_alignment, temperature=0.5 ):
    system_msg = """
    You are an expert in designing advertising banners. When given a theme for an event or product to be advertised, provide a
    suitable standard color palette name that effectively conveys the essence of the poster. Please respond with only the color palette name.
    Additionally, incorporate some variation in the user message context to enhance creativity and diversity in your responses. 
    """

    user_msg = f"""
    Advertisement for product {product_name} and product alignment is {product_alignment}. Theme is {theme}.
    """
    color_palettes = ask_gpt4(system_msg, user_msg, temperature=temperature)
    #color_palettes = "red,pink,brown"  # temp sk1
    return(color_palettes)


def get_layout_theme( user_theme, temperature=0.5 ):
 
    system_msg = """
    You are an expert in designing themes for ad banners. Based on the user theme, suggest a simple, clean, and concise 
    theme name (1-3 words) for the background of the banner. The theme should be easy to understand, avoid complexity, 
    and should enhance the clarity of the ad. Try to strictly use a key word from user prompt into
    theme name as it is. For e.g if user theme is "summer sale" then theme name could be "soft summer".
    Use standard themes. If user theme has some festivals or seasons then give theme else give "simple" as theme name. 
    """

    user_msg = f"""
    user theme is {user_theme}.
    """
    layout_theme = ask_gpt4(system_msg, user_msg, temperature=temperature)
    return(layout_theme)


def get_ad_layouts( product_name, product_alignment, color_palettes, layout_theme, heading = "Heading", promo = "20% OFF",  variation=3, temperature=0.5 ):

    system_msg1 = """
    You are an expert in designing advertising horizontal banner layouts. Dont generate any text unless specified. Your task is to create designs
    using only three components as mentioned in below format. Each component must be filled with a solid color chosen from the 
    specified color palettes provided by the user. Do not specify exact sizes or positions numerically. Your design should focus on
    relative size and positioning of components withinthe layout. Use a complementary solid color that enhances readability and
    aligns with the overall theme. Follow following format. 

    main:
        Color: 
        texture:
    product:    
        product name:
        alignment:
    component1:
        Color: 
        text:"text"
    promo:
        shape:
        color:
        text:"offer"
    layout pattern:

    Each component area should be clearly distinct from other components. So choose color accordingly.
    Product should be strictly in Product component and not overlapping with any other component
    Product image should occupy larger portion of layout. Put each variation in above formate seperated by "next", no capital 
    letter, no bold *, no inverted comas around next. Make sure each variation doesnot exceed 70 token limit. So keep format 
    precise but important information should not be missed. Elobrate layout pattern in detail based on user input and color 
    pallete and text to put clearly but keep it precise to fit it to token limit of 70. Make layout pattern very creative 
    and eye catching to increase click through rate. Dont change text value in above format.
    """

    system_msg2 = """
    You are an expert in designing advertising horizontal banner layouts. Dont generate any text unless specified.
    Choose any one color from color palettes provided by the user. Do not specify exact sizes or positions numerically. 
    Your design should focus on relative size and positioning of components withinthe layout. 
    Product image should occupy larger portion of layout. Put each variation in above formate 
    seperated by "next", no capital letter, no bold *, no inverted comas around next. Make sure each variation doesnot 
    exceed 70 token limit. Elobrate layout pattern in detail based on user input theme and color 
    pallete but keep it precise to fit it to token limit of 70. Make layout pattern very creative 
    and eye catching to increase click through rate. Dont put any text other than from "text". 
    Use actual product name from user_msg in layout pattern. Just product is not allowed. Take input from user.
    if product belongs to fruit or vegetable class then you may place them in basket in layout pattern.

    Follow following format strictly.
    main:
        Color: 
    product:    
        product name:
    component:
        text:
    promo:
        text:
    layout pattern:
    """

    user_msg = f"""
    Create an eye-catching poster layout featuring four unique components, each utilizing solid colors from the selected 
    color palette: {color_palettes}. theme is {layout_theme}.
    This layout is intended for advertising product name {product_name}. alignment of product is {product_alignment}.
    Please provide {variation} variations of the design by creatively adjusting the size and positioning of the components, 
    without using specific numerical values. component text is {heading}. promo text is {promo}
    """

    raw_output = ask_gpt4(system_msg2, user_msg, temperature=temperature)
    return(raw_output)



def get_imglayouts_and_descriptions(pipe, raw_ad_layouts, product_name, layout_theme="", sub_variation_count=2, save=True):

    #layouts = [layout.strip() for layout in raw_output.split('next') if layout.strip()]
    #layouts = re.findall(r'(main:.*?)(?=\n\s*next|$)', raw_ad_layouts, re.DOTALL)

    layouts = re.findall(r'(layout pattern:.*?)(?=\n\s*next|$)', raw_ad_layouts, re.DOTALL)

    layouts_info = {}

    if save:
        output_directory = f"outputs"
        os.makedirs(output_directory, exist_ok=True)

    system_prompt = """You are a highly skilled assistant designed to analyze images and offer comprehensive descriptions and 
    evaluations. Your task is to provide detailed information about key elements, colors, objects, and especially text that appears
    outside the products. Focus on text and visual elements not directly written on or embedded within the product itself.
    Give information about text only once. Dont repeat in another way. 
    """
    question = "Provide a detailed description of the image, highlighting key elements, colors, objects, and any text outside of the product"
    filepath = ""
    for i, layout in enumerate(layouts, start=1):
        #print(f">>>>>> Layout {i}:")
        
        layout = f"advevrtisement banner. {layout} Theme is {layout_theme} "
        #layout = f"{layout} Theme is {layout_theme}"

        print(layout)
        for v in range(sub_variation_count):
            if save:
                filepath = f"{output_directory}/{product_name.split(' ')[0]}_layout_{i}_variation_{v}.png"    #sk1

            img = generate_and_display_image(pipe, prompt=layout, save=save, filename=filepath)
            image_ans  = image_query(question, system_msg=system_prompt, numpy_img=np.array(img))
            file_name = f"{product_name.split(' ')[0]}_layout_{i}_variation_{v}.png"    #sk1
            print( ">>>>> file_name >>>>> ", file_name )
            layouts_info[file_name] = (img, layout, image_ans)

        print("\n" + "="*40 + "\n")

    return(layouts_info )


def extract_parameters(ad_feedback):

    description_match = re.search(r'Image Description Matching:\s*(Yes|No)', ad_feedback).group(1)
    product_visibility = re.search(r'Product Visibility:\s*(Yes|No)', ad_feedback).group(1)


    text_identified = re.findall(r'- Text:\s*"([^"]+):([^"]+)"', ad_feedback)
    text_identification = [{"text": text[0], "color": text[1]} for text in text_identified]

    aesthetic_score = int(re.search(r'Aesthetic Appeal:\s*Score:\s*(\d+)', ad_feedback).group(1))
    layout_score = int(re.search(r'Layout Effectiveness:\s*Score:\s*(\d+)', ad_feedback).group(1))
    overall_impression_score = int(re.search(r'Overall Advertising Impact:\s*Score:\s*(\d+)', ad_feedback).group(1))

    # Output the extracted data
    feedback_result = {
        "description_match": description_match,
        "product_visibility": product_visibility,
        "text_identification": text_identification,
        "aesthetic_score": aesthetic_score,
        "layout_score": layout_score,
        "overall_impression_score": overall_impression_score
    }

    return( feedback_result )

# Define thresholds for each criterion and total score
AESTHETIC_THRESHOLD = 7
LAYOUT_THRESHOLD = 7
OVERALL_IMPRESSION_THRESHOLD = 7
MIN_WORD_COUNT = 2

def validate_ad_layout(image_name, feedback_result):

    total_score = 0
    is_pass = True
    if feedback_result["description_match"].lower() == 'no':
        print( f"Rejected: Image '{image_name}' - Description does not match the prompt." )
        is_pass = False
    
    if feedback_result["product_visibility"].lower() == 'no':
        print( f"Rejected: Image '{image_name}' - Product is not clearly visible." )
        is_pass = False
    
    # Check if individual scores are too low
    if feedback_result["aesthetic_score"] < AESTHETIC_THRESHOLD:
        print( f"Rejected: Image '{image_name}' - Aesthetic score too low (Score: {feedback_result['aesthetic_score']} )." )
        is_pass = False

    if len(feedback_result["text_identification"]) < MIN_WORD_COUNT:
        print( f"Rejected: Image '{image_name}' - words count too low (count: {len(feedback_result['text_identification'])})." )
        is_pass = False

    if feedback_result["layout_score"] < LAYOUT_THRESHOLD:
        print( f"Rejected: Image '{image_name}' - Layout effectiveness score too low (Score: {feedback_result['layout_score']})." )
        is_pass = False

    if feedback_result["overall_impression_score"] < OVERALL_IMPRESSION_THRESHOLD:
        print( f"Rejected: Image '{image_name}' - Overall impression score too low (Score: {feedback_result['overall_impression_score']})." )
        is_pass = False

    # If all checks pass, accept the layout
    print( f"Image '{image_name}' is_pass:{is_pass} " )
    if is_pass:
        total_score = feedback_result['aesthetic_score'] + feedback_result["layout_score"] + feedback_result["overall_impression_score"]
        
    return(is_pass, total_score)


def save_image(image_data, file_path, format='PNG'):
    try:
        if isinstance(image_data, Image.Image):
            image_data.save(file_path, format=format)
        else:
            # If it's not a PIL Image, we can convert numpy array to Image
            image = Image.fromarray(image_data)
            image.save(file_path, format=format)
        print(f"Image saved successfully at {file_path}")
        return True
    except Exception as e:
        print(f"Error saving image: {e}")
        return False


def filter_ad_images( layouts_info, product_name, heading=None, promo=None, save=False ):

    if save:
        output_directory = f"outputs/{product_name.split(' ')[0]}"   #sk1
        os.makedirs(output_directory, exist_ok=True)

    system_msg = """
    You are an expert in image validation for advertisements, known for strict assessments but dont cut score for incorrect 
    text or misspelling. Based on the provided image description, which includes details about the product, 
    text (excluding any written directly on the product), and layout, provide structured responses to the following 
    questions. Your evaluations must be detailed.

    1. Image Description Matching: Does the image description match the prompt provided, dont be strict here, even little match is good? (yes/no)

    2. Product Visibility: Is the product fully visible and clear in the image description? Partial visibility is not 
    acceptable. Overlapped with other object is not allowed. (yes/no)

    3. Text Details (Exclude text on the product and strictly follow format as this example: - Text: "Delicious:black"):
    - Format as: "Text: [visible text:color and nothing else]"
    - Only include text not written directly on the product itself.

    4. Aesthetic Appeal: Rate the aesthetic quality of this image description for an advertisement on a scale from 0 to 10, with 10 being highly appealing and 0 being unappealing. (Score: 0 to 10)

    5. Layout Effectiveness: Evaluate how effectively the layout conveys the intended message. Give less score if offer is found at more than 1 places. Provide a score from 0 to 10, with 10 being highly effective. Include a brief explanation. (Score: 0 to 10)

    6. Overall Advertising Impact: Provide an overall impression of the image description's effectiveness for advertising. Score from 0 to 10, with a short justification for your rating. (Score: 0 to 10)

    Please respond in the format shown above with clearly labeled answers. Dont give less score for misspelling or incorrect text. In above wherever socre present, follow format as score:
    """

    passed_images = {}
    for idx, (filename, (image, layout, answer)) in enumerate(layouts_info.items()):
        print(f"Image name >>>, {idx} : {filename}")
        print("layout >>>", textwrap.fill(layout, width=120))
        print("img desciption >>>",textwrap.fill(answer, width=120))

        user_msg = f"""
            prompt provided is {layout}. Generated image description is {answer}. Product is {product_name}."
            If the generated image description does not contain the {heading} and {promo} then assign an 
            Overall Advertising Impact score of 2. 
            if {heading} and {promo} is present more than once then give Overall Advertising Impact score as 4.
        """
        validation_output = ask_gpt4(system_msg, user_msg, temperature=0.2)
        print("validation_output >>>> ",validation_output)

        feedback_result = extract_parameters(validation_output)
        print("feedback_result >>>> ",feedback_result)
        is_pass, total_score  = validate_ad_layout(filename, feedback_result)

        if is_pass:
            passed_images[filename] = (image, total_score)
            if save:
                filename1 = f"{output_directory}/passed_{product_name.split(' ')[0]}_{idx}_score_{total_score}.png"
                save_image(image, filename1, format='PNG')

        print("\n" + "="*40 + "\n")

    return(passed_images)        


def get_ad_heading(theme, product_name, temperature=0.5 ):

    system_msg = """
    You are an expert in crafting attention-grabbing headlines for advertising banners. headline should align with the product 
    and the user's theme, consisting of no more than two words. You may give 3 words heading only if words are small. Keep it simple yet impactful, avoiding complex language. The headline 
    should function as a tagline that instantly captures attention. Be creative and engaging. Give more importance to product in heading.
    """

    user_msg = f"""
    Advertisement for product {product_name}. theme is {theme}.
    """
    heading = ask_gpt4(system_msg, user_msg, temperature=temperature)
    return(heading)


def create_ad( pipe, image_path, seasonal_theme, heading, promo, color_palettes, variation=1, sub_variation_count=1, save_img=False ):

    product_name, product_class, product_alignment = upload_product(image_path)
    print("product_name >> ",product_name)
    print("product_class >>",product_class)
    print("product_alignment >>",product_alignment)

    # User gives theme
    #color_palettes = get_theme_based_color_palettes(seasonal_theme, product_name, product_alignment, temperature=0.5 )
    print("color_palettes >> ",color_palettes)
    layout_theme = get_layout_theme( seasonal_theme, temperature=0.2 )
    print( "layout_theme >> ", layout_theme )

    raw_ad_layouts = get_ad_layouts( product_name, product_alignment, color_palettes, layout_theme, 
                                    heading = heading, promo = promo, variation=variation, temperature=0.5 )
    print( "raw_ad_layouts >> ",raw_ad_layouts )

    layouts_info = get_imglayouts_and_descriptions(pipe, raw_ad_layouts, product_name, layout_theme=layout_theme, 
                                                sub_variation_count=sub_variation_count, save=save_img)
    
    
    passed_images = filter_ad_images( layouts_info, product_name, heading=heading, promo=promo, save=save_img )
    return( passed_images, product_class )


def ad_creation_pipeline( pipe, image_path, seasonal_theme, heading, promo, color_palettes, save_img=False, max_attempts=3 ):

    final_selected_img = None
    best_score = 0
    attempt = 0  
    
    while final_selected_img is None and attempt < max_attempts:
        attempt += 1
        passed_images, product_class = create_ad( pipe, image_path, seasonal_theme, heading, promo,color_palettes, variation=1, sub_variation_count=1, save_img=save_img )

        best_score = 0
        for idx, (filename, (image, total_score)) in enumerate(passed_images.items()):
            print(f"!!!!!!!!!!!!!!!! file:{filename} and score:{total_score}")

            if best_score < total_score:
                best_score = total_score
                final_selected_img = image

            plt.figure(figsize=(6,6))
            plt.imshow(image)
            plt.axis('off') 
            plt.show()

        if final_selected_img is None:
            print(f"No suitable image found on attempt {attempt}.")
    
    if final_selected_img is None:
        print("Maximum attempts reached. Try again.")
    
    return(final_selected_img, product_class)

def layout_gen_pipeline(object_alpha_img, user_theme, user_heading, promo_text, color_palettes):

    output_directory = f"outputs"
    os.makedirs(output_directory, exist_ok=True)

    print(object_alpha_img.mode)
    # Ensure the image is in RGBA mode
    if object_alpha_img.mode != 'RGBA':
        print("Mode not correct")
        object_alpha_img = object_alpha_img.convert('RGBA')

    # Save the image in RGBA format
    image_path = f"{output_directory}/object_alpha_img.png"
    object_alpha_img.save(image_path, format='PNG')  # PNG format retains RGBA information

    max_attempts = 4
    final_selected_img, object_cls = ad_creation_pipeline( pipe, image_path, user_theme, user_heading, promo_text, color_palettes, save_img=True, max_attempts=max_attempts )
    print(object_cls)
    return final_selected_img, object_cls