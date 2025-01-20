from typing import Optional

import gradio as gr
import numpy as np
import torch
from PIL import Image
import io


import base64, os
from utils import check_ocr_box, get_yolo_model, get_caption_model_processor, get_som_labeled_img
import torch
from PIL import Image
import argparse
from fastapi import FastAPI, File, UploadFile
from gradio.routes import mount_gradio_app
import json



MARKDOWN = """
# OmniParser for Pure Vision Based General GUI Agent 🔥
<div>
    <a href="https://arxiv.org/pdf/2408.00203">
        <img src="https://img.shields.io/badge/arXiv-2408.00203-b31b1b.svg" alt="Arxiv" style="display:inline-block;">
    </a>
</div>

OmniParser is a screen parsing tool to convert general GUI screen to structured elements. 
"""

DEVICE = torch.device('cuda')

# @spaces.GPU
# @torch.inference_mode()
# @torch.autocast(device_type="cuda", dtype=torch.bfloat16)

parser = argparse.ArgumentParser(description='Process model paths and names.')
parser.add_argument('--icon_detect_model', type=str, required=True, default='weights/icon_detect/best.pt', help='Path to the YOLO model weights')
parser.add_argument('--icon_caption_model', type=str, required=True, default='florence2',  help='Name of the caption model')

args = parser.parse_args()
icon_detect_model, icon_caption_model = args.icon_detect_model, args.icon_caption_model

# Initialize models globally
yolo_model = get_yolo_model(model_path=icon_detect_model)
caption_model_processor = None  # Initialize as None first

if icon_caption_model == 'florence2':
    try:
        # First ensure the config exists
        if not os.path.exists("weights/icon_caption_florence"):
            os.makedirs("weights/icon_caption_florence")
            
        # Download and save the model with correct config
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained("microsoft/Florence-2-base-ft", trust_remote_code=True)
        config.vision_config.model_type = "davit"  # Set the required vision model type
        config.save_pretrained("weights/icon_caption_florence")
        
        model, processor = get_caption_model_processor("florence2")
        caption_model_processor = {'model': model, 'processor': processor}
        print("Successfully loaded Florence model")
    except Exception as e:
        print(f"Error loading Florence model: {e}")
        raise
elif icon_caption_model == 'blip2':
    caption_model_processor = get_caption_model_processor(
        model_name="blip2", 
        model_name_or_path="Salesforce/blip2-opt-2.7b"
    )

def process(
    image_input,
    box_threshold,
    iou_threshold,
    use_paddleocr,
    imgsz,
    icon_process_batch_size,
) -> Optional[Image.Image]:
    image_save_path = 'imgs/saved_image_demo.png'
    image_input.save(image_save_path)
    image = Image.open(image_save_path)
    box_overlay_ratio = image.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }
    # import pdb; pdb.set_trace()

    ocr_bbox_rslt, is_goal_filtered = check_ocr_box(image_save_path, display_img = False, output_bb_format='xyxy', goal_filtering=None, easyocr_args={'paragraph': False, 'text_threshold':0.9}, use_paddleocr=use_paddleocr)
    text, ocr_bbox = ocr_bbox_rslt
    # print('prompt:', prompt)
    dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_save_path, yolo_model, 
        BOX_TRESHOLD=box_threshold, 
        output_coord_in_ratio=True, 
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config, 
        caption_model_processor=caption_model_processor, 
        ocr_text=text,
        iou_threshold=iou_threshold, 
        imgsz=imgsz, 
        batch_size=icon_process_batch_size
    )
    
    image = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
    print('finish processing')
    
    # Create a formatted response that includes coordinates
    formatted_response = []
    for i, item in enumerate(parsed_content_list):
        coords = label_coordinates.get(str(i), None)
        response_item = {
            'type': item['type'],
            'content': item['content'],
            'interactivity': item['interactivity'],
            'coordinates': {
                'x': int(coords[0]) if coords else None,  # x coordinate
                'y': int(coords[1]) if coords else None,  # y coordinate
                'width': int(coords[2]) if coords else None,  # width
                'height': int(coords[3]) if coords else None  # height
            }
        }
        formatted_response.append(response_item)
    
    # Convert to string with nice formatting
    response_text = '\n'.join([
        f"type: {x['type']}, "
        f"content: {x['content']}, "
        f"interactivity: {x['interactivity']}, "
        f"coordinates: (x={x['coordinates']['x']}, y={x['coordinates']['y']}, "
        f"width={x['coordinates']['width']}, height={x['coordinates']['height']})"
        for x in formatted_response
    ])
    
    return image, response_text

with gr.Blocks() as demo:
    gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            image_input_component = gr.Image(
                type='pil', label='Upload image')
            # set the threshold for removing the bounding boxes with low confidence, default is 0.05
            box_threshold_component = gr.Slider(
                label='Box Threshold', minimum=0.01, maximum=1.0, step=0.01, value=0.05)
            # set the threshold for removing the bounding boxes with large overlap, default is 0.1
            iou_threshold_component = gr.Slider(
                label='IOU Threshold', minimum=0.01, maximum=1.0, step=0.01, value=0.1)
            use_paddleocr_component = gr.Checkbox(
                label='Use PaddleOCR', value=False)
            imgsz_component = gr.Slider(
                label='Icon Detect Image Size', minimum=640, maximum=3200, step=32, value=1920)
            icon_process_batch_size_component = gr.Slider(
                label='Icon Process Batch Size', minimum=1, maximum=256, step=1, value=64)
            submit_button_component = gr.Button(
                value='Submit', variant='primary')
        with gr.Column():
            image_output_component = gr.Image(type='pil', label='Image Output')
            text_output_component = gr.Textbox(label='Parsed screen elements', placeholder='Text Output')

    submit_button_component.click(
        fn=process,
        inputs=[
            image_input_component,
            box_threshold_component,
            iou_threshold_component,
            use_paddleocr_component,
            imgsz_component,
            icon_process_batch_size_component
        ],
        outputs=[image_output_component, text_output_component]
    )

# demo.launch(debug=False, show_error=True, share=True)
demo.launch(share=True, server_port=7861, server_name='0.0.0.0')


# python gradio_demo.py --icon_detect_model weights/icon_detect/best.pt --icon_caption_model florence2
# python gradio_demo.py --icon_detect_model weights/icon_detect_v1_5/model_v1_5.pt --icon_caption_model florence2

# Create FastAPI app
app = FastAPI()

# Add a new endpoint for direct API access
@app.post("/api/parse_image")
async def parse_image(file: UploadFile = File(...)):
    # Save uploaded image
    image_save_path = 'imgs/saved_image_api.png'
    image_content = await file.read()
    with open(image_save_path, "wb") as f:
        f.write(image_content)
    
    image = Image.open(image_save_path)
    box_overlay_ratio = image.size[0] / 3200
    draw_bbox_config = {
        'text_scale': 0.8 * box_overlay_ratio,
        'text_thickness': max(int(2 * box_overlay_ratio), 1),
        'text_padding': max(int(3 * box_overlay_ratio), 1),
        'thickness': max(int(3 * box_overlay_ratio), 1),
    }

    ocr_bbox_rslt, _ = check_ocr_box(
        image_save_path, 
        display_img=False, 
        output_bb_format='xyxy', 
        goal_filtering=None, 
        easyocr_args={'paragraph': False, 'text_threshold':0.9}, 
        use_paddleocr=False
    )
    text, ocr_bbox = ocr_bbox_rslt

    _, label_coordinates, parsed_content_list = get_som_labeled_img(
        image_save_path, 
        yolo_model, 
        BOX_TRESHOLD=0.05,  # default values
        output_coord_in_ratio=False,  # return actual pixel coordinates
        ocr_bbox=ocr_bbox,
        draw_bbox_config=draw_bbox_config, 
        caption_model_processor=caption_model_processor, 
        ocr_text=text,
        iou_threshold=0.1,
        imgsz=1920,
        batch_size=64
    )

    # Create structured response
    response_data = []
    for i, item in enumerate(parsed_content_list):
        coords = label_coordinates.get(str(i), None)
        response_item = {
            'type': item['type'],
            'content': item['content'],
            'interactivity': item['interactivity'],
            'coordinates': {
                'x': int(coords[0]) if coords else None,
                'y': int(coords[1]) if coords else None,
                'width': int(coords[2]) if coords else None,
                'height': int(coords[3]) if coords else None
            }
        }
        response_data.append(response_item)

    return {"elements": response_data}

# Mount the Gradio app
gradio_app = gr.mount_handlers(demo.app)
app = mount_gradio_app(app, gradio_app, path="/")
