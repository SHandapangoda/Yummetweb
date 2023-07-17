from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import openai
import os
import numpy as np
from PIL import Image
import cv2
from dotenv import load_dotenv



load_dotenv()  # take environment variables from .env.




app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = './uploads'

# Load the YOLO model
model = YOLO('best1.pt')

GPT_CALLS_ENABLED = True

# Dictionary to store the hashes of the images and their output paths
image_hashes = {}

def draw_boxes(image, results):
    for result in results:
        for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
            label = model.names[int(cls)]
            bbox = [int(coord) for coord in box.cpu().numpy()]
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(image, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    return image



def analyze_image(image_path, inflow_chemicals, inflow_quantities):
    # Analyze image
    img = Image.open(image_path).convert("RGB")
    results = model.predict(img, conf=0.01)
    
    # Create an output file name based on input file name
    input_filename = os.path.basename(image_path)
    filename_without_extension, extension = os.path.splitext(input_filename)
    output_filename = f'{filename_without_extension}_output{extension}'
    output_path = os.path.join('static/predicted', output_filename)
    output_path = output_path.replace('\\', '/')
    
    # If processed image doesn't exist, process it and save the result
    if not os.path.exists(output_path):
        image_with_boxes = draw_boxes(cv2.imread(image_path), results)
        
        # Check if output directory exists
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        cv2.imwrite(output_path, image_with_boxes)
    
    # Continue with the rest of the function..

    # Object detection and sorting
    detected_objects = []
    for r in results:
        for box, cls in zip(r.boxes.xyxy, r.boxes.cls):
            label = model.names[int(cls)]
            bbox = [int(coord) for coord in box.cpu().numpy()]
            detected_objects.append({'label': label, 'bbox': bbox})

    sorted_objects = sorted(detected_objects, key=lambda obj: (obj['bbox'][0], obj['bbox'][1]))

    # Flow creation
    flow = [sorted_objects[0]]
    remaining_objects = sorted_objects[1:]
    while remaining_objects:
        next_object = find_next_object(flow[-1], remaining_objects)
        flow.append(next_object)
        remaining_objects.remove(next_object)

    # Prompt creation
    prompt = "In a process flow diagram, the inflow of chemicals includes "
    for chemical, quantity in zip(inflow_chemicals, inflow_quantities):
        prompt += f"{chemical} with a mass ratio of {quantity}, "
    prompt += ". The components in order from left to right are: "
    for item in flow:
        prompt += f"{item['label']}, "
    prompt += ". Given this information, what could be the probable outflow of chemicals and their possible mass ratios?, and please provide only outflow chemicals with their mass ratios"

    # GPT-4 API call
    message = ''
    if GPT_CALLS_ENABLED:
        openai.api_key = os.getenv("OPENAI_KEY")
        response = openai.ChatCompletion.create(
            model="gpt-4",
            max_tokens=1024,
            messages=[{"role":"user", "content": f"{prompt}"}]
        )

        message = response['choices'][0]['message']['content']

    return message, output_path

def find_next_object(current_object, remaining_objects):
    # Find objects that are vertically aligned (same x)
    vertically_aligned = [obj for obj in remaining_objects if obj['bbox'][0] == current_object['bbox'][0]]
    if vertically_aligned:
        # If there are vertically aligned objects, return the one with the smallest y (top-most)
        return min(vertically_aligned, key=lambda obj: obj['bbox'][1])
    else:
        # If no vertically aligned objects, return the object with the smallest x (left-most)
        return min(remaining_objects, key=lambda obj: obj['bbox'][0])

@app.route('/', methods=['GET', 'POST'])
def upload_and_analyze():
    if request.method == 'POST':
        inflow_chemicals = request.form.get('inflow_chemicals').split(',')
        inflow_quantities = request.form.get('inflow_quantities').split(',')
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Generate the extension of the output file based on the input file
        file_extension = os.path.splitext(filename)[1]

        message, img_path = analyze_image(os.path.join(app.config['UPLOAD_FOLDER'], filename), inflow_chemicals, inflow_quantities)

        # Create the name of the output file
        output_file = f'{os.path.splitext(filename)[0]}_output{file_extension}'

        return render_template('webUI.html', message=message, img_path=f'static/predicted/{output_file}')
    return render_template('webUI.html', message=None, img_path=None)


if __name__ == '__main__':
    app.run()
