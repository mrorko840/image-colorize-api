import os
import io
import sys
import torch
import numpy as np
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS  # Import CORS
from PIL import Image
from colorizers import siggraph17, preprocess_img, postprocess_tens
from werkzeug.exceptions import RequestEntityTooLarge

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

app = Flask(__name__)
CORS(app)

# Increase max content length (set it to 10MB or any desired limit)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB

# Handle large file size error
@app.errorhandler(RequestEntityTooLarge)
def handle_file_size_error(error):
    return jsonify({'error': 'File is too large. Maximum allowed size is 10MB.'}), 413

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
colorizer = siggraph17(pretrained=True).eval().to(device)


def colorize_image(image):
    """ কালো-সাদা ছবি রঙিন করে এবং PIL ইমেজ হিসেবে রিটার্ন করে """
    try:
        # PIL Image -> NumPy Array (RGB ফরম্যাটে)
        img_array = np.array(image)

        # কালো-সাদা ছবির প্রিপ্রসেসিং
        tens_l_orig, tens_l_rs = preprocess_img(img_array, HW=(256, 256))

        # ইমেজ প্রসেসিং
        with torch.no_grad():
            ab_output = colorizer(tens_l_rs.to(device)).cpu()

        # আউটপুট ছবি প্রোসেস করা
        out_img = postprocess_tens(tens_l_orig, ab_output)

        # NumPy Array থেকে PIL ইমেজ কনভার্ট করা
        out_img = (out_img * 255).astype(np.uint8)
        return Image.fromarray(out_img)

    except Exception as e:
        print(f"Error in colorization: {e}")
        return None
@app.route('/', methods=["GET"])
def welcome():
    return "<h1 align='center'>API is Running</h1>"

@app.route('/colorize', methods=['POST'])
def colorize():
    print("API called...")
    """ API Endpoint যেখানে ক্লায়েন্ট ছবি পাঠাবে, এবং রঙিন ছবি পাবে """
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    print("File received:", file.filename)

    try:
        image = Image.open(file).convert("RGB")  # ইমেজ ওপেন করে RGB তে কনভার্ট করা
    except Exception as e:
        return jsonify({'error': f'Invalid image file: {e}'}), 400

    # ইমেজ প্রসেসিং এবং কালারাইজ করা
    colorized_img = colorize_image(image)

    if colorized_img is None:
        return jsonify({'error': 'Failed to process image'}), 500

    # ইমেজকে বাইনারি ফরম্যাটে পাঠানোর জন্য মেমোরি বাফারে সংরক্ষণ
    img_io = io.BytesIO()
    colorized_img.save(img_io, format="PNG")
    img_io.seek(0)

    print("Colorization complete...")
    return send_file(img_io, mimetype='image/png')


if __name__ == '__main__':
    # Increased timeout for better handling of large files
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True, use_reloader=False)