from flask import Flask, render_template, request, send_file, jsonify
from PIL import Image
import numpy as np
import io
import random
import qrcode
from math import log10, sqrt
import base64  # Importing the base64 module

app = Flask(__name__)

# Normalize Image Matrix to Quantum States (simulated as rotation angles)
def normalize_to_quantum_state(value):
    return (value / 255.0) * (2 * np.pi)

# Apply Quantum Rotation (simulated)
def apply_quantum_rotation(value, angle):
    return int((value + angle) % 256)

# Function to encode text into image using QNN-inspired techniques
def encode_image(img, data):
    img = img.convert('RGBA')
    pixels = np.array(img)
    data += chr(0)  # Null character to mark the end of the text

    data_bits = ''.join([format(ord(char), '08b') for char in data])
    data_index = 0
    pixel_count = pixels.shape[0] * pixels.shape[1]

    for i in range(0, pixel_count, 2):
        for j in range(2):
            if data_index < len(data_bits):
                r, g, b, a = pixels[(i + j) // pixels.shape[1], (i + j) % pixels.shape[1]]

                r_angle = normalize_to_quantum_state(r)
                r_rotated = apply_quantum_rotation(r, r_angle) 
                r_rotated = (r_rotated & ~1) | int(data_bits[data_index])
                data_index += 1

                if data_index < len(data_bits):
                    g_angle = normalize_to_quantum_state(g)
                    g_rotated = apply_quantum_rotation(g, g_angle)
                    g_rotated = (g_rotated & ~1) | int(data_bits[data_index])
                    data_index += 1

                if data_index < len(data_bits):
                    b_angle = normalize_to_quantum_state(b)
                    b_rotated = apply_quantum_rotation(b, b_angle)
                    b_rotated = (b_rotated & ~1) | int(data_bits[data_index])
                    data_index += 1

                pixels[(i + j) // pixels.shape[1], (i + j) % pixels.shape[1]] = (r_rotated, g_rotated, b_rotated, a)

    return Image.fromarray(pixels)

def decode_image(img):
    img = img.convert('RGBA')
    pixels = np.array(img)
    data_bits = ""
    height, width, _ = pixels.shape

    # Traverse through pixels to extract LSB
    for row in range(height):
        for col in range(width):
            r, g, b, a = pixels[row, col]
            data_bits += str(r & 1)
            data_bits += str(g & 1)
            data_bits += str(b & 1)

    # Convert bits to characters
    data = []
    for i in range(0, len(data_bits), 8):
        byte = data_bits[i:i + 8]
        if len(byte) < 8:
            break  # Avoid processing incomplete bytes
        char = chr(int(byte, 2))
        if char == chr(0):  # Stop at null character
            break
        data.append(char)

    return ''.join(data)


# Generate QR Code
def generate_qr(data):
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(data)
    qr.make(fit=True)
    qr_img = qr.make_image(fill='black', back_color='white')

    buffer = io.BytesIO()
    qr_img.save(buffer, format='PNG')
    buffer.seek(0)
    return buffer

# Calculate PSNR
def calculate_psnr(original, modified):
    original = np.array(original.convert('L'))  # Convert to grayscale
    modified = np.array(modified.convert('L'))  # Convert to grayscale

    mse = np.mean((original - modified) ** 2)
    if mse == 0:  # Prevent divide by zero
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/encode', methods=['POST'])
def encode():
    image_file = request.files['image']
    secret_data = request.form['secret_data']

    if image_file and secret_data:
        img = Image.open(image_file)
        encoded_img = encode_image(img, secret_data)

        buffer = io.BytesIO()
        encoded_img.save(buffer, format='PNG')
        buffer.seek(0)

        return send_file(buffer, mimetype='image/png', as_attachment=True, download_name='encoded_image.png')
    else:
        return "Error: Missing image or data", 400

@app.route('/decode', methods=['POST'])
def decode():
    image_file = request.files['image']
    img = Image.open(image_file)
    decoded_data = decode_image(img)
    return f"Decoded Data: {decoded_data}"

@app.route('/generate_qr', methods=['POST'])
def generate_qr_code():
    data = request.get_json()  # Parse JSON data
    qr_data = data.get('qr_data')  # Get 'qr_data' from JSON
    if not qr_data:
        return jsonify({'error': 'No data provided'}), 400

    # Generate QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(qr_data)
    qr.make(fit=True)

    # Create image
    img = qr.make_image(fill_color="black", back_color="white")
    buffered = io.BytesIO()  # Use the correct BytesIO here
    img.save(buffered, format="PNG")
    qr_code_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')  # Convert to Base64

    return jsonify({'qr_code': qr_code_base64})

@app.route('/calculate_psnr', methods=['POST'])
def psnr():
    original_file = request.files['original']
    modified_file = request.files['modified']

    original_img = Image.open(original_file)
    modified_img = Image.open(modified_file)
    # org_psnr = calculate_psnr(original_img)
    # mod_psnr = calculate_psnr(modified_img)
    psnr_value = calculate_psnr(original_img, modified_img)
    return jsonify({'psnr': f"{psnr_value:.2f} dB"})


if __name__ == '__main__':
    app.run(debug=True)
