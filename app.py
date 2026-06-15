import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import time

# Import the stylize_image function from style_transfer.py
from style_transfer import stylize_image, download_if_missing

app = Flask(__name__)

MODELS = {
    'starry_night': 'https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/starry_night.t7',
    'la_muse': 'https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/la_muse.t7',
    'mosaic': 'https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/mosaic.t7',
    'candy': 'https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/candy.t7',
    'the_scream': 'https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/the_scream.t7',
    'udnie': 'https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/udnie.t7'
}
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')
app.config['OUTPUT_FOLDER'] = os.path.join('static', 'outputs')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stylize', methods=['POST'])
def stylize():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        # Get the requested model, default to starry_night
        model_name = request.form.get('model', 'starry_night')
        if model_name not in MODELS:
            model_name = 'starry_night'
            
        # Make sure the model exists
        model_path = os.path.join('models', f"{model_name}.t7")
        download_if_missing(model_path, MODELS[model_name])
        
        # Generate output filename
        base_name = os.path.splitext(filename)[0]
        timestamp = int(time.time())
        output_filename = f"art_{base_name}_{timestamp}.jpg"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
        
        # Perform style transfer
        try:
            success = stylize_image(input_path, model_path, output_path)
            if success:
                # Return the URL to the stylized image
                output_url = f"/{app.config['OUTPUT_FOLDER']}/{output_filename}"
                # Convert backslashes to forward slashes for URLs on Windows
                output_url = output_url.replace('\\', '/')
                return jsonify({'success': True, 'output_url': output_url})
            else:
                return jsonify({'error': 'Style transfer failed during processing.'}), 500
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg'}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
