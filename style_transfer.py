import cv2
import os
import sys
import urllib.request
import ssl

# Fix for potential SSL certificate errors when downloading
ssl._create_default_https_context = ssl._create_unverified_context

# Default style model URL
STYLE_MODEL_URL = "https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/starry_night.t7"

def download_if_missing(filename, url):
    """Download a file from URL if it does not exist locally."""
    if not os.path.exists(filename):
        print(f"Downloading {os.path.basename(filename)}...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")

def stylize_image(content_path, model_path, output_name):
    """Applies neural style transfer to a single image and saves the result."""
    # Validate inputs
    if not os.path.exists(content_path):
        print(f"Error: Content image not found: {content_path}")
        return False
    if not os.path.exists(model_path):
        print(f"Error: Style model not found: {model_path}")
        return False

    net = cv2.dnn.readNetFromTorch(model_path)

    image = cv2.imread(content_path)
    if image is None:
        print(f"Error: Failed to load image (may be corrupt): {content_path}")
        return False

    h, w = image.shape[:2]

    # Resize to max 600px width/height to avoid memory/performance issues
    scale = 600.0 / max(h, w)
    if scale < 1.0:
        image = cv2.resize(image, None, fx=scale, fy=scale)
        h, w = image.shape[:2]

    # Create a blob from the image and perform a forward pass
    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
    net.setInput(blob)
    output = net.forward()

    # Post-process the output
    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680

    # output is currently in float format, normalize to 0-255 and convert to uint8
    output = output.transpose(1, 2, 0)
    output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Make sure the output directory exists
    output_dir = os.path.dirname(output_name)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    cv2.imwrite(output_name, output)
    return True


if __name__ == "__main__":
    # --- Single Image Mode ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    # Determine the input image path
    if len(sys.argv) > 1:
        input_image = sys.argv[1]
    else:
        # Default: look for the first image in dataset/input
        default_input_dir = os.path.join(SCRIPT_DIR, "dataset", "input")
        if os.path.isdir(default_input_dir):
            images = [f for f in os.listdir(default_input_dir)
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if images:
                input_image = os.path.join(default_input_dir, images[0])
            else:
                print("Error: No images found in dataset/input/")
                print("Usage: python style_transfer.py <path_to_image> [model.t7]")
                sys.exit(1)
        else:
            print("Error: No image specified and dataset/input/ folder not found.")
            print("Usage: python style_transfer.py <path_to_image> [model.t7]")
            sys.exit(1)

    # Resolve full path
    if not os.path.isabs(input_image):
        input_image = os.path.abspath(input_image)

    if not os.path.exists(input_image):
        print(f"Error: Image not found: {input_image}")
        sys.exit(1)

    # Determine the model path
    models_dir = os.path.join(SCRIPT_DIR, "models")
    os.makedirs(models_dir, exist_ok=True)

    if len(sys.argv) > 2:
        model_name = sys.argv[2]
        model_path = os.path.join(models_dir, model_name)
    else:
        model_path = os.path.join(models_dir, "starry_night.t7")
        
    if not os.path.exists(model_path):
        # We only auto-download starry_night for fallback
        if "starry_night.t7" in model_path:
            download_if_missing(model_path, STYLE_MODEL_URL)
        else:
            print(f"Error: Model not found at {model_path}.")
            print("Please ensure the model exists in the models/ directory.")
            sys.exit(1)

    # Generate output filename
    base_name = os.path.splitext(os.path.basename(input_image))[0]
    model_base = os.path.splitext(os.path.basename(model_path))[0]
    output_dir = os.path.join(SCRIPT_DIR, "dataset", "csv_output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"art_{base_name}_{model_base}.jpg")

    print("==================================================")
    print("  Neural Style Transfer - Single Image Mode")
    print("==================================================")
    print(f"  Input:  {input_image}")
    print(f"  Model:  {model_path}")
    print(f"  Output: {output_path}")
    print("==================================================")
    print("Applying Neural Style Transfer...", end=" ", flush=True)

    import time
    t0 = time.time()
    success = stylize_image(input_image, model_path, output_path)
    t1 = time.time()

    if success:
        print(f"Done! ({t1 - t0:.2f}s)")
        print(f"\nStylized image saved to: {output_path}")
    else:
        print("Failed!")
        sys.exit(1)
