import cv2
import os
import urllib.request
import ssl

# Fix for potential SSL certificate errors when downloading
ssl._create_default_https_context = ssl._create_unverified_context

def download_if_missing(filename, url):
    if not os.path.exists(filename):
        print(f"Downloading {os.path.basename(filename)}...")
        urllib.request.urlretrieve(url, filename)

def stylize_image(content_path, model_path, output_name):
    """Applies neural style transfer to a single image and saves the result."""
    net = cv2.dnn.readNetFromTorch(model_path)
    
    image = cv2.imread(content_path)
    if image is None:
        print(f"Failed to load image: {content_path}")
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
    
    cv2.imwrite(output_name, output)
    return True

