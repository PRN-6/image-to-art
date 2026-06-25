<!-- Page 1: Inner Page -->
# AI-Based Image-to-Art Neural Style Transfer

**USN:-** [Enter your USN here]  
**Name:-** Prinson [Add full name]  
**Guide Name:-** [Enter Guide Name here]  

---
<!-- Page 2: Certificate Page -->
# Certificate

This is to certify that the project work entitled **"AI-Based Image-to-Art Neural Style Transfer"** is a bonafide work carried out by **Prinson [Add full name]** (USN: **[Enter your USN here]**) in partial fulfillment for the award of the degree of Master of Computer Applications.

**Guide:** [Enter Guide Name here]

**Head of Department:** [Enter HOD Name here]

**Principal:** [Enter Principal Name here]

---
<!-- Page 3: Acknowledgement -->
# Acknowledgement

I would like to express my profound gratitude to everyone who supported me throughout the course of this project.

First and foremost, I express my sincere gratitude to my guide, **[Enter Guide Name here]**, for their invaluable guidance, continuous support, and constructive feedback throughout the development of this project. Their deep insights into the field of Machine Learning and Computer Vision were instrumental in shaping the direction of this research.

I also extend my sincere thanks to the Head of the Department, **[Enter HOD Name here]**, and the Principal, **[Enter Principal Name here]**, for providing the necessary infrastructure, resources, and an encouraging environment at AJIET.

Furthermore, I am deeply thankful to the Department of MCA, AJIET, and all the faculty members whose teachings provided the foundational knowledge required to undertake this complex technical endeavor. Finally, I would like to thank my family and friends for their unwavering encouragement.

---
<!-- Page 4: Abstract/ Synopsis -->
# Abstract / Synopsis

In the modern digital era, the intersection of artificial intelligence and creative arts has led to groundbreaking innovations, reshaping how we perceive, create, and interact with visual media. This project, "AI-Based Image-to-Art Neural Style Transfer", explores and implements a highly sophisticated deep learning technique that takes a standard digital photograph and a style reference image (such as a famous historical painting) and mathematically synthesizes them into a single coherent output. The resulting image retains the structural and semantic integrity of the original photograph but is rendered utilizing the colors, textures, and brushstrokes of the chosen artwork.

Historically, Neural Style Transfer (NST), introduced by Gatys et al., relied on computationally expensive iterative optimization processes, making it inaccessible for real-time applications or consumer-grade hardware. To overcome this limitation, this project leverages "Fast Neural Style Transfer" techniques, utilizing pre-trained Convolutional Neural Networks (CNNs) optimized for rapid inference. By employing feed-forward networks, the application effectively bypasses the traditional bottlenecks of iterative image generation, reducing processing times from minutes to mere fractions of a second.

The system is developed using Python, OpenCV's Deep Neural Network (DNN) module, NumPy, and the Flask web framework. It democratizes AI-assisted art creation, ensuring users do not need high-end GPUs or cloud computing resources to generate high-quality stylized images. The project features two main components: a robust command-line batch-processing pipeline capable of processing large datasets automatically via CSV metadata, and an intuitive, interactive Flask-based web application that allows users to upload images and apply styles in real-time. This dual-faceted approach ensures the system is versatile enough for both bulk content generation by digital artists and casual experimentation by everyday users.

---
<!-- Page 5: Table of Contents -->
# Table of Contents
1. Introduction
   1.1 Overview of Artificial Intelligence in Art
   1.2 The Concept of Neural Style Transfer
   1.3 Evolution of Style Transfer Techniques
   1.4 Scope and Organization of the Report
2. Problem Statement
   2.1 Traditional Limitations
   2.2 The Accessibility Gap
   2.3 Detailed Problem Definition
3. Objectives
4. Proposed Methodology
   4.1 System Architecture
   4.2 Deep Learning Model Selection
   4.3 Image Pre-processing and Blob Creation
   4.4 Forward Propagation (Inference)
   4.5 Post-processing and Denormalization
   4.6 Application Workflow
5. Dataset Description
   5.1 Input Dataset Structure
   5.2 CSV Metadata Processing
   5.3 Style Model Configurations
6. Tools and Technologies Used
   6.1 Python Ecosystem
   6.2 OpenCV and DNN Module
   6.3 NumPy Matrix Operations
   6.4 Flask Web Framework
7. Implementation Plan & Algorithm Used
   7.1 Algorithm Explanation (Fast Neural Style Transfer)
   7.2 Mathematical Foundations
   7.3 Software Development Life Cycle (SDLC)
8. Output and Expected Results
   8.1 Batch Processing Evaluation
   8.2 Web Interface Usability
   8.3 Output Screenshots
9. Conclusion and Future Scope
   9.1 Summary of Contributions
   9.2 Future Enhancements
10. References

---
<!-- Header as requested -->
AJIET Department of MCA 2026

## 1. Introduction

### 1.1 Overview of Artificial Intelligence in Art
Artificial Intelligence (AI) has transcended its traditional boundaries of data processing and logical computation to enter the realm of creativity and art. Over the past decade, advancements in Deep Learning and Computer Vision have enabled machines to analyze, understand, and even replicate human artistic expressions. Generative AI, a subfield dedicated to creating new content from learned data distributions, has paved the way for automated music composition, text generation, and notably, visual art synthesis. The integration of AI into the artistic process has sparked a new era of digital creativity, empowering artists with novel tools and allowing individuals without formal artistic training to create compelling visual pieces.

### 1.2 The Concept of Neural Style Transfer
Neural Style Transfer (NST) is one of the most prominent and visually striking applications of deep learning in computer vision. At its core, NST is a technique that blends two images: a "content image" (typically a photograph) and a "style image" (usually a painting by a renowned artist like Vincent van Gogh or Pablo Picasso). The objective is to generate a third image, the "output image," which preserves the fundamental structure, objects, and layout of the content image while adopting the color palette, textures, brushstrokes, and overall aesthetic of the style image. This is achieved by leveraging the hierarchical feature extraction capabilities of Convolutional Neural Networks (CNNs). Deep layers of the network capture the high-level semantic content, while shallower layers capture localized texture and style patterns.

### 1.3 Evolution of Style Transfer Techniques
The foundational paper on Neural Style Transfer by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge in 2015 demonstrated that deep neural networks could separate and recombine the content and style of arbitrary images. However, their approach utilized an iterative optimization algorithm. Starting with a noise image, the system would iteratively update the pixel values to minimize a complex loss function combining "content loss" and "style loss". While the visual results were unprecedented, the computational cost was immense, often requiring several minutes on a high-end GPU to process a single image.

To address this, researchers led by Justin Johnson developed "Perceptual Losses for Real-Time Style Transfer and Super-Resolution" (2016). This introduced the concept of "Fast Neural Style Transfer." Instead of optimizing the image pixels iteratively, they trained a feed-forward Image Transformation Network for each specific style. Once trained, applying the style to a new image simply required a single forward pass through the network. This breakthrough reduced inference times by orders of magnitude, making real-time processing and desktop applications feasible. This project is built entirely on the foundations of Fast Neural Style Transfer.

### 1.4 Scope and Organization of the Report
The scope of this project encompasses the development of an end-to-end, locally hosted software application capable of executing Fast Neural Style Transfer. It includes a backend processing engine, a batch-processing tool utilizing CSV metadata, and a web-based frontend interface. The report is organized to logically guide the reader through the project's lifecycle, beginning with the problem definition, outlining the methodology and architectural design, detailing the implementation and algorithmic specifics, and concluding with an evaluation of the system's outputs.

## 2. Problem Statement

### 2.1 Traditional Limitations
Before the advent of deep learning, applying an artistic style to a photograph relied heavily on manual digital painting techniques or heuristic-based image filters. Traditional filters (such as those found in early versions of Adobe Photoshop or basic photo editing apps) applied uniform algorithmic changes, such as color mapping, edge detection, or pixelation. These methods failed to understand the semantic content of the image; a filter would apply a "canvas" texture indiscriminately across the entire image, lacking the contextual awareness of a human artist who varies their brushstrokes based on the object they are painting.

### 2.2 The Accessibility Gap
While Gatys's original NST algorithm solved the contextual awareness problem, it introduced a significant accessibility barrier. The iterative optimization process required substantial computational power and time. To generate a single piece of art, users needed access to expensive hardware (like NVIDIA Tesla or high-end RTX graphics cards) and required the technical knowledge to set up complex Python environments, configure CUDA drivers, and manage memory constraints. This restricted the technology to researchers and high-end creative studios, alienating casual users and independent creators.

### 2.3 Detailed Problem Definition
The central problem this project addresses is the need for a rapid, accessible, and context-aware image stylization tool that operates efficiently on consumer-grade hardware without relying on paid cloud APIs. 

Specifically, the project seeks to solve:
1. The inability to rapidly apply artistic styles to large datasets of images without manual intervention.
2. The high computational overhead associated with iterative Neural Style Transfer.
3. The lack of an intuitive, locally hosted user interface that abstracts the underlying deep learning complexities from the end-user.

By solving these issues, the project aims to bridge the gap between advanced deep learning research and practical, everyday artistic application.

## 3. Objectives

The primary objective of this project is to construct an automated, highly efficient system for artistic image generation. To achieve this overarching goal, the project is divided into several specific, measurable objectives:

1. **Implement Fast Neural Style Transfer:** To utilize pre-trained deep feed-forward Convolutional Neural Networks to perform style transfer in a single pass, drastically reducing the inference time compared to optimization-based methods.
2. **Hardware Optimization:** To leverage the OpenCV Deep Neural Network (`dnn`) module to ensure the models run efficiently on standard Central Processing Units (CPUs), completely eliminating the strict requirement for dedicated GPUs, thereby maximizing accessibility.
3. **Automated Batch Processing Pipeline:** To develop a robust scripting module capable of parsing CSV datasets containing image paths, metadata, and assigned styles. The system must autonomously process these datasets, applying the correct models and logging success/failure rates, effectively serving as an automated bulk-art generator.
4. **Interactive Web Application:** To design and deploy a Flask-based web interface that provides a seamless, user-friendly experience. Users must be able to upload personal images, select from a variety of pre-trained styles, and receive the stylized output directly in their web browser in near real-time.
5. **Cross-Platform Compatibility:** To ensure the entire codebase relies on standardized, cross-platform libraries (Python, OpenCV) so the application functions identically across Windows, macOS, and Linux environments.

## 4. Proposed Methodology

### 4.1 System Architecture
The system architecture is designed to be modular and scalable, separated into three primary layers: the User Interface Layer, the Application Logic Layer, and the Deep Learning Engine Layer.
- **User Interface Layer:** Comprises the HTML/CSS/JavaScript frontend served via Flask, allowing image uploads and style selection.
- **Application Logic Layer:** Handles routing, file system management (saving uploads, serving outputs), CSV parsing for batch tasks, and error handling.
- **Deep Learning Engine Layer:** Encapsulates the core `style_transfer.py` script. It handles model loading, image tensor conversions, memory management, and the actual execution of the OpenCV `dnn.forward()` pass.

### 4.2 Deep Learning Model Selection
Instead of training models from scratch, which requires massive datasets (like MS COCO) and extensive GPU clusters, the project adopts Transfer Learning. It utilizes pre-trained Image Transformation Networks saved in the Torch `.t7` format. These models were originally trained by Justin Johnson using a VGG-16 network backbone as a "loss network". The available styles include:
*   **Starry Night:** Mimics Vincent van Gogh's iconic swirling skies and vibrant blues/yellows.
*   **La Muse:** Adopts the cubist-inspired, vivid colors of Pablo Picasso's work.
*   **Mosaic:** Applies a segmented, tiled stained-glass aesthetic.
*   **Candy:** Introduces vibrant, high-contrast, abstract artistic textures.
*   **The Scream:** Mimics Edvard Munch's famous painting with fluid lines and distinct color palettes.
*   **Udnie:** Applies Francis Picabia's abstract, geometric stylizations.

### 4.3 Image Pre-processing and Blob Creation
Deep learning models cannot process raw image files directly; they require specifically formatted numerical tensors.
1.  **Resizing:** To prevent Out-Of-Memory (OOM) errors and ensure rapid processing on CPUs, the input image is dynamically resized so its maximum dimension (width or height) does not exceed 600 pixels, while maintaining the original aspect ratio.
2.  **Blob Conversion:** The image is converted into a 4-dimensional "blob" (Binary Large Object) using `cv2.dnn.blobFromImage`. The dimensions represent `(Batch Size, Channels, Height, Width)`.
3.  **Mean Subtraction:** During blob creation, specific mean values (e.g., 103.939, 116.779, 123.680 for the B, G, R channels) are subtracted from the image. This centers the data distribution, matching the exact normalization process used when the VGG networks were originally trained on the ImageNet dataset, ensuring optimal feature extraction.

### 4.4 Forward Propagation (Inference)
Once the blob is prepared, it is set as the input to the loaded `.t7` neural network (`net.setInput(blob)`). The `net.forward()` command is executed. This initiates a single forward pass through the network's convolutional layers, residual blocks, and transposed convolutional layers (up-sampling). The network mathematically applies the stylistic transformations learned during its training phase, outputting a new tensor representing the stylized image.

### 4.5 Post-processing and Denormalization
The output from the neural network is a multidimensional tensor of floating-point numbers that must be converted back into a viewable image.
1.  **Reshaping:** The tensor is reshaped and transposed from the network format `(Channels, Height, Width)` back to the standard OpenCV image format `(Height, Width, Channels)`.
2.  **Mean Re-addition:** The mean values subtracted during pre-processing (103.939, 116.779, 123.680) are mathematically added back to the respective color channels to restore accurate color representation.
3.  **Normalization and Type Conversion:** The floating-point values are clipped and normalized into the standard 0 to 255 range and converted into an 8-bit unsigned integer type (`uint8`), rendering it a standard RGB (or BGR in OpenCV's case) image array.

### 4.6 Application Workflow
The workflow supports two modes:
*   **Batch Mode:** The script reads `image_dataset.csv`, extracts the image URL/path, downloads it if necessary, applies the specified model, saves the result to `dataset/csv_output`, and generates an evaluation report regarding total time and average time per image.
*   **Web Mode:** A user navigates to the Flask application (`index.html`), uploads a photo via an HTML form, and selects a style from a dropdown menu. The Flask route securely saves the file, invokes the `stylize_image()` function, and upon completion, returns the URL of the synthesized image to be displayed asynchronously on the webpage.

## 5. Dataset Description

### 5.1 Input Dataset Structure
For the automated batch-processing feature of the application, a structured dataset approach is utilized. Rather than manually passing single images via the command line, the system relies on a dataset folder (`dataset/input/`) containing various test images (e.g., landscapes, portraits, architecture). 

### 5.2 CSV Metadata Processing
The core of the batch system is governed by a Comma Separated Values (CSV) file, `image_dataset.csv`. This file acts as the orchestration map for the script. The structure of the CSV includes the following vital columns:
*   `image_id`: A unique numerical identifier for the transaction.
*   `image_url`: The relative local path or remote URL to the content image (e.g., `dataset/input/cat.jpg`).
*   `description`: A human-readable description of the image, utilized for generating readable output filenames.
*   `style_model`: The specific `.t7` neural network model file to be applied to this particular image (e.g., `the_scream.t7`).

By utilizing this structured metadata, the Python script (`process_csv_dataset.py`) can programmatically iterate through hundreds of images, dynamically switching deep learning models on the fly without user intervention.

### 5.3 Style Model Configurations
The project utilizes six distinct pre-trained style models. These models were selected to provide a diverse range of artistic transformations, ranging from impressionism to abstract cubism. The models are hosted remotely and the application features an intelligent `download_if_missing` function. If a user requests a style (e.g., `mosaic.t7`) that is not currently present in the local `models/` directory, the application automatically establishes an SSL connection, fetches the multi-megabyte weight file from the official Stanford CS repository, and caches it locally for all subsequent inferences.

## 6. Tools and Technologies Used

The successful implementation of this project relies on a robust stack of modern programming languages, libraries, and frameworks.

### 6.1 Python Ecosystem (Python 3.8+)
Python is the primary programming language for this project. Its unparalleled ecosystem for data science and machine learning makes it the absolute industry standard for AI development. Python's syntax allows for rapid prototyping, and its extensive standard library handles file system operations, CSV parsing, and network requests seamlessly.

### 6.2 OpenCV and DNN Module
Open Source Computer Vision Library (OpenCV) is the cornerstone of the project's image processing capabilities. Specifically, the `cv2.dnn` (Deep Neural Network) module is heavily utilized. 
Unlike frameworks such as TensorFlow or PyTorch, which are heavily designed for *training* models, the OpenCV DNN module is strictly designed for *inference* (executing pre-trained models). It is highly optimized and written in C/C++, providing bindings for Python. It automatically fuses network layers and optimizes memory allocation, allowing complex networks like VGG to run exceptionally fast purely on a CPU.

### 6.3 NumPy Matrix Operations
Images in OpenCV are fundamentally represented as NumPy arrays (N-dimensional arrays). NumPy is utilized for high-performance mathematical operations on these massive arrays. The post-processing steps—where the output tensor is transposed (shifting the axes from `[C, H, W]` to `[H, W, C]`) and where specific floating-point mean values are added across entire channels simultaneously—are executed using vectorized NumPy operations, which are orders of magnitude faster than standard Python `for` loops.

### 6.4 Flask Web Framework
Flask is a lightweight, widely-used Web Server Gateway Interface (WSGI) web application framework. It is chosen for this project due to its micro-framework nature; it provides the essential tools for routing HTTP requests and rendering HTML templates without the bloat of larger frameworks like Django. Flask, combined with the Werkzeug utility library for secure file upload handling, forms the bridge between the backend deep learning Python scripts and the frontend user interface.

## 7. Implementation Plan & Algorithm Used

### 7.1 Algorithm Explanation (Fast Neural Style Transfer)
The algorithm driving this project is "Fast Neural Style Transfer." 
In the original optimization-based style transfer, the process required thousands of forward and backward passes through a network to update the pixels of a white noise image.
Fast Neural Style Transfer changes the paradigm. A separate, specialized "Image Transformation Network" (a deep convolutional network with residual blocks and up-sampling layers) is trained beforehand. During the training of this transformation network, a fixed "Loss Network" (typically VGG-16) is used to compute the Perceptual Loss (measuring the difference in content and style). 
By the time the model is saved as a `.t7` file, the transformation network has entirely learned how to apply the chosen style. Therefore, in our application, the algorithm is straightforward: we simply pass the content image forward through this pre-trained transformation network exactly once. The output of the network is the final stylized image.

### 7.2 Mathematical Foundations
While the application does not train the models, understanding the underlying mathematics of the pre-trained weights is crucial. The models were trained to minimize a joint loss function:
*   **Feature Reconstruction Loss (Content Loss):** Ensures the output image matches the high-level features of the content image. It is mathematically defined as the squared Euclidean distance between the feature representations of the content image and the output image at specific layers of the VGG network.
*   **Style Reconstruction Loss (Style Loss):** Ensures the output image matches the textures and colors of the style image. This involves calculating the Gram matrix (the inner product of feature maps) at multiple layers, which captures the correlations between different filter responses, effectively capturing the "texture" independent of spatial arrangement.

### 7.3 Software Development Life Cycle (SDLC)
The project was developed following an agile, phase-wise SDLC model:
1.  **Requirement Analysis & Research:** Understanding the mathematical differences between standard NST and Fast NST. Selecting OpenCV DNN as the inference engine to meet the objective of CPU accessibility.
2.  **Environment Setup & Core Logic (Phase 1):** Writing the foundational `style_transfer.py` script. Implementing the `readNetFromTorch` function, blob creation, and complex post-processing denormalization.
3.  **Batch Processing Pipeline (Phase 2):** Developing the CSV parsing logic. Implementing robust error handling so that if one image fails or is missing, the script logs the error and continues processing the rest of the dataset. Implementing timing mechanisms to evaluate performance.
4.  **Web Integration (Phase 3):** Wrapping the core logic inside a Flask application. Designing the RESTful API endpoint (`/stylize`) that receives multi-part form data, processes the image, and returns the generated URL.
5.  **Testing and Deployment (Phase 4):** Conducting performance tests on images of various sizes. Implementing the 600px resizing constraint to prevent the application from crashing when users upload high-resolution 4K images.

## 8. Output and Expected Results

The application yields highly favorable results, successfully balancing visual fidelity with processing speed.

### 8.1 Batch Processing Evaluation
When the `process_csv_dataset.py` script is executed against the `image_dataset.csv`, the CLI output generates a comprehensive report. The expected results demonstrate that the OpenCV DNN module can execute a forward pass of the complex neural network in approximately 1.5 to 3.5 seconds per image on a standard multi-core Intel/AMD CPU. This is a massive improvement over traditional methods which would take several minutes. The batch processor successfully reads the dynamic style requirements from the CSV and routes the image to the correct `.t7` model, outputting all synthesized images into the `csv_output` directory.

### 8.2 Web Interface Usability
The Flask web interface provides a seamless user experience. Users can interact with a clean, responsive HTML interface to upload their personal photographs. The system handles file validation (ensuring only `.jpg`, `.png`, `.jpeg` files are accepted), secures the filename using Werkzeug, and processes the image. The user receives real-time visual feedback, and the final artwork is rendered directly in their browser.

### 8.3 Output Screenshots
*(Note: Visual screenshots are to be inserted here to demonstrate the application's capabilities.)*

*   **Screenshot 1: The Web Interface.** [Insert screenshot of the Flask web UI showing the upload form and style selection dropdown].
*   **Screenshot 2: Content vs Style.** [Insert a side-by-side comparison of an original input photograph (e.g., a cityscape) and the corresponding style reference image].
*   **Screenshot 3: Synthesized Result.** [Insert the final output image demonstrating the successful blending of the content's structure with the style's aesthetic].
*   **Screenshot 4: CLI Batch Processing.** [Insert a screenshot of the terminal/command prompt showing the execution of the CSV processor, highlighting the processing time metrics and the final evaluation report].

## 9. Conclusion and Future Scope

### 9.1 Summary of Contributions
This project successfully achieved its primary objective: the development of a rapid, accessible, and offline AI-based Neural Style Transfer application. By strategically moving away from iterative optimization algorithms and embracing pre-trained feed-forward Convolutional Neural Networks, the project eliminated the massive computational bottlenecks traditionally associated with AI art generation. 

The implementation of the OpenCV Deep Neural Network (`dnn`) module proved highly effective, allowing complex deep learning models to execute flawlessly on standard consumer CPUs. The dual-interface approach—featuring both a CSV-driven batch processing script and a user-friendly Flask web application—ensures that the system is highly versatile. It serves as a powerful automated tool for digital creators handling bulk datasets, while simultaneously providing an accessible, interactive platform for everyday users exploring AI art. 

### 9.2 Future Enhancements
While the current system is robust, the field of Computer Vision is rapidly evolving, offering several avenues for future enhancement:
1.  **Arbitrary Style Transfer:** The current limitation is that the system relies on specific `.t7` models pre-trained on specific styles. Future iterations could implement Arbitrary Neural Style Transfer models (like AdaIN - Adaptive Instance Normalization), allowing users to upload *any* style image, rather than selecting from a pre-defined list.
2.  **Real-Time Video Processing:** Given the sub-second inference speeds achievable, the core logic could be expanded using `cv2.VideoCapture` to apply style transfer to real-time webcam feeds or MP4 video files on a frame-by-frame basis.
3.  **High-Resolution Up-scaling:** Currently, images are downscaled to 600px for speed. Future enhancements could integrate a secondary AI model (such as ESRGAN - Enhanced Super-Resolution Generative Adversarial Networks) as a post-processing step to up-scale the stylized output back to 4K resolution without losing quality.
4.  **Mobile Deployment:** The Python backend could be adapted into a REST API to serve a cross-platform mobile application (using Flutter or React Native), allowing users to apply neural styles directly from their smartphones.

## 10. References

1. Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). *Image Style Transfer Using Convolutional Neural Networks*. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). This foundational paper introduced the concept of utilizing CNNs to separate and recombine image content and style.
2. Johnson, J., Alahi, A., & Fei-Fei, L. (2016). *Perceptual Losses for Real-Time Style Transfer and Super-Resolution*. European Conference on Computer Vision (ECCV). This research introduced the feed-forward network approach, making Fast Neural Style Transfer possible.
3. Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2016). *Instance Normalization: The Missing Ingredient for Fast Stylization*. arXiv preprint arXiv:1607.08022. Important research detailing the normalization techniques used in the pre-trained models.
4. Official OpenCV Documentation: *Deep Neural Networks (dnn module)* - https://docs.opencv.org/master/d6/d0f/group__dnn.html. Documentation detailing the usage of `cv2.dnn.readNetFromTorch` and blob manipulation.
5. Fast Neural Style Models Repository by Justin Johnson - https://cs.stanford.edu/people/jcjohns/fast-neural-style/. The source repository providing the pre-trained `.t7` Torch models utilized in this project.
6. Flask Web Development framework documentation - https://flask.palletsprojects.com/. Reference for building the web interface and API endpoints.
