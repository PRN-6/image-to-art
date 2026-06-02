# Image to Art (Neural Style Transfer) with Dataset Processing

This is a Machine Learning project that transforms regular images into stylized pieces of art using Neural Style Transfer. It uses a pre-trained deep learning model via OpenCV to apply the style of a famous painting (Van Gogh's Starry Night) to a dataset of content images.

## Features
- **Neural Style Transfer**: Uses a lightweight pre-trained `.t7` model for quick inference without needing a GPU.
- **Dataset Batch Processing**: Reads image metadata and paths from a CSV file and processes the entire dataset sequentially.
- **Performance Evaluation**: Calculates processing time per image and overall throughput.

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

### 1. CSV Dataset Processing (Batch Mode)
To run the ML project pipeline on your CSV dataset:
```bash
python csv_data_processing/process_csv_dataset.py
```
This will:
1. Read `csv_data_processing/image_dataset.csv` for image URLs or local paths (from the `dataset/input` directory).
2. Download any required images from URLs or load the local images.
3. Run the Neural Style Transfer model on every image specified in the CSV.
4. Save all results in the `dataset/csv_output` folder.
5. Display an evaluation report showing total processing time and average time per image.

### 2. Single Image Transfer
To stylize just one image:
```bash
python style_transfer.py
```
Or specify an image path:
```bash
python style_transfer.py "path/to/image.jpg"
```
