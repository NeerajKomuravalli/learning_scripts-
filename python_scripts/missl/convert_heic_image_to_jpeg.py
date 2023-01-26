from PIL import Image
from pillow_heif import register_heif_opener
import sys
import os

register_heif_opener()

source_heic_image_path = sys.argv[1]
if not os.path.exists(source_heic_image_path):
	raise Exception("Source image path does not exist")

taregt_jpeg_file_path = sys.argv[2]
taregt_jpeg_folder_path = '/'.join(taregt_jpeg_file_path.split('/')[:-1])
if not os.path.exists(taregt_jpeg_folder_path):
	raise Exception("target image folder path does not exist")

image = Image.open(source_heic_image_path)

image.save(taregt_jpeg_file_path, format='jpeg')