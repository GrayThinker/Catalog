import streamlit as st
import cv2
from PIL import Image
from segmenter import extract_foreground

def run():
    st.title("Out painting")

    image_file = st.file_uploader("Image", ["jpg", "png"])

    if image_file:
        # image = Image.open(image)
        image = cv2.imread(image_file)
        


def resize(image, max_size=640):
    w, h = image.size
    print(f"loaded input image of size ({w}, {h})")
    if max(w, h) > max_size:
        factor = max_size / max(w, h)
        w = int(factor*w)
        h = int(factor*h)
    # resize to integer multiple of 64
    width, height = map(lambda x: x - x % 64, (w, h))
    image = image.resize((width, height))
    print(f"resized to ({width}, {height})")
    return image


if __name__ == "__main__":
    run()