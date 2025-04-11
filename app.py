import streamlit as st
import cv2
import numpy as np
from scipy.fftpack import dct, idct
import tempfile
import os
from collections import Counter, defaultdict
import heapq
from moviepy.editor import ImageSequenceClip

st.title("🎓 Educational Video Compression Using Image Compression Techniques")

# RLE Encoding
def rle_encode(img):
    pixels = img.flatten()
    encoding = []
    prev_pixel = pixels[0]
    count = 1
    for pixel in pixels[1:]:
        if pixel == prev_pixel:
            count += 1
        else:
            encoding.append((prev_pixel, count))
            prev_pixel = pixel
            count = 1
    encoding.append((prev_pixel, count))
    return encoding

# Huffman Coding
class Node:
    def __init__(self, symbol=None, freq=None):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq):
    heap = [Node(symbol, freq[symbol]) for symbol in freq]
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        new_node = Node()
        new_node.freq = left.freq + right.freq
        new_node.left = left
        new_node.right = right
        heapq.heappush(heap, new_node)
    return heap[0]

def build_codes(node, prefix="", codebook={}):
    if node is None:
        return
    if node.symbol is not None:
        codebook[node.symbol] = prefix
    build_codes(node.left, prefix + "0", codebook)
    build_codes(node.right, prefix + "1", codebook)
    return codebook

def huffman_encode(img):
    flat = img.flatten()
    freq = Counter(flat)
    tree = build_huffman_tree(freq)
    codebook = build_codes(tree)
    encoded = ''.join(codebook[pixel] for pixel in flat)
    return encoded, codebook

# DCT and IDCT

def apply_dct(img):
    return dct(dct(img.T, norm='ortho').T, norm='ortho')

def apply_idct(img_dct):
    return idct(idct(img_dct.T, norm='ortho').T, norm='ortho')

# Upload video file
uploaded_video = st.file_uploader("Upload Lecture Video", type=["mp4", "avi"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)

    frames = []
    st.write("Extracting frames...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()
    st.success(f"Total Frames Extracted: {len(frames)}")

    method = st.selectbox("Choose Compression Method", ["RLE", "Huffman Coding", "DCT"])

    compressed_data = []
    decompressed_frames = []

    st.write("Applying compression...")
    for i, frame in enumerate(frames):
        if method == "RLE":
            compressed = rle_encode(frame)
            decompressed = np.zeros_like(frame)  # For display only; decoding not implemented here
        elif method == "Huffman Coding":
            compressed, _ = huffman_encode(frame)
            decompressed = np.zeros_like(frame)  # For display only; decoding not implemented here
        elif method == "DCT":
            dct_frame = apply_dct(frame)
            decompressed = apply_idct(dct_frame)
            decompressed = np.clip(decompressed, 0, 255)
        compressed_data.append(compressed)
        decompressed_frames.append(decompressed)

    st.success(f"Compression complete using {method}.")
    st.image(np.uint8(decompressed_frames[0]), caption="Decompressed Frame Example")

    if st.button("Export Compressed Video"):
        temp_video_path = os.path.join(tempfile.gettempdir(), "compressed_output.mp4")
        clip = ImageSequenceClip([np.uint8(f) for f in decompressed_frames], fps=20)
        clip.write_videofile(temp_video_path, codec='libx264')
        with open(temp_video_path, "rb") as file:
            btn = st.download_button(
                label="Download Compressed Video",
                data=file,
                file_name="compressed_output.mp4",
                mime="video/mp4"
            )

    os.unlink(tfile.name)

