import os
import cv2
import numpy as np
import easyocr
import urllib.parse
import requests
from flask import Flask, render_template, request, redirect

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure the uploads folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def search_google_books(query):
    """Searches the Google Books API and returns title, authors, description, and image URL."""
    encoded_query = urllib.parse.quote(query)
    url = f"https://www.googleapis.com/books/v1/volumes?q={encoded_query}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if 'items' in data:
            book = data['items'][0]['volumeInfo']
            title = book.get('title', 'No Title Found')
            authors = ', '.join(book.get('authors', ['No Author Found']))
            description = book.get('description', 'No Description Available')
            image_url = book.get('imageLinks', {}).get('thumbnail', 'No Image Available')
            return title, authors, description, image_url
    return None, None, "No books found.", None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Step 1: Load and resize the image
    image = cv2.imread(filepath)
    image = cv2.resize(image, (600, 800))  # Resize for easier processing

    # Step 2: Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 3: Compute the sum of pixel intensities for each column in RGB channels
    sum_r = np.sum(image[:, :, 0], axis=0)  # Red channel
    sum_g = np.sum(image[:, :, 1], axis=0)  # Green channel
    sum_b = np.sum(image[:, :, 2], axis=0)  # Blue channel

    # Combine the RGB sums to detect color transitions
    combined_sum = (sum_r + sum_g + sum_b) / 3

    # Step 5: Plot the grayscale intensity sum graph
    gray_sum = np.sum(gray, axis=0)

    # Step 6: Detect peaks (book boundaries) using a combination of RGB and grayscale information
    from scipy.signal import find_peaks
    peaks_gray, _ = find_peaks(-gray_sum, distance=30, prominence=500)
    peaks_color, _ = find_peaks(-combined_sum, distance=30, prominence=500)

    # Combine both sets of peaks for more accurate spine detection
    all_peaks = np.unique(np.concatenate((peaks_gray, peaks_color)))

    # Sort the detected peaks and add start and end points to include the first and last books
    all_peaks_sorted = np.sort(all_peaks)
    width = image.shape[1]  # Get the width of the image
    all_peaks_sorted = np.concatenate(([0], all_peaks_sorted, [width]))

    # Segment and save each book spine as an individual image
    book_images = []
    for i in range(len(all_peaks_sorted) - 1):
        left = all_peaks_sorted[i]
        right = all_peaks_sorted[i + 1]
        book_spine = image[:, left:right]
        book_images.append(book_spine)

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Perform OCR on each rotated book spine and search Google Books API
    results = []
    for book_spine in book_images:
        # Rotate each segmented book spine 90 degrees anticlockwise
        rotated_spine = cv2.rotate(book_spine, cv2.ROTATE_90_COUNTERCLOCKWISE)

        # Perform OCR on the rotated spine using EasyOCR
        rgb_spine = cv2.cvtColor(rotated_spine, cv2.COLOR_BGR2RGB)  # Convert the image to RGB
        ocr_results = reader.readtext(rgb_spine)

        # Extracting and concatenating text
        ocr_text = ' '.join([result[1] for result in ocr_results]).strip()  # Clean text

        # Search Google Books API using the OCR result
        if ocr_text:
            title, authors, description, image_url = search_google_books(ocr_text)
            results.append({
                'title': title,
                'authors': authors,
                'description': description,
                'image_url': image_url
            })

    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
