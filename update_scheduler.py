import os
import tarfile

import gdown



def download_model_from_google_drive():
    model_file_path = './models/trained_model.pkl'
    if not os.path.exists(model_file_path):
        print('Model not found. Downloading model...')
        file_id = "1OuWtu2ojLQLdxnzMKm452rfaVH3crEWA"
        destination = "./models/trained_model.pkl"
        destination_dir = os.path.dirname(destination)
        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, destination, quiet=False)

def download_and_unpack_cache():
    # Ensure the cache directory exists
    if not os.path.exists('./cache'):
        os.makedirs('./cache')

    # Google Drive file id
    file_id = "1LzpeMkfnjxXZrKy7FxSiFVCq6mAoKKoh"
    # Destination path
    destination = "./cache/archive.tar.gz"
    # Google Drive download link
    url = f"https://drive.google.com/uc?id={file_id}"
    # Download the file
    gdown.download(url, destination, quiet=False)

    # Unpack the .tar.gz file
    with tarfile.open(destination, 'r:gz') as tar:
        tar.extractall(path='./cache')


def download_weather_from_google_drive():
    if not os.path.exists('./cache'):
        os.makedirs('./cache')
    file_id = "1zNO2r4daZrEZB3pYtNHS14FYqal_GtMc"
    destination = "./cache/"
    destination_dir = os.path.dirname(destination)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)

def download_state_from_google_drive():
    if not os.path.exists('./cache'):
        os.makedirs('./cache')
    file_id = "1vPj4S__ZK-cv17kfFvZ4oU_rVgAWMLpL"
    destination = "./cache/"
    destination_dir = os.path.dirname(destination)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)


