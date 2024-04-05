import os

import gdown



def download_file_from_google_drive():
    file_id = "1OuWtu2ojLQLdxnzMKm452rfaVH3crEWA"
    destination = "./models/trained_model.pkl"
    destination_dir = os.path.dirname(destination)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)



