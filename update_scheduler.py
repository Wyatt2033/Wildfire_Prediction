import gdown



def download_file_from_google_drive():
    file_id = "1OuWtu2ojLQLdxnzMKm452rfaVH3crEWA"
    destination = "./models"

    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)



