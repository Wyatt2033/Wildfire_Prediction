import gdown

file_id = "https://drive.google.com/file/d/1OuWtu2ojLQLdxnzMKm452rfaVH3crEWA/view?usp=drive_link"
destination = ""


def download_file_from_google_drive(file_id, destination):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, destination, quiet=False)


download_file_from_google_drive(file_id, destination)
