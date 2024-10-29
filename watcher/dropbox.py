import dropbox
import os

ACCESS_TOKEN = 'your_access_token'
LOCAL_INPUT_DIR = '/data/input'

dbx = dropbox.Dropbox(ACCESS_TOKEN)

def download_new_files():
    for entry in dbx.files_list_folder('/sync_folder').entries:
        if isinstance(entry, dropbox.files.FileMetadata):
            local_path = os.path.join(LOCAL_INPUT_DIR, entry.name)
            with open(local_path, "wb") as f:
                metadata, res = dbx.files_download(path=entry.path_lower)
                f.write(res.content)
                print(f"Downloaded {entry.name}")

download_new_files()
