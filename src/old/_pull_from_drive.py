"""
Created on 09/15/2024

@author: Dan Schumacher
How to use:
python ./src/_pull_from_drive.py --replace
"""
import os
import argparse
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2.credentials import Credentials
import io

# Initialize Google Drive API
def initialize_drive():
    creds = Credentials.from_authorized_user_file('./resources/drive_token.json', ['https://www.googleapis.com/auth/drive'])
    service = build('drive', 'v3', credentials=creds)
    return service

# Download file from Google Drive
def download_file(service, file_id, file_path, replace=False):
    if os.path.exists(file_path) and not replace:
        print(f"Skipped {file_path}, file already exists")
        return

    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(file_path, 'wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        print(f"Downloading {file_path}: {int(status.progress() * 100)}%")
    print(f"Downloaded {file_path}")

# Recursively download folder structure from Google Drive
def download_folder(service, folder_id, local_folder_path, replace=False):
    # Ensure the local folder exists
    if not os.path.exists(local_folder_path):
        os.makedirs(local_folder_path)

    # List all files and folders in the current folder
    query = f"'{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name, mimeType)').execute()
    items = results.get('files', [])

    for item in items:
        if item['mimeType'] == 'application/vnd.google-apps.folder':
            # If it's a folder, recursively download its contents
            subfolder_path = os.path.join(local_folder_path, item['name'])
            download_folder(service, item['id'], subfolder_path, replace)
        else:
            # If it's a file, download it
            file_path = os.path.join(local_folder_path, item['name'])
            download_file(service, item['id'], file_path, replace)

# Get folder ID by name
def get_folder_id_by_name(service, parent_folder_id, folder_name):
    query = f"'{parent_folder_id}' in parents and name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    items = results.get('files', [])
    if items:
        return items[0]['id']
    return None

# Main script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Download models folder from Google Drive")
    parser.add_argument('--replace', action='store_true', help="Replace existing files")
    args = parser.parse_args()

    local_folder_path = './models'  # Local folder where the files will be downloaded
    drive_folder_id = '1BzRf1A-ierIDJMlvCOLLd1eGXYYZ57kv'  # Google Drive folder ID from which you want to download

    service = initialize_drive()
    download_folder(service, drive_folder_id, local_folder_path, replace=args.replace)
