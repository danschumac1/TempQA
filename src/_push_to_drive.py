"""
Created on 09/27/2024

@author: Dan
python ./src/_push_to_drive.py --replace
"""
import argparse
import os
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials

# Initialize Google Drive API
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request

def initialize_drive():
    creds = None
    token_path = './resources/drive_token.json'
    creds_path = './resources/drive_credentials.json'
    
    # Check if token already exists
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, ['https://www.googleapis.com/auth/drive'])
    
    # If there are no valid credentials available, prompt the user to log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(creds_path, ['https://www.googleapis.com/auth/drive'])
            creds = flow.run_local_server(port=0)
        
        # Save the credentials for the next run
        with open(token_path, 'w') as token:
            token.write(creds.to_json())
    
    service = build('drive', 'v3', credentials=creds)
    return service

# Check if a file exists in Google Drive
def check_file_exists(service, file_name, folder_id):
    query = f"'{folder_id}' in parents and name='{file_name}' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    files = results.get('files', [])
    if len(files) > 0:
        return files[0]['id']  # Return file ID if found
    return None

# Create a folder on Google Drive
def create_drive_folder(service, parent_folder_id, folder_name):
    query = f"'{parent_folder_id}' in parents and name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()

    if len(results['files']) > 0:
        # Folder already exists
        return results['files'][0]['id']
    else:
        # Create new folder
        file_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_folder_id]
        }
        folder = service.files().create(body=file_metadata, fields='id').execute()
        return folder['id']

# Upload file to Google Drive
def upload_file(service, file_path, folder_id, replace=False):
    file_name = os.path.basename(file_path)
    
    # Check if file exists
    existing_file_id = check_file_exists(service, file_name, folder_id)
    
    if existing_file_id and replace:
        print(f"Replacing existing file: {file_name}")
        service.files().delete(fileId=existing_file_id).execute()
    
    if not existing_file_id or replace:
        file_metadata = {'name': file_name, 'parents': [folder_id]}
        media = MediaFileUpload(file_path, resumable=True)
        service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print(f"Uploaded {file_name}")
    else:
        print(f"Skipped {file_name}")

# Upload entire folder structure
def upload_folder(service, folder_path, parent_drive_folder_id, replace=False):
    for root, dirs, files in os.walk(folder_path):
        # Create folder structure on Google Drive
        relative_path = os.path.relpath(root, folder_path)
        current_drive_folder_id = parent_drive_folder_id

        if relative_path != ".":
            # Create subfolder structure on Drive
            for folder in relative_path.split(os.sep):
                current_drive_folder_id = create_drive_folder(service, current_drive_folder_id, folder)

        for file_name in files:
            file_path = os.path.join(root, file_name)
            upload_file(service, file_path, current_drive_folder_id, replace)

# Main script
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Upload models folder to Google Drive")
    parser.add_argument('--replace', action='store_true', help="Replace existing files with the same name")
    args = parser.parse_args()

    folder_paths = ['./models/gemma/MenatQA', './data']  # Local models folder
    drive_folder_id = '1BzRf1A-ierIDJMlvCOLLd1eGXYYZ57kv'  # Google Drive folder ID where you want to upload

    service = initialize_drive()
    for folder_path in folder_paths:
        print(f"Uploading {folder_path} to Google Drive")
        upload_folder(service, folder_path, drive_folder_id, replace=args.replace)
    print("All uploads complete")