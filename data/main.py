import os
import shutil
import synapseclient
from dotenv import load_dotenv

load_dotenv();
access_token = os.getenv("ACCESS_TOKEN")

syn = synapseclient.Synapse()
syn.login(authToken=access_token)

project_dir = r"D:\7thSemProject\mPower_Audio"
os.makedirs(project_dir, exist_ok=True)

print("Querying the dataset... (This might take a minute)")
# Query without the LIMIT to get all 65,022 rows
results = syn.tableQuery("SELECT * FROM syn5511444 LIMIT 24100")

print("Starting download... (This will take a while depending on your internet speed)")
# Download the actual audio files
downloaded_files = syn.downloadTableColumns(results, ["audio_audio.m4a"])

print("Copying downloaded files to your project folder...")
for file_path in downloaded_files.values():
    if file_path and os.path.exists(file_path):
        shutil.copy(file_path, project_dir)

print("Download and copy complete!")