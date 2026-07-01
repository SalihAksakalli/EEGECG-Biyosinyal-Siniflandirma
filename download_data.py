import os
import shutil
import kagglehub

def main():
    print("Starting dataset download via kagglehub...")
    # Download dataset files
    download_path = kagglehub.dataset_download("shayanfazeli/heartbeat")
    print(f"Dataset downloaded to cache at: {download_path}")

    # Local data directory in the project
    local_data_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(local_data_dir, exist_ok=True)
    print(f"Ensuring local data directory exists: {local_data_dir}")

    # Copy files
    copied_files = []
    for item in os.listdir(download_path):
        src_path = os.path.join(download_path, item)
        if os.path.isfile(src_path) and item.endswith(".csv"):
            dest_path = os.path.join(local_data_dir, item)
            print(f"Copying {item} to {dest_path}...")
            shutil.copy2(src_path, dest_path)
            copied_files.append(item)

    print("Copy process completed.")
    print("Files in local data directory:", os.listdir(local_data_dir))

if __name__ == "__main__":
    main()
