import kagglehub

# Download latest version
path = kagglehub.dataset_download("tolgahayit/yellowrust19-yellow-rust-disease-in-wheat")

print("Path to dataset files:", path)