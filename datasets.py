import kagglehub

# Download latest version
path = kagglehub.dataset_download("aneesarom/rider-with-helmet-without-helmet-number-plate")

print("Path to dataset files:", path)