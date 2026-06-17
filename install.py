
# Installing datasets
import kagglehub

# Plant Village import
path = kagglehub.dataset_download(
    "mohitsingh1804/plantvillage", output_dir="datasets/plant_village", force_download=True)

# PlantDoc import
path = kagglehub.dataset_download(
    "andresmgs/plantdec", output_dir="datasets/plant_doc", force_download=True)
