import dotenv
import os
dotenv.load_dotenv()

DATASET_DIR = os.getenv("DATASET")
METADATA_DIR = os.getenv("METADATA_DIR")
LABEL_CSV = os.getenv("LABELS")
