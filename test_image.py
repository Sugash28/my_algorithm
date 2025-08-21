import sys
import os
from al import load_model_and_detect

if __name__ == "__main__":
    # Prompt user for model file path and test image path
    model_file = "neuro_sobel_model.npy"  # Default model file
    test_image = "training_set/training_set/dogs/dog.1.jpg"  # Default test image

    if not os.path.exists(model_file):
        print(f"Model file '{model_file}' not found.")
        sys.exit(1)
    if not os.path.exists(test_image):
        print(f"Test image '{test_image}' not found.")
        sys.exit(1)

    load_model_and_detect(model_file, test_image)
    sys.exit(1)

    load_model_and_detect(model_file, test_image)
