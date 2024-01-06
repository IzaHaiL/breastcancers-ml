from tensorflow.keras.utils import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import ResNet50, VGG19
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense

from imutils import paths
import numpy as np
import pickle
import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default="data/datasets/", help='Root directory of data')
args = parser.parse_args()

def process_model(model, output_directory, model_name):
    print(f"[INFO] Loading {model_name} network...")
    le = None

    # loop over the data splits
    for split in ("breast_cancer", "DUMMY"):
        # grab all image paths in the current split
        print(f"[INFO] Processing '{split}' split for {model_name}...")
        p = os.path.sep.join([args.root, split])
        imagePaths = list(paths.list_images(p))

        # randomly shuffle the image paths and then extract the class
        # labels from the file paths
        # random.shuffle(imagePaths)
        labels = [p.split(os.path.sep)[-2] for p in imagePaths]

        # if the label encoder is None, create it
        if le is None:
            le = LabelEncoder()
            le.fit(labels)

        # open the output CSV file for writing
        csvPath = os.path.sep.join([output_directory, f"{model_name}_{split}.csv"])
        csv = open(csvPath, "w")

        # loop over the images in batches
        for (b, i) in enumerate(tqdm(range(0, len(imagePaths), 32), desc=f"Processing {model_name} {split}", unit="batch")):
            # extract the batch of images and labels, then initialize the
            # list of actual images that will be passed through the network
            # for feature extraction
            batchPaths = imagePaths[i:i + 32]
            batchLabels = le.transform(labels[i:i + 32])
            batchImages = []

            # loop over the images and labels in the current batch
            for imagePath in batchPaths:
                # load the input image using the Keras helper utility
                # while ensuring the image is resized to 224x224 pixels
                image = load_img(imagePath, target_size=(224, 224))
                image = img_to_array(image)

                # preprocess the image by (1) expanding the dimensions and
                # (2) subtracting the mean RGB pixel intensity from the
                # ImageNet dataset
                image = np.expand_dims(image, axis=0)
                image = imagenet_utils.preprocess_input(image)

                # add the image to the batch
                batchImages.append(image)

            # pass the images through the network and use the outputs as
            # our actual features, then reshape the features into a
            # flattened volume
            batchImages = np.vstack(batchImages)
            features = model.predict(batchImages, batch_size=16)
            features = features.reshape((features.shape[0], -1))

            # loop over the class labels and extracted features
            for (label, vec) in zip(batchLabels, features):
                # construct a row that exists of the class label and
                # extracted features
                vec = ",".join([str(v) for v in vec])
                csv.write("{},{}\n".format(vec, label))

        # close the CSV file
        csv.close()

    # serialize the label encoder to disk
    f = open(os.path.join(output_directory, f"{model_name}_le.cpickle"), "wb")
    f.write(pickle.dumps(le))
    f.close()

# Process ResNet50
process_model(ResNet50(weights="imagenet", include_top=False), "data/resnet50_output", "resnet50")

# Process VGG19
process_model(VGG19(weights="imagenet", include_top=False), "data/vgg19_output", "vgg19")
