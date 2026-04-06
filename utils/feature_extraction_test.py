import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import numpy as np

from utils.data_loader import get_data_loader


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")


class FeatureExtractionTest:

    def __init__(self, train_loader, test_loader, batch_size):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.batch_size = batch_size

        print(f"Train batches: {len(self.train_loader)}")
        print(f"Test batches: {len(self.test_loader)}")

       
        # Pretrained ResNet152 as feature extractor
        # Output: 2048-dim feature vector
       
        backbone = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(backbone.children())[:-1])
        self.model.to(device)
        self.model.eval()


    # Feature extraction method 1: Flatten pixels

    def flatten_images(self):
        x_train, y_train = [], []
        x_test, y_test = [], []

        with torch.no_grad():
            for images, labels in self.train_loader:
                images = images.numpy()
                labels = labels.numpy()

                for i in range(images.shape[0]):
                    x_train.append(images[i].flatten())
                    y_train.append(labels[i])

            for images, labels in self.test_loader:
                images = images.numpy()
                labels = labels.numpy()

                for i in range(images.shape[0]):
                    x_test.append(images[i].flatten())
                    y_test.append(labels[i])

        return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

    # Feature extraction method 2: ResNet152 (Transfer Learning)

    def resnet_feature_extraction(self):
        x_train, y_train = [], []
        x_test, y_test = [], []

        with torch.no_grad():
            for images, labels in self.train_loader:
                images = images.to(device)

                outputs = self.model(images)
                features = outputs.squeeze().cpu().numpy()
                labels = labels.numpy()

                x_train.extend(features)
                y_train.extend(labels)

            for images, labels in self.test_loader:
                images = images.to(device)

                outputs = self.model(images)
                features = outputs.squeeze().cpu().numpy()
                labels = labels.numpy()

                x_test.extend(features)
                y_test.extend(labels)

        return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

  
    # Feature extraction method 3: GAN discriminator
    
    def GAN_feature_extraction(self, discriminator):
        x_train, y_train = [], []
        x_test, y_test = [], []

        discriminator.to(device)
        discriminator.eval()

        with torch.no_grad():
            for images, labels in self.train_loader:
                images = images.to(device)

                outputs = discriminator.feature_extraction(images)
                features = outputs.view(outputs.size(0), -1).cpu().numpy()
                labels = labels.numpy()

                x_train.extend(features)
                y_train.extend(labels)

            for images, labels in self.test_loader:
                images = images.to(device)

                outputs = discriminator.feature_extraction(images)
                features = outputs.view(outputs.size(0), -1).cpu().numpy()
                labels = labels.numpy()

                x_test.extend(features)
                y_test.extend(labels)

        return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

    
    # Linear evaluation (Logistic Regression)
  
    def calculate_score(self, method="resnet", discriminator=None, runs=10):
        mean_score = 0.0

        for i in range(runs):
            print(f"\nRun {i + 1}/{runs}")

            train_loader, test_loader = get_data_loader(args)
            self.train_loader = train_loader
            self.test_loader = test_loader

            if method == "resnet":
                x_train, y_train, x_test, y_test = self.resnet_feature_extraction()
            elif method == "gan":
                x_train, y_train, x_test, y_test = self.GAN_feature_extraction(discriminator)
            elif method == "flatten":
                x_train, y_train, x_test, y_test = self.flatten_images()
            else:
                raise ValueError("Unknown feature extraction method")

            clf = LogisticRegression(max_iter=1000, n_jobs=-1)
            clf.fit(x_train, y_train)

            preds = clf.predict(x_test)
            score = accuracy_score(y_test, preds)
            print(f"Accuracy: {score:.4f}")

            mean_score += score

        mean_score /= runs
        print(f"\nMean Accuracy over {runs} runs: {mean_score:.4f}")
        return mean_score