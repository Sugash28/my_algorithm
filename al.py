import numpy as np
from scipy.ndimage import rotate, laplace, gaussian_laplace
import matplotlib.pyplot as plt
from imageio import imread
import os

class NeuroSobel:
    def __init__(self, learning_rate=0.01, threshold=0.01, min_accuracy=50, max_loops=5, angle_step=15):
        self.learning_rate = learning_rate
        self.threshold = threshold
        self.min_accuracy = min_accuracy
        self.max_loops = max_loops
        self.angle_step = angle_step
        self.learned_weights = None
        self.direction_names = ["N", "NE", "E", "SE", "S", "SW", "W", "NW", "C"]
        self.class_edge_means = {}  

    def train(self, image_path):
        image = self._read_grayscale(image_path)
        h, w = image.shape
        self.learned_weights = np.array([1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 2.0])
        self.learned_weights /= np.sum(self.learned_weights)

        angles = list(range(0, 360, self.angle_step))
        for angle in angles:
            print(f"\n Training at Rotation: {angle}°")
            rotated_image = rotate(image, angle, reshape=False, order=3, mode='reflect')
            direction_tensor = self._build_direction_tensor(rotated_image)
            pseudo_gt = self._laplacian_edge(rotated_image)           

            for loop in range(1, self.max_loops + 1):
                weighted_avg = np.tensordot(direction_tensor, self.learned_weights, axes=([2], [0])) / np.sum(self.learned_weights)
                v = self._sobel_vertical(weighted_avg)
                h_ = self._sobel_horizontal(weighted_avg)
                mag = np.sqrt(v**2 + h_**2)

                h_m, w_m = mag.shape
                direction_tensor_cropped = direction_tensor[:h_m, :w_m, :]
                pseudo_gt_crop = pseudo_gt[:h_m, :w_m]

                loss, dL_dw = self._compute_loss_and_grad(direction_tensor_cropped, mag, pseudo_gt_crop)

                self.learned_weights -= self.learning_rate * dL_dw
                self.learned_weights = np.clip(self.learned_weights, 0, None)
                self.learned_weights /= np.sum(self.learned_weights)

                acc = self._evaluate_accuracy(mag, self.threshold)
                print(f"    Attempt {loop}: Loss = {loss:.4f}, Accuracy = {acc:.2f}%")

                if acc >= self.min_accuracy:
                    print(f"    Success at {angle}°")
                    break

        print("\n Final Learned Weights:", np.round(self.learned_weights, 3))
        return self.learned_weights

    def detect(self, image_path):
        if self.learned_weights is None:
            raise ValueError("Model not trained yet.")

        image = self._read_grayscale(image_path)
        direction_tensor = self._build_direction_tensor(image)
        weighted_avg = np.tensordot(direction_tensor, self.learned_weights, axes=([2], [0])) / np.sum(self.learned_weights)

        v = self._sobel_vertical(weighted_avg)
        h = self._sobel_horizontal(weighted_avg)
        mag = np.sqrt(v**2 + h**2)

        self._show_image(weighted_avg, title="Weighted Average")
        self._show_image(mag, title="Edge Magnitude (NeuroSobel)")

        return mag

    def save_weights(self, filepath):
       
        np.save(filepath, self.learned_weights, allow_pickle=False)
        print(f"Weights saved to {filepath}")

    def load_weights(self, filepath):
      
        self.learned_weights = np.load(filepath, allow_pickle=False)
        print(f"Weights loaded from {filepath}")

    def _read_grayscale(self, path):
        img = imread(path)
        if len(img.shape) == 3:
            return np.mean(img, axis=2).astype(np.uint8)
        return img

    def _build_direction_tensor(self, image):
        padded = np.pad(image, pad_width=1, mode='reflect')
        N = padded[:-2, 1:-1]
        NE = padded[:-2, 2:]
        E = padded[1:-1, 2:]
        SE = padded[2:, 2:]
        S = padded[2:, 1:-1]
        SW = padded[2:, :-2]                                                            
        W = padded[1:-1, :-2]
        NW = padded[:-2, :-2]
        C = padded[1:-1, 1:-1]
        return np.stack([N, NE, E, SE, S, SW, W, NW, C], axis=2)

    def _sobel_vertical(self, image):
        kernel = np.array([
            [-2, -1, 0, 1, 2],
            [-2, -1, 0, 1, 2],
            [-4, -2, 0, 2, 4],
            [-2, -1, 0, 1, 2],
            [-2, -1, 0, 1, 2]
        ])
        return self._manual_convolve(image, kernel)

    def _sobel_horizontal(self, image):
        kernel = np.array([
            [-2, -2, -4, -2, -2],
            [-1, -1, -2, -1, -1],
            [ 0,  0,  0,  0,  0],
            [ 1,  1,  2,  1,  1],
            [ 2,  2,  4,  2,  2]
        ])
        return self._manual_convolve(image, kernel)

    def _manual_convolve(self, image, kernel):
        kh, kw = kernel.shape
        ih, iw = image.shape
        result = np.zeros((ih - kh + 1, iw - kw + 1))
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                patch = image[i:i+kh, j:j+kw]
                result[i, j] = np.sum(patch * kernel)
        return result

    def _laplacian_edge(self, image):
        lap = np.abs( gaussian_laplace(image,sigma=1.0))
        return lap / (np.max(lap) + 1e-6)

    def _compute_loss_and_grad(self, direction_tensor, mag, pseudo_gt):
        mag_norm = mag / (np.max(mag) + 1e-6)
        gt_norm = pseudo_gt / (np.max(pseudo_gt) + 1e-6)
        loss = np.mean((mag_norm - gt_norm)**2)
        dL_dmag = 2 * (mag_norm - gt_norm) / mag.size
        dmag_davg = mag / (np.max(mag) + 1e-6)
        dL_davg = dL_dmag * dmag_davg
        dL_dw = np.sum(direction_tensor * np.expand_dims(dL_davg, axis=2), axis=(0, 1))
        return loss, dL_dw

    def _evaluate_accuracy(self, mag, threshold=30):
        return np.sum(mag > threshold) / mag.size * 100

    def _show_image(self, image, title="Image"):
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()


def load_dataset_and_train(folder_path, model_path="neuro_sobel_model.npy"):
    model = NeuroSobel(learning_rate=0.05, threshold=0.01, min_accuracy=70, max_loops=50, angle_step=15)
    class_edge_means = {}

    for label in os.listdir(folder_path):
        label_path = os.path.join(folder_path, label)
        if os.path.isdir(label_path):
            print(f"/n[INFO] Training on label: {label}")
            edge_means = []
            for img_file in os.listdir(label_path):
                if img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(label_path, img_file)
                    model.train(img_path)
                   
                    mag = model.detect(img_path)
                    edge_means.append(np.mean(mag))
                    
            if edge_means:
                class_edge_means[label] = np.mean(edge_means)
    model.class_edge_means = class_edge_means
    model.save_weights(model_path)
    # Save class_edge_means as a .npz file with a separate extension
    class_means_path = os.path.splitext(model_path)[0] + ".npz"
    np.savez(class_means_path, **class_edge_means)
    print("Class edge means saved:", class_edge_means)

def load_model_and_detect(model_path, image_path):
    import sys
    model = NeuroSobel()
    model.load_weights(model_path)
    class_means_path = os.path.splitext(model_path)[0] + ".npz"
    if not os.path.exists(class_means_path):
        print(f"Error: Class means file '{class_means_path}' not found. Please run training first to generate it.")
        sys.exit(1)
    # Load class_edge_means from .npz file
    class_edge_means_npz = np.load(class_means_path)
    class_edge_means = {k: float(class_edge_means_npz[k]) for k in class_edge_means_npz.files}
    model.class_edge_means = class_edge_means
    mag = model.detect(image_path)
    test_mean = np.mean(mag)
    pred_label = min(class_edge_means, key=lambda k: abs(class_edge_means[k] - test_mean))
    print(f"The tested image is predicted to be a: {pred_label}")
    return pred_label

if __name__ == "__main__":
    dataset_folder = "train"  
    model_file = "neuro_sobel_model.npy"

    class_means_path = os.path.splitext(model_file)[0] + ".npz"
    if not os.path.exists(class_means_path):
        print("Class means file not found. Running training first...")
        load_dataset_and_train(dataset_folder, model_file)

    test_image = "training_set/training_set/cats/cat.1.jpg"  
    load_model_and_detect(model_file, test_image)

