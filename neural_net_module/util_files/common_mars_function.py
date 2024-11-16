
from util_files.common_imports import *

vgg16_model = VGG16(weights='imagenet', include_top=False)

## load image

def load_images_from_directory(dir_path):
    """
    Function to get all the images within a certain directory (folder).

    params:
    :dir_path str: Path to the directory containing the image file.

    return:
    None: The function returns a list of the images and the filename associated with the image
    """
    images = []
    filenames = []
    for filename in os.listdir(dir_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.JPG')): 
            img = skimage.io.imread(os.path.join(dir_path, filename))
            if img.ndim == 3:
                img = rgb2gray(img)
            img = img/255.0 # normalize pixels
            images.append(img)
            filenames.append(filename)
    return images, filenames


def split_image_into_tiles(dir_path, image, cropsize, output_folder):
    """
    Function to split an image into square tiles, optionally applying padding using reflection from the original image boundaries.

    params:
    :dir_path str: Path to the directory containing the image file.
    :image str: Name of the image file to be split.
    :cropsize int: The size (in pixels) of each square tile.
    :output_folder str: Directory to save the resulting image tiles.

    return:
    None: The function saves cropped and padded image tiles to the specified output folder.
    """

    # Create the directory
    os.makedirs(output_folder, exist_ok=True)
    image_path = f"{dir_path}/{image}"

    # Open the image file
    img = skimage.io.imread(image_path)

    # Convert to grayscale 
    if img.ndim == 3:  # Check color channel
        img = rgb2gray(img)

    height, width = img.shape
    crop_size = cropsize
    padding_size = crop_size // 2  # ensure enough padding

    for i in range(0, height, crop_size):
        for j in range(0, width, crop_size):
            # Determine the crop boundaries, including padding from the full image
            start_row = max(i - padding_size, 0)
            end_row = min(i + crop_size + padding_size, height)
            start_col = max(j - padding_size, 0)
            end_col = min(j + crop_size + padding_size, width)

            # Crop the region from the full image including padding
            crop_with_context = img[start_row:end_row, start_col:end_col]

            # Extract the center crop from the padded region
            crop = crop_with_context[
                start_row - i + padding_size: start_row - i + padding_size + crop_size,
                start_col - j + padding_size: start_col - j + padding_size + crop_size
            ]

            # Convert the image patch to 8-bit format
            crop_8bit = img_as_ubyte(crop)

            # Save the image patch as a JPEG file
            skimage.io.imsave(f"{output_folder}/tile_{i}_{j}_{image}", crop_8bit)

def resize_image(image, target_size = (64, 64)):
    """
    Function to resize image to target size
    params:
    :images ndarray: An image represented as NumPy array.
    :target_size tuple of int, optional: The dimensions to which all images will be resized. Default is (64, 64).

    return:
    :ndarray: The images transformed into the PCA space, represented as a NumPy array with 'n_components' principal components.
    """
    resized_image = skimage.transform.resize(image, target_size, anti_aliasing=True)
    return resized_image

def image_flattening(images, target_size = (64, 64)):
    """
    Function to resize and flatten a collection of images into 1D vectors.

    params:
    :images list of ndarray: A list of images represented as NumPy arrays.
    :target_size tuple of int, optional: The size to which each image will be resized. Default is (64, 64).

    return:
    :ndarray: A 2D array where each row represents a flattened image with dimensions (n_samples, width * height).
    """
    images_resized = [resize_image(img, target_size) for img in images]
    flattened_images = np.array([img.flatten() for img in images_resized])

    return flattened_images

def extract_features_fourier(images):
    """
    Function to extract magntitude from fourier transfrom from a list of images.

    params:
    :images list of ndarray: A list of images represented as NumPy arrays, where each image is expected to have a consistent shape.

    return:
    :ndarray: A 2D array where each row represents the extracted features of an image, suitable for further processing or analysis.
    """
    features = []
    for img in images:
        f_transform = np.fft.fft2(img)
        f_transform_shifted = np.fft.fftshift(f_transform)  # Shift zero frequency components to center
        magnitude = np.abs(f_transform_shifted)
        features.append(magnitude.flatten())  # Flatten for clustering

    return np.array(features, dtype=object)

def apply_tsne(images, n_components=2, perplexity=30):
    """
    Function to apply t-Distributed Stochastic Neighbor Embedding (t-SNE) for dimensionality reduction on a set of images.

    params:
    :images ndarray: A 2D array of shape (n_samples, n_features), where each row represents a flattened image.
    :n_components int, optional: The dimension of the embedded space to which the data will be reduced. Default is 2.
    :perplexity float, optional: A parameter that determines the balance between local and global aspects of the data. Default is 30.

    return:
    :ndarray: A 2D array of shape (n_samples, n_components) containing the t-SNE transformed data.
    """
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    tsne_result = tsne.fit_transform(images)
    
    return tsne_result

def apply_gmm(mapping, n_components=10):
    gmm = GaussianMixture(n_components= n_components)  
    gmm.fit(mapping)
    labels = gmm.predict(mapping)

    return labels

def apply_gmm_proba(mapping, n_components=10):
    gmm = GaussianMixture(n_components= n_components, covariance_type='full')  
    gmm.fit(mapping)
    probabilities = gmm.predict_proba(mapping)

    return probabilities

def apply_DBSCAN(mapping, eps=0.38, min_samples=20):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(mapping)

    return labels


def perform_pca_on_images(images, n_components=5):
    """
    Function to apply Principal Component Analysis (PCA) to a collection of images.

    params:
    :images list of ndarray: A list of images represented as NumPy arrays.
    :n_components int, optional: The number of principal components to retain. Default is 5.

    return:
    :ndarray: The images transformed into the PCA space, represented as a NumPy array with 'n_components' principal components.
    """
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(images)

    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    print("Explained variance ratio per component:", explained_variance_ratio)
    print("Cumulative explained variance:", cumulative_explained_variance)

    return pca_result


def plot_dim_reduction(result, title = "t_SNE"):
    """
    Function to visualize the results of a dimensionality reduction technique in 2D.

    params:
    :result ndarray: A 2D array of shape (n_samples, 2), where each row represents a data point in the reduced 2D space.
    :title str, optional: The title for the plot. Default is "t_SNE", but can be customized based on the dimensionality reduction method used.

    return:
    :None: Displays a scatter plot of the 2D reduced data with an appropriate title.
    """
    plt.figure(figsize=(10, 7))
    plt.scatter(result[:, 0], result[:, 1], s=12, alpha = 0.5)
    
    plt.title(f'{title} of Images')
    plt.show()


def plot_clustering_predictions(results, labels, title = "t-SNE"):
    """
    Function to visualize clustering predictions by plotting the results in a 2D space.

    params:
    :results ndarray: A 2D array of shape (n_samples, n_features) representing the data points to be plotted.
    :labels array-like: An array of cluster labels corresponding to each data point in 'results'. 
    :title  string: Title for the plot. Default is t-SNE.

    return:
    :None: Displays a scatter plot where each point is colored according to its assigned cluster label.
    """
    plt.scatter(results[:, 0], results[:, 1], c=labels, s=12, alpha = 0.5, cmap='viridis')
    plt.title(f"{title} Clustering Results")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


def random_select_images(images, result, num_images=100, seed=None):
    """
    Randomly selects a specified number of images and their corresponding results.
    
    :param images: List or array of images.
    :param result: Array of associated results (e.g., labels or transformed coordinates).
    :param num_images: Number of images to randomly select (default is 100).
    :param seed: Optional random seed for reproducibility.
    
    :return: Tuple of selected images and corresponding results.
    """

    if seed is not None:
        random.seed(seed)
        
    selected_indices = random.sample(range(len(images)), num_images) 
    selected_images = [images[i] for i in selected_indices]
    selected_results = result[selected_indices]
    
    return selected_images, selected_results


def plot_with_images(result, title, selected_images, selected_result, new_size=(32, 32)):
    """
    Function to visualize 2D projection results along with corresponding images.

    params:
    :result ndarray: A 2D array of shape (n_samples, 2), where each row represents a data point in the reduced 2D space.
    :title str: The title of the plot.
    :selected_images list of ndarray: A list of images (as NumPy arrays) corresponding to the data points selected for visualization.
    :selected_result ndarray: A subset of 'result' containing the 2D coordinates for the selected images.
    :new_size tuple of int, optional: The size to which each image will be resized for display. Default is (32, 32).

    return:
    :None: Displays a 2D scatter plot with images positioned at their corresponding coordinates in the reduced space.
    """
    plt.figure(figsize=(10, 10))

    plt.scatter(result[:, 0], result[:, 1], s=9, alpha=0)
    
    # Overlay the randomly selected images (after resizing)
    for _, (image, coord) in enumerate(zip(selected_images, selected_result)):
        # Resize the image to smaller size before plotting
        resized_image = resize_image(image, new_size)
        imagebox = OffsetImage(resized_image, zoom=1, cmap='gray')
        ab = AnnotationBbox(imagebox, (coord[0], coord[1]), frameon=False)
        plt.gca().add_artist(ab)

    plt.title(f'{title} of Images with Resized Images in Plot')
    plt.show()

def overlay_orig_img_with_clusters(image, labels, block_size=64):

    # add more colors for number of clusters
    colors = {
        0: [255, 0, 0],   # red
        1: [0, 255, 0],   # green
        2: [0, 0, 255]    # blue
    }

    # number of 64x64 blocks
    block_size = 64
    num_blocks_y = image.shape[0] // block_size
    num_blocks_x = image.shape[1] // block_size

    # RGB copy of orig image
    rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # overlay with colors corresponding to labels
    for i in range(min(len(labels), num_blocks_y * num_blocks_x)):
        row = (i // num_blocks_x) * block_size
        col = (i % num_blocks_x) * block_size
        color = colors[labels[i]]
        rgb_image[row:row + block_size, col:col + block_size] = rgb_image[row:row + block_size, col:col + block_size] * 0.5 + np.array(color) * 0.5

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Grayscale Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(rgb_image)
    plt.title('Image with Cluster Colors')
    plt.axis('off')

    plt.show()