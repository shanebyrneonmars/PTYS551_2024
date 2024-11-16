import numpy as np
import os
import copy
import matplotlib
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Function to calculate class weights
def calculate_class_weights(masks, num_classes):
    """
    Calculate class weights to handle class imbalance in the dataset.

    Parameters:
    masks (numpy array): Array of mask images.
    num_classes (int): Number of unique classes.

    Returns:
    dict: Dictionary mapping class indices to their corresponding weights.
    """
    # Flatten the mask array to get a 1D array of class labels
    flat_masks = masks.flatten()
    
    # Compute class weights using sklearn's compute_class_weight function
    class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=flat_masks)
    
    # Create a dictionary mapping class indices to weights
    class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    return class_weights_dict

# Function to load and preprocess an image
def load_and_preprocess_image(img_path, target_size):
    """
    Load and preprocess an image.

    Parameters:
    img_path (str): Path to the image file.
    target_size (tuple): Target size to resize the image.

    Returns:
    numpy array: Preprocessed image array.
    """
    # Load the image and resize it to the target size
    img = load_img(img_path, target_size=target_size)
    
    # Convert the image to a numpy array
    img = img_to_array(img)
    
    # Normalize the image to the range [0, 1]
    img = (img - img.min()) / (img.max() - img.min())
    
    return img

# Function to load and preprocess a mask
def load_and_preprocess_mask(mask_path, target_size, 
                             class_mapping, color_mapping,
                             classes_to_combine=None, combined_class_name="CombinedClass",
                             selected_classes=None, inclusive=False):
    """
    Load and preprocess a mask image, with options to combine classes and select specific classes.

    Parameters:
    mask_path (str): Path to the mask file.
    target_size (tuple): Target size to resize the mask.
    class_mapping (dict): Dictionary mapping class names to class indices.
    color_mapping (dict): Dictionary mapping class names to colors.
    classes_to_combine (list, optional): List of class names to combine into a single class.
    combined_class_name (str, optional): Name of the combined class.
    selected_classes (list, optional): List of class names to select.
    inclusive (bool, optional): Whether to include masks that contain any of the selected classes.

    Returns:
    tuple: Preprocessed mask array and unique values in the mask.
    """
    # Load the mask image and resize it to the target size
    mask = load_img(mask_path, target_size=target_size, color_mode="grayscale")
    
    # Convert the mask to a numpy array
    mask = img_to_array(mask)
    
    # Remove the last dimension (channel) as it's not needed for grayscale images
    mask = np.squeeze(mask, axis=-1)
    
    # Get unique values in the mask
    unique_values = np.unique(mask)
    
    # If selected_classes is provided, filter the mask based on selected classes
    if selected_classes is not None:
        selected_class_indices = [class_mapping[cls] for cls in selected_classes]
        if inclusive:
            contains_selected_classes = any(val in selected_class_indices for val in unique_values)
        else:
            contains_selected_classes = all(val in selected_class_indices for val in unique_values) and \
                                        all(val in unique_values for val in selected_class_indices)
        if not contains_selected_classes:
            return None, None
        
        if classes_to_combine is None:
            # Remap class values
            remap_dict = {old: new for new, old in enumerate(selected_class_indices)}
            mask = np.vectorize(remap_dict.get)(mask)
        
    # If classes_to_combine is provided, combine the specified classes into a single class
    if classes_to_combine is not None:
        combined_class_mapping, _, original_class_ids = reduce_and_combine_classes(class_mapping, color_mapping,
                                                                                   classes_to_combine, combined_class_name=combined_class_name,
                                                                                   selected_classes=selected_classes)
        # Create a remap_dict to map original class values to new values
        remap_dict = {}
        for cls, cls_id in original_class_ids.items():
            if cls in classes_to_combine:
                remap_dict[cls_id] = combined_class_mapping[combined_class_name]
            elif cls in combined_class_mapping:
                remap_dict[cls_id] = combined_class_mapping[cls]
                
        # Remap class values in the mask
        mask = np.vectorize(remap_dict.get)(mask)

    # Expand the mask dimensions to add a channel dimension
    mask = np.expand_dims(mask, axis=-1)
    
    # Get unique values in the remapped mask
    unique_values = np.unique(mask)
    
    return mask.astype(np.int32), unique_values

# Load images and masks
def load_data(images_dir, masks_dir, target_size, 
              class_mapping, color_mapping,
              classes_to_combine=None, combined_class_name="CombinedClass", 
              num_images=None, selected_classes=None, inclusive=False):
    """
    Load and preprocess images and masks from directories.

    Parameters:
    images_dir (str): Directory containing image files.
    masks_dir (str): Directory containing mask files.
    target_size (tuple): Target size to resize the images and masks.
    class_mapping (dict): Dictionary mapping class names to class indices.
    color_mapping (dict): Dictionary mapping class names to colors.
    classes_to_combine (list, optional): List of class names to combine into a single class.
    combined_class_name (str, optional): Name of the combined class.
    num_images (int, optional): Number of images to load.
    selected_classes (list, optional): List of class names to select.
    inclusive (bool, optional): Whether to include masks that contain any of the selected classes.

    Returns:
    tuple: Arrays of preprocessed images and masks, and the number of unique classes.
    """
    # Get sorted list of image and mask files
    image_files = sorted(os.listdir(images_dir))
    mask_files = sorted(os.listdir(masks_dir))
    
    # If num_images is specified, limit the number of files to load
    if num_images is not None:
        image_files = image_files[:num_images]
        mask_files = mask_files[:num_images]
    
    images = []
    masks = []
    all_unique_values = set()
    
    # Load and preprocess each image and mask
    for img_file, mask_file in zip(image_files, mask_files):
        img_path = os.path.join(images_dir, img_file)
        mask_path = os.path.join(masks_dir, mask_file)
        
        img = load_and_preprocess_image(img_path, target_size)
        mask, unique_values = load_and_preprocess_mask(mask_path, target_size, 
                                                       class_mapping, color_mapping,
                                                       classes_to_combine=classes_to_combine, combined_class_name=combined_class_name,
                                                       selected_classes=selected_classes, inclusive=inclusive)
        
        if mask is not None:
            images.append(img)
            masks.append(mask)
            all_unique_values.update(unique_values)
    
    images = np.array(images)
    masks = np.array(masks)
    num_classes = len(all_unique_values)
    
    return images, masks, num_classes

def reduce_and_combine_classes(class_mapping, color_mapping, 
                               classes_to_combine, combined_class_name="CombinedClass", 
                               selected_classes=None, cmap_name='viridis'):
    """
    Combine specified classes into a single class and update class and color mappings.

    Parameters:
    class_mapping (dict): Dictionary mapping class names to class indices.
    color_mapping (dict): Dictionary mapping class names to colors.
    classes_to_combine (list): List of class names to combine into a single class.
    combined_class_name (str, optional): Name of the combined class.
    selected_classes (list, optional): List of class names to select.
    cmap_name (str, optional): Name of the colormap to use for generating colors.

    Returns:
    tuple: Updated class mapping, updated color mapping, and original class IDs.
    """
    combined_class_id = max(class_mapping.values()) + 1
    
    # Store original class IDs
    original_class_ids = copy.deepcopy(class_mapping)
    class_mapping_ = copy.deepcopy(class_mapping)
    
    # Update class mapping by removing classes to combine and adding the combined class
    for cls in classes_to_combine:
        if cls in class_mapping_:
            del class_mapping_[cls]
    class_mapping_[combined_class_name] = combined_class_id

    # Update color mapping by removing classes to combine and adding the combined class color
    combined_color = (128, 128, 128)  # Example color for the combined class
    for cls in classes_to_combine:
        if cls in color_mapping:
            del color_mapping[cls]
    color_mapping[combined_class_name] = combined_color
    
    if selected_classes is not None:
        # Remove classes to combine from the selected_classes list and add the combined class
        selected_classes = [cls for cls in selected_classes if cls not in classes_to_combine]
        selected_classes.append(combined_class_name)

        # Remove all the classes from class_mapping that are not in selected_classes
        selected_classes_mapping = {cls: class_mapping_[cls] for cls in selected_classes if cls in class_mapping_}
        class_mapping_ = {old: new for new, old in enumerate(selected_classes_mapping)}

    # Update color mapping with new class indices
    class_names = class_mapping_.keys()
    viridis = matplotlib.colormaps[cmap_name].resampled(len(class_names))
    colors = viridis(np.linspace(0, 1, len(class_names)))
    color_mapping = {name: colors[idx] for idx, name in enumerate(class_names)}
    
    return class_mapping_, color_mapping, original_class_ids

# Function to apply color mapping to mask image
def apply_color_mapping(mask, class_mapping, color_mapping):
    """
    Apply color mapping to a mask image.

    Parameters:
    mask (numpy array): Mask image array.
    class_mapping (dict): Dictionary mapping class names to class indices.
    color_mapping (dict): Dictionary mapping class names to colors.

    Returns:
    numpy array: Colored mask image.
    """
    height, width = mask.shape
    colored_mask = np.zeros((height, width, 4))  # RGBA image

    # Apply color mapping to each class in the mask
    for class_name, color in color_mapping.items():
        class_value = class_mapping[class_name]
        colored_mask[mask == class_value] = color

    return colored_mask

# Display a sample image and mask with custom color mapping
def display_color_sample(image, mask, class_mapping, color_mapping):
    """
    Display a sample image and its corresponding mask with custom color mapping.

    Parameters:
    image (numpy array): Image array.
    mask (numpy array): Mask array.
    class_mapping (dict): Dictionary mapping class names to class indices.
    color_mapping (dict): Dictionary mapping class names to colors.
    """
    colored_mask = apply_color_mapping(mask[:, :, 0], class_mapping, color_mapping)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Sample Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(colored_mask)
    plt.title('Corresponding Mask')
    plt.axis('off')
    plt.show()

# Display a sample image and mask
def display_sample(image, mask):
    """
    Display a sample image and its corresponding mask.

    Parameters:
    image (numpy array): Image array.
    mask (numpy array): Mask array.
    """
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Sample Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask[:, :, 0], cmap='gray')
    plt.title('Corresponding Mask')
    plt.axis('off')
    plt.show()
