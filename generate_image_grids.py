import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.gridspec as gridspec


def plot_image_rows(real_image_path, fake_images, mask_images, output_images, output_path):
    """
    Plots three rows of images for each variation with larger column headings and saves to output_path.

    Args:
        real_image_path (str): Path to the real image.
        fake_images (list of str): Paths to the fake images (3 variations).
        mask_images (list of str): Paths to the mask images (3 variations).
        output_images (list of str): Paths to the output images (3 variations).
        output_path (str): Path to save the rows of images.
    """
    num_rows = len(fake_images)  # Should be 3
    num_cols = 4  # Real, Fake (Inpainted), Mask, Output (DeCLIP)
    column_titles = ["Real", "Inpainted", "Mask", "DeCLIP"]

    # Set up the figure with a tighter layout
    fig = plt.figure(figsize=(num_cols * 4, (num_rows + 0.5) * 4))
    spec = gridspec.GridSpec(num_rows + 1, num_cols, figure=fig, height_ratios=[0.2] + [1] * num_rows)

    # Add column headings with larger font size
    for i, title in enumerate(column_titles):
        ax = fig.add_subplot(spec[0, i])
        ax.text(0.5, 0.5, title, fontsize=24, ha="center", va="center")  # Increased font size
        ax.axis("off")

    # Adjust spacing to make headings closer to images
    plt.subplots_adjust(top=0.9, hspace=0.1)

    # Add images in subsequent rows
    for i in range(num_rows):
        images = [
            mpimg.imread(real_image_path),
            mpimg.imread(fake_images[i]),
            mpimg.imread(mask_images[i]),
            mpimg.imread(output_images[i]),
        ]

        for j, img in enumerate(images):
            ax = fig.add_subplot(spec[i + 1, j])
            ax.imshow(img)
            ax.axis("off")

    # Save the plot as an image
    plt.tight_layout(pad=0.5)
    plt.savefig(output_path)
    plt.close()


def process_dataset(dataset, real_dir, fake_dir_template, mask_dir_template, output_dir_template, output_rows_dir):
    """
    Processes the dataset, creating rows of images.

    Args:
        dataset (str): Dataset name.
        real_dir (str): Directory of real images.
        fake_dir_template (str): Template for fake image directory (contains [dataset]).
        mask_dir_template (str): Template for mask directory (contains [dataset]).
        output_dir_template (str): Template for output image directory (contains [dataset]).
        output_rows_dir (str): Directory to save the output rows of images.
    """
    # Update paths with the dataset name
    fake_dir = fake_dir_template.replace("[dataset]", dataset)
    mask_dir = mask_dir_template.replace("[dataset]", dataset)
    output_dir = output_dir_template.replace("[dataset]", dataset)
    output_rows_dataset_dir = os.path.join(output_rows_dir, dataset)

    # Create output directory if it doesn't exist
    os.makedirs(output_rows_dataset_dir, exist_ok=True)

    # Process each real image
    for real_image_name in sorted(os.listdir(real_dir)):
        if real_image_name.endswith(".png"):
            real_image_path = os.path.join(real_dir, real_image_name)

            # Generate fake, mask, and output image paths
            base_name = os.path.splitext(real_image_name)[0]
            fake_images = [os.path.join(fake_dir, f"{base_name}-{i}.png") for i in range(3)]
            mask_images = [os.path.join(mask_dir, f"{base_name}-{i}.png") for i in range(3)]
            output_images = [os.path.join(output_dir, f"{base_name}-{i}.png") for i in range(3)]

            # Ensure all files exist
            if all(os.path.exists(p) for p in fake_images + mask_images + output_images):
                # Save the rows of images
                output_row_path = os.path.join(output_rows_dataset_dir, f"{base_name}_grid.png")
                plot_image_rows(real_image_path, fake_images, mask_images, output_images, output_row_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate rows of images for comparison.")
    parser.add_argument("dataset", type=str, help="Dataset name to process.")
    parser.add_argument("--real_dir", type=str, default="./datasets/dolos_data/celebahq/real/test",
                        help="Path to the directory of real images.")
    parser.add_argument("--fake_dir_template", type=str,
                        default="./datasets/dolos_data/celebahq/fake/[dataset]/images/test",
                        help="Template path for fake image directory, containing [dataset].")
    parser.add_argument("--mask_dir_template", type=str,
                        default="./datasets/dolos_data/celebahq/fake/[dataset]/masks/test",
                        help="Template path for mask directory, containing [dataset].")
    parser.add_argument("--output_dir_template", type=str,
                        default="./output-images/[dataset]",
                        help="Template path for output image directory, containing [dataset].")
    parser.add_argument("--output_rows_dir", type=str, default="output-grids",
                        help="Directory to save the output rows of images.")

    args = parser.parse_args()
    process_dataset(args.dataset, args.real_dir, args.fake_dir_template, args.mask_dir_template, args.output_dir_template, args.output_rows_dir)
