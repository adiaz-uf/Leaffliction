import random
import shutil
from pathlib import Path
import sys
import argparse


def split_dataset_stratified(base_dir=".", dst_dir="data", train_ratio=0.8):
    """
    Splits a dataset with a specific nested structure into
    train and validation sets, maintaining class proportions and structure.

    Expected source structure:
    base_dir/
        Apple/
            Apple_Black_rot/
                img1.jpg, ...
            Apple_healthy/
                img1.jpg, ...
            ...
        Grape/
            Grape_Black_rot/
                img1.jpg, ...
            ...

    Generated output structure (maintains plant type folders):
    dst_dir/
        train/
            Apple/
                Apple_Black_rot/
                    img_copy1.jpg, ...
                Apple_healthy/
                    img_copy1.jpg, ...
            Grape/
                Grape_Black_rot/
                    img_copy1.jpg, ...
        validation/
            Apple/
                Apple_Black_rot/
                    img_copy2.jpg, ...
                Apple_healthy/
                    img_copy2.jpg, ...
            Grape/
                Grape_Black_rot/
                    img_copy2.jpg, ...
    """

    print("Starting dataset split...")
    print(f"  Source: '{base_dir}'")
    print(f"  Output: '{dst_dir}'")
    print(f"  Ratio: {train_ratio*100}% train /"
          "{(1-train_ratio)*100}% validation")

    # 1. Validate source directory
    base_path = Path(base_dir)
    if not base_path.exists():
        print(f"[Error] Source directory '{base_dir}' does not exist.",
              file=sys.stderr)
        return

    if not base_path.is_dir():
        print(f"[Error] '{base_dir}' is not a directory.", file=sys.stderr)
        return

    # 2. Define output paths
    output_path = Path(dst_dir)
    train_path = output_path / "train"
    val_path = output_path / "validation"

    # 3. Check for existing output directory
    if output_path.exists():
        print(f"\n[Warning] Output directory '{output_path}' already exists.")
        try:
            # Check for user prompt availability
            if sys.stdin.isatty():
                user_input = input("  Do you want to delete it "
                                   "and start over? (y/n): ").lower()
                if user_input != 'y':
                    print("Operation cancelled by user.")
                    return
            else:
                print("  Non-interactive shell. Assuming 'yes' to overwrite.")

            shutil.rmtree(output_path)
            print("  Removed old directory.")
        except OSError as e:
            print(f"[Error] Could not remove directory '{output_path}': {e}",
                  file=sys.stderr)
            return

    try:
        # 4. Create the directory structure
        print("  Creating new directory structure...")
        train_path.mkdir(parents=True, exist_ok=True)
        val_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"[Error] Could not create directories: {e}", file=sys.stderr)
        return

    plant_types = [
        d for d in base_path.iterdir()
        if d.is_dir() and d.name in ["Apple", "Grape"]
    ]

    if not plant_types:
        print(f"[Error] No 'Apple' or 'Grape' directories found in"
              f"'{base_path}'", file=sys.stderr)
        return

    print(f"  Found plant types: {[p.name for p in plant_types]}")

    total_classes = 0
    total_files = 0

    # 5. Iterate over each plant type
    for plant_dir in plant_types:
        plant_name = plant_dir.name

        # Create plant type folders in train and validation
        (train_path / plant_name).mkdir(parents=True, exist_ok=True)
        (val_path / plant_name).mkdir(parents=True, exist_ok=True)

        # Class names (e.g., "Apple_healthy", "Grape_Black_rot")
        class_dirs = [d for d in plant_dir.iterdir() if d.is_dir()]

        for class_path in class_dirs:
            class_name = class_path.name
            total_classes += 1
            print(f"\n  Processing class: {plant_name}/{class_name}")

            # Create destination class dirs maintaining structure
            (train_path / plant_name / class_name).mkdir(parents=True,
                                                         exist_ok=True)
            (val_path / plant_name / class_name).mkdir(parents=True,
                                                       exist_ok=True)

            # Get list of all images in the class
            try:
                images = [
                    f for f in class_path.iterdir()
                    if f.is_file() and
                    f.suffix.lower() in ['.jpg', '.jpeg', '.png']
                ]
            except Exception as e:
                print(f"    [Warning] Could not read files from "
                      f"{class_path}: {e}", file=sys.stderr)
                continue

            if not images:
                print(f"    [Warning] No images found in {class_path}")
                continue

            # 6. Shuffle the images
            random.shuffle(images)

            # 7. Calculate split index
            split_idx = int(len(images) * train_ratio)

            # 8. Divide the list
            train_images = images[:split_idx]
            val_images = images[split_idx:]

            print(f"    Total: {len(images)} images.")
            print(f"    - Copying {len(train_images)} images to "
                  f"'train/{plant_name}/{class_name}'...")
            print(f"    - Copying {len(val_images)} images to "
                  f"'validation/{plant_name}/{class_name}'...")

            total_files += len(images)

            # 9. Copy files to train directory (maintaining structure)
            for img_path in train_images:
                try:
                    shutil.copy(
                        img_path,
                        train_path / plant_name / class_name / img_path.name
                    )
                except Exception as e:
                    print(f"    [Error] Failed to copy {img_path}: {e}",
                          file=sys.stderr)

            # 10. Copy files to validation directory (maintaining structure)
            for img_path in val_images:
                try:
                    shutil.copy(
                        img_path,
                        val_path / plant_name / class_name / img_path.name
                    )
                except Exception as e:
                    print(f"    [Error] Failed to copy {img_path}: {e}",
                          file=sys.stderr)

    print("\n---------------------------------")
    print("Dataset split complete.")
    print(f"  Total classes processed: {total_classes}")
    print(f"  Total files copied: {total_files}")
    print(f"  Output directory: '{output_path}'")
    print("---------------------------------")


if __name__ == "__main__":
    # Seed for random
    random.seed(42)

    parser = argparse.ArgumentParser(
        description="Split dataset into train and validation sets "
        "while maintaining directory structure.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument(
        "source_dir",
        help="Source directory containing Apple/ and Grape/ folders"
        "with class subdirectories"
    )

    parser.add_argument(
        "-o", "--output",
        dest="dst_dir",
        default="data_split",
        help="Output directory for train/validation split "
        "(default: data_split)"
    )

    parser.add_argument(
        "-r", "--ratio",
        dest="train_ratio",
        type=float,
        default=0.8,
        help="Train ratio (default: 0.8 for 80%% train, 20%% validation)"
    )

    args = parser.parse_args()

    # Validate train ratio
    if not 0 < args.train_ratio < 1:
        print(f"[Error] Train ratio must be between 0 and 1, "
              f"got {args.train_ratio}", file=sys.stderr)
        sys.exit(1)

    split_dataset_stratified(args.source_dir, args.dst_dir, args.train_ratio)
