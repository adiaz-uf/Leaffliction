import sys
import os
import argparse
from image_transformations import transform_image, transform_directory


def main():

    parser = argparse.ArgumentParser(
        description="Part 3: Image Transformation (Leaffliction)",
        formatter_class=argparse.RawTextHelpFormatter,
        usage="%(prog)s [image_path] | [-src SRC_DIR -dst DST_DIR]"
        "[transformation_flag] [-h]"
    )

    # Mode 1: Single Image
    parser.add_argument(
        "image_path",
        nargs='?',  # Makes the argument optional
        default=None,
        help="Path to a single image. Triggers single-image mode."
    )

    # Mode 2: Directory
    parser.add_argument(
        "-src",
        dest="src_dir",
        help="Source directory. Triggers directory mode (requires -dst and"
        "a flag)."
    )
    parser.add_argument(
        "-dst",
        dest="dst_dir",
        help="Destination directory. Triggers directory mode (requires -src"
        "and a flag)."
    )

    args = parser.parse_args()

    # Mode 1: Directory processing
    if args.src_dir and args.dst_dir:
        transform_directory(args.src_dir, args.dst_dir)

    # Mode 2: Single image processing
    elif args.image_path:
        if args.src_dir or args.dst_dir:
            print("Error: Cannot mix single image path with directory"
                  "arguments (-src, -dst).", file=sys.stderr)
            parser.print_help()
            sys.exit(1)

        # Validate that the file exists
        if not os.path.exists(args.image_path):
            print(f"Error: File '{args.image_path}' does not exist.",
                  file=sys.stderr)
            sys.exit(1)

        # Validate that it's a file (not a directory)
        if not os.path.isfile(args.image_path):
            print(f"Error: '{args.image_path}' is not a file.",
                  file=sys.stderr)
            sys.exit(1)

        # Validate that it's an image file
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        file_ext = os.path.splitext(args.image_path.lower())[1]
        if file_ext not in valid_extensions:
            print(f"Error: '{args.image_path}' is not a valid image file.",
                  file=sys.stderr)
            print(f"Supported formats: {', '.join(valid_extensions)}",
                  file=sys.stderr)
            sys.exit(1)

        transform_image(args.image_path)

    else:
        print("Error: No arguments provided.", file=sys.stderr)
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
