import sys

from image_transformations import transform_image


def main():
    transform_image(sys.argv[1])


if __name__ == "__main__":
    main()
