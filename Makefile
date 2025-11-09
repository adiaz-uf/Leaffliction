.PHONY: all create-venv install-deps jupyter

all: create-venv install-deps

create-venv:
	python3 -m venv venv
	@printf "Virtual environment created!"

install-deps:
	venv/bin/pip install -r requirements.txt

augmentation-apple:
	@venv/bin/python3 src/Part2/Augmentation.py Apple

augmentation-grape:
	@venv/bin/python3 src/Part2/Augmentation.py Grape

augmentation-one:
	@venv/bin/python3 src/Part2/Augmentation.py "apple/Apple_Black_rot/image (1).JPG"

train:
	@venv/bin/python3 src/Part4/train.py data/original

train-apple:
	@venv/bin/python3 src/Part4/train.py data/original/apple

train-grape:
	@venv/bin/python3 src/Part4/train.py data/original/grape

jupyter:
	@venv/bin/jupyter lab
