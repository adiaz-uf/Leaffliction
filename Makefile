.PHONY: all create-venv install-deps jupyter

all: create-venv install-deps

create-venv:
	python3 -m venv venv
	@printf "Virtual environment created!"

install-deps:
	venv/bin/pip install -r requirements.txt


augmentation-apple:
	python src/Part2/Augmentation.py Apple

augmentation-grape:
	python src/Part2/Augmentation.py Grape


augmentation-one:
	python src/Part2/Augmentation.py "apple/Apple_Black_rot/image (1).JPG"

jupyter:
	@venv/bin/jupyter lab
