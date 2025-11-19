all: create-venv install-deps

create-venv:
	python3 -m venv venv
	@printf "Virtual environment created!"

install-deps:
	venv/bin/pip install -r requirements.txt

distribution-apple:
	@venv/bin/python3 src/Part1/Distribution.py Apple

distribution-grape:
	@venv/bin/python3 src/Part1/Distribution.py Grape

augmentation-apple:
	@venv/bin/python3 src/Part2/Augmentation.py data/original/Apple

augmentation-apple-train:
	@venv/bin/python3 src/Part2/Augmentation.py data/original/Apple -t

augmentation-grape:
	@venv/bin/python3 src/Part2/Augmentation.py data/original/Grape

augmentation-grape-train:
	@venv/bin/python3 src/Part2/Augmentation.py data/original/Grape -t

augmentation-one:
	@venv/bin/python3 src/Part2/Augmentation.py "data/original/Apple/Apple_Black_rot/image (1).JPG"

transform-one:
	@venv/bin/python3 src/Part3/Transformation.py "data/original/Apple/Apple_Black_rot/image (1).JPG"

transform-apple:
	@venv/bin/python3 src/Part3/Transformation.py -src data/original/Apple/ -dst data/tranformed/Apple/ 

transform-grape:
	@venv/bin/python3 src/Part3/Transformation.py -src data/original/Grape/ -dst data/tranformed/Grape/ 

train:
	@venv/bin/python3 src/Part4/train.py data/augmented/augmented_to_train

predict:
	@venv/bin/python3 src/Part4/predict.py data/original/Apple/Apple_rust/image\ \(56\).JPG

jupyter:
	@venv/bin/jupyter lab


.PHONY: all create-venv install-deps jupyter
