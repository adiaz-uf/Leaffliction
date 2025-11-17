# Leaffliction

Computer vision: Image classification by disease recognition on leaves

# 🚀 Getting Started

1. **Clone the repository**

```bash
git clone https://github.com/adiaz-uf/Leaffliction.git
cd Leaffliction
```

2. **Create environment and install dependencies**

```bash
make
```

## Part 1: Analysis of the Data Set

```bash
python src/Part1/Distribution.py Apple/
```

<img width="1512" height="776" alt="Screenshot 2025-11-17 at 12 30 51" src="https://github.com/user-attachments/assets/e829faaf-8b58-4091-9102-0c876c4ca517" />

## Part 2: Data augmentation

Displays different data augmentation techniques applied to the image or directory of images passed as argument.

```bash
make augment-one

make augment-apple

make augment-apple-train

make augment-grape

make augment-grape-train
```

<img width="1612" height="976" alt="Screenshot 2025-11-17 at 12 30 21" src="https://github.com/user-attachments/assets/09e449ed-a35e-41bd-a697-9451f4aadf46" />

## Part 3: Image transformations

```bash
make transform-one

make transform-apple

make transform-grape
```

<img width="1258" height="880" alt="Captura desde 2025-11-10 11-38-35" src="https://github.com/user-attachments/assets/f3877974-d68d-4419-bf0d-473ffda18bdb" />
<img width="1503" height="875" alt="image" src="https://github.com/user-attachments/assets/5a22c1d4-7ad9-42d6-983d-2df440f478c9" />

## Part 4: Classification

### Model predictions

This program takes as arguments a path to an image from our learnings, displays the original
image and the transformed image, and gives the type of disease specified in the leaf.

Train uses the images passed as argument to train the model, and creates `leaf_disease_model.h5` and `class_indices.json` files.

You must train the model first and then run the predictions.

```bash
make train

make predict
```

<img width="1241" height="743" alt="Captura desde 2025-11-13 19-22-30" src="https://github.com/user-attachments/assets/3fe5a00d-4254-458f-ae39-4dc1c76f3d84" />

# Team work 💪

This project was a team effort. You can checkout the team members here:

-   **Alejandro Díaz Ufano Pérez**
    -   [Github](https://github.com/adiaz-uf)
    -   [LinkedIn](https://www.linkedin.com/in/alejandro-d%C3%ADaz-35a996303/)
    -   [42 intra](https://profile.intra.42.fr/users/adiaz-uf)
-   **Alejandro Aparicio**
    -   [Github](https://github.com/magnitopic)
    -   [LinkedIn](https://www.linkedin.com/in/magnitopic/)
    -   [42 intra](https://profile.intra.42.fr/users/alaparic)
