# bird-species-detection-system

Classifying the two bird species (Mallard and Hooded Merganser) using transfer learning with EfficientNet-B0 in PyTorch.

## User Manual

### Development Environment

This project can be run using Google Colab or Jupyter Notebook. Python 3 is required to execute the notebook and train the model. Google Colab is recommended because it provides an online Python environment with a free GPU (the notebook was developed on a Tesla T4) and does not require a local Python installation.

The project uses the following Python libraries:

- torch, torchvision
- scikit-learn
- numpy
- pandas
- matplotlib, seaborn
- Pillow (PIL)
- ipywidgets (for the interactive classification demo)

If the required libraries are not installed, they can be installed using:

```
pip install torch torchvision scikit-learn numpy pandas matplotlib seaborn pillow ipywidgets
```

### Required Files

The following files are required:

- `bird-species-detection-system.ipynb`
- The bird image dataset, provided as `bird_data.zip`

The notebook file contains the Python source code, while the zip archive contains the colour and segmentation images used for training, validation and testing.

The dataset file name and paths must match the file names used in the notebook code:

```
/content/bird_data.zip
/content/bird_data/color/
/content/bird_data/segmentation/
```

### Running the Project

1. Open Google Colab or Jupyter Notebook.
2. Open `bird-species-detection-system.ipynb`.
3. If Google Colab is used, mount Google Drive (`drive.mount('/content/drive')`) and upload `bird_data.zip` to the Colab working directory so that it sits at `/content/bird_data.zip`.
4. Run the package installation cell if required.
5. Run all notebook cells from top to bottom in order.
6. Do not skip the dataset integrity checks, the preprocessing (resize to 224 × 224) step, the stratified train/validation split, or the held-out test split.
7. Wait for each cell to finish before continuing to the next cell. The cross-validation cell trains 5 folds × 25 epochs and takes several minutes on a GPU.
8. In Google Colab, the user can also select:

   Runtime → Run all

   after the dataset has been uploaded and the Drive is mounted.

### Dataset

The dataset contains 120 colour images and 120 segmentation masks, 60 of each species, split evenly between two classes:

- `087.Mallard`
- `089.Hooded_Merganser`

Colour images are supplied in varying resolutions and are converted to RGB, resized to fit a 224 × 224 square with the aspect ratio preserved, and padded with black where needed. Segmentation masks are resized with nearest-neighbour interpolation and kept as single-channel grayscale.

Class labels are assigned alphabetically by the image folder loader:

- `087.Mallard` = 0
- `089.Hooded_Merganser` = 1

### Data Integrity Checks

Before any training, the notebook verifies the dataset is usable:

- File counts and formats per class for both colour and segmentation folders
- Corrupted image detection (`PIL.Image.verify()`)
- Image size and colour-mode distribution
- Colour / segmentation pairing — every colour image has a matching mask
- Exact duplicate detection via MD5 hashing

### Model Training and Testing

The notebook trains an EfficientNet-B0 backbone pre-trained on ImageNet, with its original classifier replaced by dropout (p = 0.4) and a linear layer for 2 classes. The last two feature blocks are unfrozen so the backbone adapts to Mallards and Mergansers, while the remaining layers stay frozen.

Training uses:

- Stratified 5-fold cross-validation on a 100-image train/validation pool
- A 20-image held-out test set (10 per class), split off before cross-validation so test images are never seen during training
- AdamW with discriminative learning rates — 1e-4 for the unfrozen backbone blocks, 1e-3 for the new classification head — plus weight decay 1e-4
- 25 epochs per fold, batch size 16
- Random horizontal flip, rotation (±15°), colour jitter and random resized crop as training augmentation
- ImageNet normalisation for both training and evaluation

The best checkpoint of each fold is saved by lowest validation loss. Final predictions are made by averaging the softmax probabilities of all five fold models (a simple ensemble) before taking the argmax.

### Expected Results

After the notebook is executed successfully, the user should be able to see:

- Dataset information and per-class file counts
- Data distribution graphs
- Per-epoch training and validation loss and accuracy
- Cross-validation accuracy (mean ± standard deviation)
- Held-out test accuracy
- Precision, recall and F1-score
- Classification Report
- Confusion Matrix and its heatmap visualisation
- ROC Curve with AUC
- Precision-Recall Curve with average precision
- An interactive widget for classifying individual held-out images with confidence scores

The current model achieves approximately:

**Training Accuracy: 100%**
**Testing Accuracy: 95.0% (19/20)**

Five-fold cross-validation accuracy lands around the same level, with the per-fold best validation accuracy typically between 0.90 and 1.00. The held-out test classification report is:

```
                      precision    recall  f1-score   support

         087.Mallard      1.000     0.900     0.947        10
089.Hooded_Merganser      0.909     1.000     0.952        10

            accuracy                          0.950        20
           macro avg      0.955     0.950     0.950        20
        weighted avg      0.955     0.950     0.950        20
```

Confusion matrix:

```
[[ 9  1]
 [ 0 10]]
```

One Mallard is misclassified as a Hooded Merganser; no Merganser is misclassified. These results show the performance of the transfer-learning model in classifying Mallard and Hooded Merganser images.

Note on interpreting these figures: training accuracy reaches 100% by the later epochs, which is a sign of the model memorising this very small dataset (100 training images). The held-out and cross-validation numbers are the ones that reflect genuine generalisation.

### Troubleshooting

- If a `FileNotFoundError` occurs, check that `bird_data.zip` has been uploaded to `/content/` and that the file name matches the notebook code exactly. If the zip was extracted already, check that `/content/bird_data/` exists with `color/` and `segmentation/` subfolders.
- If the zip cannot be extracted, confirm the archive is not corrupted and that `/content/bird_data.zip` is the correct path.
- If `CUDA available: False` is printed, the runtime has no GPU. Select `Runtime → Change runtime type → T4 GPU`, or expect training to be much slower on CPU.
- If the `best_fold{fold}.pt` checkpoint files cannot be found, the cross-validation cell was skipped or the runtime restarted. Checkpoints are saved to `/content/` and are lost when the session ends, so re-run the training cells.
- If variables such as `targets`, `idx_test`, `fold_results` or `build_model` are not defined, some earlier cells may have been skipped. Run the notebook again from the top.
- If the interactive image classifier widget does not appear, run the `ipywidgets` import cell in Colab. Widgets are not rendered in a plain GitHub notebook preview; they require a live kernel.
- If Google Colab is restarted or disconnected, the dataset zip and the Drive mount may need to be set up again, and training must be re-run to regenerate the fold checkpoints.
