# Student Information 

Peter Kyaw Okkar 35933802 

Rivera Rahna Jenell Manzano 35911072 

Gu Qiyu 35139306 

# GenAI Usage

## RAHNA
Gemini was used to assist with the code writing in the following sections:
Preprocessing - fixing any errors in the code, adding a corrupted image check and performing image preprocessing 

Prompts used: 
- 'how do i add a corrupted file check for the image files?'
- 'The images need to be in 224x224 rgb format for the model, how do i achieve this for all of the images in the dataset?'
  
## QIYU
ChatGPT was to assist in debugging, reviewing, and modifying Python code within Google Colab, focusing on the following three areas:

1. Dataset Debugging: ChatGPT helped me troubleshoot an issue caused by an extra image in the dataset and provided guidance on how to verify the total image count and identify potential problems.

2. 5-Fold Cross-Validation: I utilized ChatGPT to review and refine my 5-fold cross-validation code, specifically regarding how the dataset is partitioned into training and validation sets.

3. Model Evaluation: ChatGPT assisted in verifying that the model evaluation code correctly generates the expected metrics—including accuracy, precision, recall, F1-score, and the confusion matrix.

The suggestions provided by ChatGPT were reviewed and I made necessary adjustments, and re-ran the notebook to verify the results.

Prompts used:
- I have an extra image in my bird classification dataset, which is causing errors in my code; how should I resolve this?
- Could you help me check if my 5-fold cross-validation code is implemented correctly?
- Could you verify whether the model test results and evaluation metrics are being calculated and displayed correctly?


# bird-species-detection-system

Classifying the six wetland habitat bird species using transfer learning with EfficientNet-B0 in PyTorch. This is the model training section of the Bird Watcher AI Assistant.
- Hooded Merganser
- Black Crowned Night Heron
- Cormorant
- Common Mallard
- Common Kingfisher
- Marsh Sandpiper

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
NOTE: 'the bird-species-detection-system.ipynb' file is the old version which identifies only 2 species and should not be used.
The following files are required:

- `bird-species-detection-system_v2.0.ipynb`
- The bird image dataset, provided as `bird_data.zip`

The notebook file contains the Python source code, while the zip archive contains the colour and segmentation images used for training, validation and testing.

The dataset file name and paths must match the file names used in the notebook code:

```
/content/bird_data.zip
```

### Running the Project

1. Open Google Colab or Jupyter Notebook.
2. Open `bird-species-detection-system_v2.0.ipynb`.
3. If Google Colab is used, mount Google Drive (`drive.mount('/content/drive')`) and upload `bird_data.zip` to the Colab working directory so that it sits at `/content/bird_data.zip`.
4. Run the package installation cell if required.
5. Run all notebook cells from top to bottom in order.
6. Do not skip the dataset integrity checks, the preprocessing (resize to 224 × 224) step, the stratified train/validation split, or the held-out test split.
7. Wait for each cell to finish before continuing to the next cell. The cross-validation cell trains 5 folds × 25 epochs and takes several minutes on a GPU.
8. In Google Colab, the user can also select:

   Runtime → Run all

   after the dataset has been uploaded and the Drive is mounted.

### Dataset

The dataset contains 641 colour images with 6 classes:

- `mallard`
- `hooded merganser`
- `black crowned night heron`
- `common kingfisher`
- `greater cormorant`
- `marsh sandpiper` 

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
- A 64-image held-out test set, split off before cross-validation so test images are never seen during training
- AdamW with discriminative learning rates — 1e-4 for the unfrozen backbone blocks, 1e-3 for the new classification head — plus weight decay 1e-4
- 10 epochs per fold, batch size 16
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

The current model achieves approximately:

**Training Accuracy: 78.98%**
**Testing Accuracy: 89.06%**

Five-fold cross-validation accuracy lands around the same level, with the per-fold best validation accuracy typically between 0.90 and 1.00. The held-out test classification report is:

```
                           precision    recall  f1-score   support

black crowned night heron       0.87      0.81      0.84        16
        common kingfisher       0.90      1.00      0.95         9
        greater cormorant       0.92      0.86      0.89        14
         hooded merganser       0.86      1.00      0.92         6
                  mallard       1.00      1.00      1.00         6
          marsh sandpiper       0.85      0.85      0.85        13

                 accuracy                           0.89        64
                macro avg       0.90      0.92      0.91        64
             weighted avg       0.89      0.89      0.89        64
```

Confusion matrix:

```
[[ 13	1	0	1	0	1]
[0	9	0	0	0	0]
[1	0	12	0	0	1]
[0	0	0	6	0	0]
[0	0	0	0	6	0]
[1	0	1	0	0	11]]
```

Common Kingfisher: All 9 images were correctly classified (100%).

Hooded Merganser: All 6 images were correctly classified (100%).

Mallard: All 6 images were correctly classified (100%).

Black Crowned Night Heron: 13 out of 16 images were correctly classified (81.25%).

Greater Cormorant: 12 out of 14 images were correctly classified (85.71%).

Marsh Sandpiper: 11 out of 13 images were correctly classified (84.62%).

### Troubleshooting

- If a `FileNotFoundError` occurs, check that `bird_data.zip` has been uploaded to `/content/` and that the file name matches the notebook code exactly. If the zip was extracted already, check that `/content/bird_data/` exists with `color/` and `segmentation/` subfolders.
- If the zip cannot be extracted, confirm the archive is not corrupted and that `/content/bird_data.zip` is the correct path.
- If `CUDA available: False` is printed, the runtime has no GPU. Select `Runtime → Change runtime type → T4 GPU`, or expect training to be much slower on CPU.
- If the `best_fold{fold}.pt` checkpoint files cannot be found, the cross-validation cell was skipped or the runtime restarted. Checkpoints are saved to `/content/` and are lost when the session ends, so re-run the training cells.
- If variables such as `targets`, `idx_test`, `fold_results` or `build_model` are not defined, some earlier cells may have been skipped. Run the notebook again from the top.
- If the interactive image classifier widget does not appear, run the `ipywidgets` import cell in Colab. Widgets are not rendered in a plain GitHub notebook preview; they require a live kernel.
- If Google Colab is restarted or disconnected, the dataset zip and the Drive mount may need to be set up again, and training must be re-run to regenerate the fold checkpoints.
