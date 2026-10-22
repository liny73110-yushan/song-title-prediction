# Song Title Prediction from Humming and Whistling

Mini-project for ECS7020P at Queen Mary University of London. The goal is to predict which song someone is humming or whistling from a short audio recording, framed as a multi-class classification problem.

This is a hard problem: people hum and whistle with very different pitch, tempo, rhythm and recording quality, and the dataset is small.

## Dataset

The project uses the 400-sample subset of the **MLEnd Hums and Whistles II Dataset**, provided as part of the ECS7020P module at Queen Mary University of London.

- 400 `.wav` recordings of hums and whistles
- 8 songs: Feeling, Friend, Happy, Married, Necessities, NewYork, RememberMe, TryEverything

The song label is taken from the last part of each file name (e.g. `S100_hum_2_Married.wav` → `Married`).

**The audio recordings are not included in this repository**, as they were provided for coursework use only.

## Method

Two pipelines were built and compared, using the same stratified 80/20 train/validation split (320 / 80 samples):

| | Pipeline A | Pipeline B |
|---|---|---|
| Features | 20 MFCCs, averaged over time (20-dim vector) | 128-band Mel-spectrogram (dB), padded/cropped to 300 frames, normalised |
| Model | SVM (RBF kernel) with standardisation | CNN: 3 convolution + pooling blocks, dense layer with dropout, softmax |
| Training | scikit-learn defaults | Adam, sparse categorical cross-entropy, 20 epochs, batch size 16 |

## Results

| Model | Validation accuracy |
|---|---|
| Random guessing (8 classes) | 12.5% |
| MFCC + SVM | 16.25% |
| Mel-spectrogram + CNN | 25% (final epoch), 28.75% (best epoch) |

**Key findings**

- The CNN clearly outperformed the SVM, suggesting that keeping the time–frequency structure of the melody matters more than time-averaged timbral features.
- The CNN overfitted strongly: training accuracy reached about 86% while validation accuracy stayed around 25%.
- Both models remain far from practically useful accuracy.

## Limitations

- Small dataset (400 samples) and a single train/validation split.
- The best CNN accuracy was selected on the validation set without a separate test set, so it is likely optimistic.
- The split was not grouped by participant, so recordings from the same person may appear in both sets.

## Possible improvements

- Data augmentation (pitch shifting, time stretching)
- Early stopping and a smaller model to reduce overfitting
- Pitch-based features such as chroma or pitch contours
- Participant-grouped cross-validation
- Training on the larger 800-sample subset

## How to run

The notebook was written for Google Colab, with the dataset stored in Google Drive.

1. Obtain `MLEndHWII_Sample_400.zip` (provided through the ECS7020P module) and upload it to your Google Drive (e.g. `MyDrive/`).
2. In Colab, mount Google Drive and extract the audio files to `/content/dataset/`:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')

   !mkdir -p /content/dataset
   !unzip -q -j /content/drive/MyDrive/MLEndHWII_Sample_400.zip "*.wav" -d /content/dataset
   ```

3. Run all cells in the notebook.

Requirements: `librosa`, `numpy`, `pandas`, `scikit-learn`, `tensorflow`, `matplotlib`, `seaborn`, `tqdm`
