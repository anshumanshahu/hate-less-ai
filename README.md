# hate less ai

A content moderation project that detects vulgar and abusive text in Hinglish, Hindi and English, and lets only clean content through to the database. It is built to run on CPU, so no GPU is needed for training or inference.

Currently the text classifier is trained and working. Image moderation (nudity, gore, offensive gestures, text inside images) is in progress.

## How it works

Incoming text is converted to TF-IDF features using character n-grams (2 to 4 characters). Because it looks at characters and not whole words, spelling variations such as `bakchod` and `bakchodd` score similarly. A Logistic Regression classifier then gives a probability that the text is vulgar.

That probability decides what happens to the content:

| Probability | Action |
|---|---|
| 0.8 and above | Rejected |
| 0.5 to 0.8 | Sent for human review |
| Below 0.5 | Clean, saved to the database |

## Results

Evaluated on a held-out test set of 37,094 rows (20% of the data).

| Metric | Value |
|---|---|
| Accuracy | 91.3% |
| Vulgar recall | 88.1% |
| Vulgar precision | 66.4% |
| Hindi (Devanagari) vulgar recall | 97.7% |

Precision and recall at different thresholds:

| Threshold | Precision | Recall |
|---|---|---|
| 0.5 | 0.664 | 0.881 |
| 0.6 | 0.718 | 0.844 |
| 0.7 | 0.775 | 0.791 |
| 0.8 | 0.836 | 0.713 |
| 0.9 | 0.911 | 0.587 |

## Project structure

```
hate-less-ai/
├── files/              # original dataset files
├── data/
│   ├── train.csv       # 80% of the data
│   └── test.csv        # 20% of the data
├── split.py            # cleans merged.csv and splits it into train/test
├── train_text.py       # trains the text model and prints the evaluation
├── threshold.py        # precision/recall at different thresholds
├── test_model.py       # interactive tester, type text and see the result
├── text_model.pkl      # trained model
├── requirements.txt
└── README.md
```

## Setup


**1. Clone the repository**

```bash
git clone https://github.com/<your-username>/hate-less-ai.git
cd hate-less-ai
```

**2. Create a virtual environment**

Linux :

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```


**3. Install the dependencies**

```bash
pip install -r requirements.txt


## Usage

### Try the trained model

`text_model.pkl` is included, so you can test it right away:

```bash
python test_model.py
```

Type any text and press Enter. Type `q` to quit.

```
> kal milte hain
   CLEAN  (vulgar probability: 0.05)
```

### Train on your own data

**1. Prepare the dataset.** Put all your CSV files in `files/`. Each file needs two columns:

```
text,label
kal milte hain,0
tu bakchod hai,1
```

`label` is `1` for vulgar and `0` for clean. Files should be saved as UTF-8. Merge everything into one `merged.csv` in the project folder.

**2. Split the data.**

```bash
python split.py
```

This removes empty rows, duplicates and rows with conflicting labels, then writes `data/train.csv` and `data/test.csv` (80/20, stratified).

**3. Train.**

```bash
python train_text.py
```

Training takes around 10 minutes on a normal CPU for about 185,000 rows. The model is saved as `text_model.pkl` and the evaluation is printed at the end, including separate scores for Hindi and for Roman/English text.



Planned: more clean Hinglish examples, a rule-based word filter, and retraining.

## Roadmap

- [x] Text classifier (Hinglish, Hindi, English)
- [x] Threshold tuning and reject / clean flow
- [ ] Retrain with more clean Hinglish data
- [ ] Image moderation (nudity, gore, offensive gestures)
- [ ] OCR for text inside images
- [ ] Database and review queue integration
