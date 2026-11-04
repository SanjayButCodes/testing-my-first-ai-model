# Online Shopper Purchase Prediction
Predicts whether an online shopping session ends in a purchase using logistic regression (pandas, matplotlib, scikit-learn).

AI101 Mini-Project | Team [number] | Python 3.10+

## Team and roles
| Member | Entry no. | GitHub handle | Role |
|---|---|---|---|
| [name] | [entry no.] | [handle] | Documentation lead |
| [name] | [entry no.] | [handle] | Execution lead |
| [name] | [entry no.] | [handle] | PPT lead |
| [name] | [entry no.] | [handle] | Video lead |

All four members contributed code (see commit history).

## Project structure
    online-shopper-purchase-prediction/
    ├── data/online_shoppers_intention.csv
    ├── src/
    │   ├── explore.py      # data understanding: mean, median, std, covariance, correlation
    │   ├── preprocess.py   # cleaning, encoding, split, scaling
    │   ├── train.py        # logistic regression training and plots
    │   ├── evaluate.py     # train/test metrics, confusion matrix, ROC
    │   └── main.py         # runs everything in order
    ├── outputs/            # plots and results are saved here
    ├── docs/               # slides
    ├── requirements.txt
    └── README.md

## Dataset
Online Shoppers Purchasing Intention Dataset (UCI / Kaggle). [rows] sessions, [columns] columns. Target: `Revenue` (purchase or not).

## Setup
Tested on Python [3.x] ([your OS]).

    git clone https://github.com/[username]/online-shopper-purchase-prediction.git
    cd online-shopper-purchase-prediction
    python -m venv venv
    source venv/bin/activate      # Windows: venv\Scripts\activate
    pip install -r requirements.txt

## How to run
1. From the top folder, run `python src/main.py`
2. The script prints the statistics and the train/test metrics, and saves plots in `outputs/`
3. Expected runtime: under a minute

## Expected output
[paste a screenshot of the terminal output here]

## Results and analysis
[3 to 5 sentences: key test metrics, which features matter most (odds ratios), train vs test comparison, effect of class imbalance, limitations]

## Links
Slides (5): docs/[filename] · Video: [unlisted link]

## AI-use disclosure
- [Tool]: [what it was used for]; we [how you checked it].
- No AI was used to generate the dataset or the results.
