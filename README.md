# GLL: Graph Learning Layer

Supplementary experiments for the Revision process.

## 1) Get the code (specific branch)
```bash
git clone -b Bohan_202508_exp --single-branch https://github.com/jwcalder/GraphLearningLayer.git
cd GraphLearningLayer
````

## 2) Setup (Python 3.12)

```bash
python3.12 -m venv .venv
source .venv/bin/activate        # macOS/Linux
# .\.venv\Scripts\activate       # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt
```

## 3) Pretrained checkpoints

Download **PreTrain\_SimCLR** from Google Drive and place it under `save/`:

Link: [https://drive.google.com/drive/folders/13Lhr76ig6M3hNMZGxWdLsSmqmJZxD3av?usp=sharing](https://drive.google.com/drive/folders/13Lhr76ig6M3hNMZGxWdLsSmqmJZxD3av?usp=sharing)

```
save/
└─ PreTrain_SimCLR/
   └─ ... (checkpoint folders)
```

## 4) Run

```bash
chmod +x fully_sup_train.sh      # first time only
./fully_sup_train.sh
```

## 5) Adjust `PAIRS` (optional)

Edit `PAIRS` inside `fully_sup_train.sh` to choose models and the number of training labels.
