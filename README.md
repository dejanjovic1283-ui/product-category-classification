[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dejanjovic1283-ui/product-category-classification/blob/main/Product_Category_Classification.ipynb)

# Product Category Classification

**Opis:**  
Model za **automatsku klasifikaciju proizvoda po kategorijama na osnovu naslova**. Projekat sadrži:
- Jupyter/Colab svesku sa kompletnim tokom (EDA → čišćenje → inženjering karakteristika → treniranje 2 modela → evaluacija → čuvanje modela),
- Python skripte `train_model.py` (treniranje) i `predict_category.py` (interaktivno testiranje),
- jasna uputstva za **lokalno pokretanje**, **Google Colab** i **rad preko GitHub-a**.

Skup podataka: `products.csv` (u Colab-u se čita iz **MyDrive**: `/content/drive/MyDrive/products.csv`; lokalno može biti na Desktop-u ili na putanji koju navedete).

---

## 1) Lokalno pokretanje (Windows 11 Pro, Python 3.13.3)

**Preuslovi**
- Instaliran **Python 3.13.3** (ili kompatibilna 3.x).
- Fajlovi u projektu: `Product_Category_Classification.ipynb`, `train_model.py`, `predict_category.py`, `README.md`.
- `products.csv` dostupan lokalno (npr. Desktop) ili navedite svoju putanju.

**Instalacija paketa (preporuka: virtualno okruženje)**
```bash
pip install pandas scikit-learn numpy matplotlib joblib
```

**Treniranje modela**
```bash
# primer ako je CSV na Desktop-u (zamenite <korisnik>)
python train_model.py --csv "C:\Users\<korisnik>\Desktop\products.csv"
```
Rezultat: biće sačuvan `model.pkl` (putanja je ispisana u konzoli).

**Interaktivna predikcija**
```bash
# primer ako je model sačuvan u C:\Users\<korisnik>\model.pkl
python predict_category.py --model "C:\Users\<korisnik>\model.pkl"
```
Primer unosa:
```
iphone 7 32gb gold
olympus e m10 mark iii geh use silber
q
```

---

## 2) Pokretanje u Google Colab-u

**A) Otvori svesku**  
Klikni na bedž **Open In Colab** na vrhu ovog README-a.

**B) Instalacija paketa (Colab code ćelija)**
```python
!pip install -q pandas scikit-learn numpy matplotlib joblib
```

**C) Montiranje Google Drive-a i učitavanje CSV-a (MyDrive)**
```python
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
df = pd.read_csv('/content/drive/MyDrive/products.csv')
df.head()
```

**D) Pokretanje skriptova iz Colab-a (ako klonirate repo u runtime)**
```bash
!git clone https://github.com/dejanjovic1283-ui/product-category-classification.git
!python /content/product-category-classification/train_model.py --csv "/content/drive/MyDrive/products.csv"
!python /content/product-category-classification/predict_category.py --model "/content/model.pkl"
```

**E) Alternativa**  
U svesci pokrenite **Runtime → Run all** i sledite uputstva (cela obrada je u notebook-u).

---

## 3) Rad sa GitHub-om (VS Code ili web)

**Upload preko GitHub web interfejsa**
1. Otvorite repo → **Add file → Upload files**.  
2. Dodajte: `train_model.py`, `predict_category.py`, `README.md` (ova datoteka).  
3. **Commit changes**.

**Rad kroz VS Code (po želji)**
```bash
git clone https://github.com/dejanjovic1283-ui/product-category-classification.git
cd product-category-classification
# dodajte/izmenite fajlove…
git add .
git commit -m "Add train/predict scripts and README"
git push origin main
```

---

## 4) Struktura projekta
```text
.
├─ Product_Category_Classification.ipynb   # kompletna sveska (EDA → modeli → evaluacija → save)
├─ train_model.py                          # treniranje i čuvanje modela (model.pkl)
├─ predict_category.py                     # interaktivna predikcija (učitava model.pkl)
└─ README.md                               # ovaj dokument (lokalno, Colab, GitHub uputstva)
```

---

## 5) Predaja zadatka (GitHub)

Repozitorijum treba da sadrži:
- **skup podataka** (ako odlučite da ga dodate) ili jasna uputstva za učitavanje iz **MyDrive**,
- **bar jednu .ipynb svesku** sa kompletnom analizom i razvojem rešenja,
- **Python skripte**: `train_model.py` i `predict_category.py`,
- **README.md** sa uputstvima za pokretanje i testiranje.

**Kontrolna lista**
- Notebook se otvara i izvršava (Run all).  
- `train_model.py` pravi `model.pkl`.  
- `predict_category.py` daje ispravne predikcije.  
- README je kompletan i ažuran.
