# Image Classifier con PyTorch e Streamlit

Questo progetto permette di addestrare un modello di classificazione immagini con PyTorch e di usarlo tramite una semplice interfaccia web sviluppata con Streamlit.

Il modello è generico: puoi creare il tuo dataset con le tue classi, addestrare il modello e poi caricare immagini nella UI per ottenere una predizione.

---

## Struttura del progetto

La struttura consigliata è questa:

```text
project/
│
├── dataset/
│   ├── classe_1/
│   │   ├── immagine1.jpg
│   │   └── immagine2.jpg
│   │
│   ├── classe_2/
│   │   ├── immagine1.jpg
│   │   └── immagine2.jpg
│   │
│   └── classe_3/
│       ├── immagine1.jpg
│       └── immagine2.jpg
│
├── train.py
├── ui.py
├── requirements.txt
├── README.md
│
├── model.pth
└── class_names.json
```

I file `model.pth` e `class_names.json` vengono creati automaticamente dopo il training.

---

## Requisiti

Prima di iniziare, assicurati di avere installato:

- Python 3.9 o superiore
- pip
- un ambiente virtuale Python, consigliato ma non obbligatorio

---

## Installazione

### 1. Clona il repository

```bash
git clone https://github.com/tuo-username/nome-repository.git
cd nome-repository
```

### 2. Crea un ambiente virtuale

Su macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Su Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Installa le dipendenze

```bash
pip install -r requirements.txt
```

Il file `requirements.txt` deve contenere:

```txt
streamlit>=1.30.0
torch>=2.0.0
torchvision>=0.15.0
pillow>=10.0.0
scikit-learn>=1.3.0
```

---

## Preparazione del dataset

Il dataset deve essere inserito nella cartella `dataset`.

Ogni classe deve avere una propria sottocartella. Il nome della sottocartella sarà usato come nome della classe.

Esempio:

```text
dataset/
│
├── cane/
│   ├── cane1.jpg
│   ├── cane2.jpg
│   └── cane3.jpg
│
├── gatto/
│   ├── gatto1.jpg
│   ├── gatto2.jpg
│   └── gatto3.jpg
│
└── cavallo/
    ├── cavallo1.jpg
    ├── cavallo2.jpg
    └── cavallo3.jpg
```

In questo esempio il modello imparerà a riconoscere:

- cane
- gatto
- cavallo

Le immagini non devono essere messe direttamente dentro `dataset`, ma dentro le sottocartelle delle rispettive classi.

Struttura corretta:

```text
dataset/
├── classe_1/
│   ├── img1.jpg
│   └── img2.jpg
│
└── classe_2/
    ├── img1.jpg
    └── img2.jpg
```

Struttura sbagliata:

```text
dataset/
├── img1.jpg
├── img2.jpg
└── img3.jpg
```

---

## Formati immagine supportati

Puoi usare immagini nei formati più comuni:

- `.jpg`
- `.jpeg`
- `.png`
- `.webp`

---

## Addestramento del modello

Dopo aver preparato il dataset, avvia il training con:

```bash
python train.py
```

Durante il training il codice:

1. legge le immagini dalla cartella `dataset`;
2. trova automaticamente le classi;
3. addestra un modello basato su EfficientNet V2-S;
4. usa la K-Fold Cross Validation;
5. applica data augmentation alle immagini;
6. usa early stopping;
7. salva il modello finale in `model.pth`;
8. salva i nomi delle classi in `class_names.json`.

Alla fine del training dovresti avere questi due file:

```text
model.pth
class_names.json
```

Questi file sono necessari per usare la UI.

---

## Avvio della UI Streamlit

Dopo aver completato il training, avvia l’interfaccia web con:

```bash
streamlit run ui.py
```

Streamlit aprirà automaticamente il browser.

Se il browser non si apre, copia nel browser l’indirizzo mostrato nel terminale, di solito:

```text
http://localhost:8501
```

---

## Come usare la UI

Una volta aperta la pagina Streamlit:

1. carica un’immagine dalla colonna di sinistra;
2. il modello analizza l’immagine;
3. viene mostrata la classe prevista;
4. viene mostrata la percentuale di confidenza;
5. vengono mostrate le prime 3 predizioni;
6. puoi espandere la sezione con tutte le probabilità;
7. puoi visualizzare anche tutte le classi riconosciute dal modello.

---

## File principali

### `train.py`

Serve per addestrare il modello.

Usa:

```bash
python train.py
```

Genera:

```text
model.pth
class_names.json
```

### `ui.py`

Serve per avviare l’interfaccia web.

Usa:

```bash
streamlit run ui.py
```

Carica:

```text
model.pth
class_names.json
```

### `model.pth`

Contiene i pesi del modello addestrato.

Viene creato automaticamente da `train.py`.

### `class_names.json`

Contiene i nomi delle classi del dataset.

Esempio:

```json
[
  "cane",
  "gatto",
  "cavallo"
]
```

Viene creato automaticamente da `train.py`.

---

## Parametri principali del training

Nel file `train.py` puoi modificare questi parametri:

```python
DATASET_DIR = "dataset"
MODEL_OUTPUT_PATH = "model.pth"
CLASS_NAMES_PATH = "class_names.json"

BATCH_SIZE = 16
NUM_EPOCHS = 40
LEARNING_RATE = 1e-4
IMAGE_SIZE = 224
EARLY_STOPPING_PATIENCE = 7
K_FOLDS = 5
RANDOM_STATE = 42
```

Significato:

| Parametro | Descrizione |
|---|---|
| `DATASET_DIR` | Cartella del dataset |
| `MODEL_OUTPUT_PATH` | File in cui viene salvato il modello |
| `CLASS_NAMES_PATH` | File in cui vengono salvati i nomi delle classi |
| `BATCH_SIZE` | Numero di immagini elaborate insieme |
| `NUM_EPOCHS` | Numero massimo di epoche di training |
| `LEARNING_RATE` | Velocità di apprendimento |
| `IMAGE_SIZE` | Dimensione delle immagini in input |
| `EARLY_STOPPING_PATIENCE` | Epoche senza miglioramento prima dello stop |
| `K_FOLDS` | Numero di fold per la cross validation |
| `RANDOM_STATE` | Seed per rendere ripetibile la suddivisione |

---

## Errori comuni

### `File model.pth non trovato`

Hai avviato la UI prima di addestrare il modello.

Soluzione:

```bash
python train.py
```

Poi avvia di nuovo:

```bash
streamlit run ui.py
```

### `File class_names.json non trovato`

Il training non è stato completato oppure il file è stato eliminato.

Soluzione:

```bash
python train.py
```

### Errore sulla cartella `dataset`

Controlla che la cartella `dataset` esista e che contenga sottocartelle, una per ogni classe.

Esempio corretto:

```text
dataset/
├── classe_1/
└── classe_2/
```

### Il modello predice sempre la stessa classe

Possibili cause:

- poche immagini nel dataset;
- classi sbilanciate;
- immagini inserite nelle cartelle sbagliate;
- dataset troppo piccolo;
- immagini troppo simili tra loro;
- training non sufficiente.

Prova ad aumentare il numero di immagini per classe e a bilanciare meglio il dataset.

---

## Comandi principali

Installa le dipendenze:

```bash
pip install -r requirements.txt
```

Avvia il training:

```bash
python train.py
```

Avvia la UI:

```bash
streamlit run ui.py
```

---

## Note importanti

- Se aggiungi nuove classi al dataset, devi rifare il training.
- Se modifichi molto il dataset, è consigliato rifare il training.
- `model.pth` e `class_names.json` devono essere coerenti tra loro.
- La UI funziona solo se `model.pth` e `class_names.json` sono presenti nella cartella principale del progetto.
- Il modello può classificare solo le classi su cui è stato addestrato.

---

## `.gitignore` consigliato

Puoi creare un file `.gitignore` con questo contenuto:

```gitignore
__pycache__/
*.pyc
venv/
.env/
.DS_Store

dataset/
model.pth
class_names.json
```

Se vuoi caricare su GitHub anche il modello già addestrato, rimuovi queste righe dal `.gitignore`:

```gitignore
model.pth
class_names.json
```

Se vuoi caricare anche il dataset, rimuovi:

```gitignore
dataset/
```

---

## Flusso completo di utilizzo

1. Clona il repository.
2. Crea un ambiente virtuale.
3. Installa le dipendenze con `pip install -r requirements.txt`.
4. Crea la cartella `dataset`.
5. Inserisci una sottocartella per ogni classe.
6. Inserisci le immagini nelle rispettive cartelle.
7. Avvia il training con `python train.py`.
8. Attendi la creazione di `model.pth` e `class_names.json`.
9. Avvia la UI con `streamlit run ui.py`.
10. Carica un’immagine e visualizza la predizione.
