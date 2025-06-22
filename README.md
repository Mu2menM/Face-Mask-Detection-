# Face Mask Detection

Ein sauberes, nachvollziehbares Machine-Learning-Projekt zur Bildklassifikation: Es erkennt, ob Personen auf Bildern **eine Maske korrekt tragen** oder **keine Maske tragen**. Der Fokus liegt auf Clean Code, modularer Pipeline-Struktur und Verständlichkeit – ideal für Machine-Learning-Einsteiger\:innen.

---

## Motivation

Die automatische Erkennung von Maskenträgern spielt eine wichtige Rolle in öffentlichen Einrichtungen, Gesundheitssystemen und im Kontext von Pandemien. Dieses Projekt demonstriert, wie man mit modernen ML-Werkzeugen ein robustes, aber leicht verständliches Klassifikationsmodell aufbaut.

---

## Projekt-Features

* Automatischer Download und Strukturierung eines Face-Mask-Datensatzes (via Kaggle)
* Vorverarbeitung und Visualisierung von Bildern
* Ein eigenes CNN-Modell mit Hyperparameter-Tuning
* Ein MobileNetV2 Modell mit Hyperparameter-Tuning
* Vergleich mehrerer Modellvarianten
* Auswertung mit Accuracy, Precision, Recall, F1-Score
* Visualisierung von Trainingsverläufen

---

## Setup

### Voraussetzungen

* Python 3.10+
* Miniconda-Umgebung empfohlen

### Installation

```bash
conda create -n maskenv python=3.10
conda activate maskenv
pip install -r requirements.txt
```

### Kaggle-API-Key einrichten

1. `kaggle.json` herunterladen ([https://www.kaggle.com/settings](https://www.kaggle.com/settings))
2. Im Projektordner ablegen unter: `notebooks/kaggle.json`

---

## Nutzung

### 1. Notebook starten

Das Projekt enthält ein Jupyter Notebook mit folgendem typischen Ablauf:

1. **Daten herunterladen**: Das Skript lädt den Datensatz automatisch via `kagglehub.dataset_download(...)`, sofern `Train/` noch nicht existiert.
2. **Daten visualisieren**: Beispielbilder werden mit `matplotlib` dargestellt.
3. **Daten augmentieren**: `keras.Sequential` mit zufälliger Flip-, Zoom-, Rotations-Transformation.
4. **Modelle bauen**: Über `build_cnn(...)` und `build_mobilenet(...)` werden Varianten parametrisiert erstellt.
5. **Training**: Sechs Modellvarianten (`A_small`, `B_medium`, `C_large`, `D_mobilenet_small`, `E_mobilenet_medium`, `F_mobilenet_large`) werden über eine Schleife trainiert.
6. **Evaluation**: Beste Varianten von CNN und MobileNet wird auf Testdaten ausgewertet.
7. **Visualisierung**: Accuracy über Epochen wird geplottet.

Optional kannst du auch nur eine einzelne Variante trainieren, z. B.:

```python
params = VARIANTS["A_small"]
model = build_cnn(**params)
model.fit(...)
evaluate(model, test_ds)
```

---

## Ordnerstruktur

```
Face-Mask-Detection/
├── data/                    # Enthält Train/Validation/Test nach Download
│   ├── Train/
│   ├── Validation/
│   └── Test/
├── notebooks/
│   ├── 01_face_mask_cnn.ipynb
│   └── kaggle.json          # API-Key hier ablegen
├── README.md
├── requirements.txt
└── ...
```

---

## Modellbeschreibung

### Architektur (Standardvariante "A\_small")

* Eingabe: RGB-Bild (128×128×3)
* 2× Conv2D + MaxPooling
* Flatten + Dense
* Output: Dense Softmax mit 2 Klassen

### Parameter

* Loss: `sparse_categorical_crossentropy`
* Optimizer: `Adam`
* Metrik: `accuracy`

Hyperparameter (über `VARIANTS` definiert):

* Anzahl Convolution-Blöcke
* Anzahl Dense-Neuronen
* Lernrate
* Anzahl Epochen

---

## Evaluation

* Genutzt wird `sklearn.metrics.classification_report` und `confusion_matrix`
* Vergleich der `val_accuracy` zur Modellwahl

---

---

## Lizenz & Credits

Datensatz: © Kaggle – [ashishjangra27/face-mask-12k-images-dataset](https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset)

Projekt im Rahmen des AI-Engineering-Kurses FH Campus Wien, Masterstudium Software Engineering.

---

## Autoren

* \Bondok Mohamed
* \El-shaarawi Mazen
* \Murad Mumen 




