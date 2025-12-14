# Heart Disease Classification Using Machine Learning

## Deskripsi
Proyek ini merupakan implementasi kecerdasan buatan menggunakan algoritma Random Forest
untuk mengklasifikasikan risiko penyakit jantung berdasarkan data medis pasien.

## Dataset
Dataset yang digunakan adalah Heart Disease Dataset yang diperoleh dari Kaggle,
dengan total 303 data dan 13 fitur input.

## Tahapan Penelitian
1. Menentukan objektif dan tujuan teknis
2. Menelaah dan memvalidasi data (EDA)
3. Membersihkan dan mengkonstruksi data
4. Membangun skenario dan model
5. Melakukan hyperparameter tuning
6. Evaluasi model menggunakan confusion matrix dan classification report

## Algoritma
- Random Forest Classifier
- GridSearchCV untuk optimasi parameter

## Evaluasi
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

## Cara Menjalankan
```bash
pip install -r requirements.txt
cd src
python eda.py
python preprocessing.py
python train_model.py
python evaluate_model.py
