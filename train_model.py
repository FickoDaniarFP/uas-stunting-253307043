import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# 1. Muat dataset
df = pd.read_csv("data_stunting.csv")

# 2. Preprocessing fitur & target biner (Normal vs Stunting)
df["jk_num"] = df["Jenis Kelamin"].map({"laki-laki": 1, "perempuan": 0})
df["target"] = df["Status Gizi"].apply(
    lambda x: "Normal" if x in ["normal", "tinggi"] else "Stunting"
)

X = df[["jk_num", "Umur (bulan)", "Tinggi Badan (cm)"]]
y = df["target"]

# 3. Bagi data latih (80%) dan uji (20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Latih model Decision Tree
model = DecisionTreeClassifier(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluasi & simpan paket model
paket = {
    "model": model,
    "fitur": ["Jenis Kelamin", "Umur (bulan)", "Tinggi Badan (cm)"],
    "kelas": ["Normal", "Stunting"],
    "akurasi_latih": accuracy_score(y_train, model.predict(X_train)),
    "akurasi_uji": accuracy_score(y_test, model.predict(X_test)),
    "confusion_matrix": confusion_matrix(
        y_test, model.predict(X_test), labels=["Normal", "Stunting"]
    ),
}
joblib.dump(paket, "model_stunting.joblib")
print("Sukses! File model_stunting.joblib berhasil dibuat.")