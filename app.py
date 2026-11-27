import base64
import io
import os
import sys

import django
from django.conf import settings
from django.core.wsgi import get_wsgi_application

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 1. KONFIGURASI DJANGO
if not settings.configured:
    settings.configure(
        DEBUG=True,
        SECRET_KEY="uas-sistem-cerdas-ficko-253307043",
        ROOT_URLCONF=__name__,
        ALLOWED_HOSTS=["*"],
        STATIC_URL="/static/",
        INSTALLED_APPS=[
            "django.contrib.contenttypes",
            "django.contrib.staticfiles",
        ],
        MIDDLEWARE=[
            "django.middleware.common.CommonMiddleware",
            "django.middleware.csrf.CsrfViewMiddleware",
        ],
        TEMPLATES=[
            {
                "BACKEND": "django.template.backends.django.DjangoTemplates",
                "DIRS": [],
                "APP_DIRS": False,
                "OPTIONS": {
                    "context_processors": [
                        "django.template.context_processors.request",
                    ],
                },
            }
        ],
    )
    django.setup()

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from django.core.management import execute_from_command_line
from django.http import HttpResponse
from django.template import engines
from django.urls import path
from sklearn.tree import plot_tree

# 2. MUAT MODEL & BUAT GAMBAR POHON KEPUTUSAN SEKALI SAAT SERVER NYALA
paket = joblib.load(os.path.join(BASE_DIR, "model_stunting.joblib"))
model = paket["model"]

fig, ax = plt.subplots(figsize=(20, 10), dpi=200)
plot_tree(
    model,
    feature_names=paket["fitur"],
    class_names=paket["kelas"],
    filled=True,
    rounded=True,
    max_depth=3,
    fontsize=9,
    ax=ax,
)
plt.tight_layout()
buf = io.BytesIO()
fig.savefig(buf, format="png")
plt.close(fig)
POHON_BASE64 = base64.b64encode(buf.getvalue()).decode("utf-8")

# 3. TEMPLATE HTML DENGAN UI MODERN & INTERAKTIF
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deteksi Status Gizi Balita | UAS Sistem Cerdas</title>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css" rel="stylesheet">
    <style>
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background: linear-gradient(180deg, #eef6f7 0%, #f8fafc 100%);
            color: #1e293b;
        }
        .hero-banner {
            background: linear-gradient(135deg, #0f766e 0%, #0d9488 100%);
            color: white;
            border-radius: 16px;
            padding: 1.75rem 2rem;
            box-shadow: 0 10px 25px -5px rgba(13, 148, 136, 0.25);
        }
        .card-modern {
            border: none;
            border-radius: 16px;
            box-shadow: 0 4px 20px rgba(15, 23, 42, 0.05);
            background: #ffffff;
        }
        .acuan-box {
            background-color: #f0fdfa;
            border-left: 4px solid #0d9488;
            border-radius: 8px;
            padding: 0.75rem 1rem;
            font-size: 0.9rem;
            color: #115e59;
        }
        .btn-teal {
            background-color: #0f766e;
            border-color: #0f766e;
            color: white;
            font-weight: 600;
            padding: 0.6rem 1.4rem;
            border-radius: 10px;
            transition: all 0.2s;
        }
        .btn-teal:hover {
            background-color: #115e59;
            border-color: #115e59;
            color: white;
            transform: translateY(-1px);
        }
        .metric-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 1rem;
        }
        details.custom-details {
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 0.85rem 1.1rem;
            background: #ffffff;
            transition: all 0.2s;
        }
        details.custom-details[open] {
            background: #fcfdfd;
            border-color: #99f6e4;
        }
        details.custom-details summary {
            cursor: pointer;
            list-style: none;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-weight: 600;
            color: #0f172a;
        }
        details.custom-details summary::-webkit-details-marker {
            display: none;
        }
        .form-range::-webkit-slider-thumb {
            background: #0d9488;
        }
    </style>
</head>
<body>
<div class="container py-4" style="max-width: 880px;">

    <!-- HEADER BANNER -->
    <div class="hero-banner mb-4">
        <div class="d-flex flex-wrap justify-content-between align-items-center gap-2 mb-2">
            <span class="badge bg-white text-teal fw-semibold px-3 py-2 rounded-pill" style="color: #0f766e;">
                <i class="bi bi-cpu-fill me-1"></i> UAS Sistem Cerdas • Decision Tree
            </span>
            <span class="small opacity-75">Standar Antropometri Permenkes RI No. 2/2020</span>
        </div>
        <h2 class="fw-bold mb-1"><i class="bi bi-heart-pulse-fill me-2"></i>Klasifikasi Status Gizi (Stunting) Balita</h2>
        <p class="mb-0 opacity-90 small">Ficko Daniar Fachryza Putra &bull; NIM: 253307043 &bull; D3 Teknologi Informasi PNM</p>
    </div>

    <!-- BAGIAN 1: FORM INPUT -->
    <div class="card card-modern mb-4">
        <div class="card-body p-4">
            <h5 class="fw-bold mb-3 d-flex align-items-center">
                <span class="badge rounded-circle me-2" style="background:#0d9488;">1</span>
                Input Data Pemeriksaan Balita
            </h5>
            <form method="POST">
                {% csrf_token %}
                <div class="row g-3 align-items-end">
                    <div class="col-md-4">
                        <label class="form-label fw-semibold small text-secondary">Jenis Kelamin</label>
                        <select name="jenis_kelamin" id="inputJK" class="form-select form-select-lg fs-6" onchange="updateAcuan()">
                            <option value="Laki-laki" {% if jenis_kelamin == "Laki-laki" %}selected{% endif %}>👦 Laki-laki</option>
                            <option value="Perempuan" {% if jenis_kelamin == "Perempuan" %}selected{% endif %}>👧 Perempuan</option>
                        </select>
                    </div>
                    <div class="col-md-4">
                        <div class="d-flex justify-content-between">
                            <label class="form-label fw-semibold small text-secondary">Umur Balita</label>
                            <span class="badge bg-teal-subtle text-dark border fw-bold" id="lblUmur">{{ umur_bulan }} bulan</span>
                        </div>
                        <input type="range" name="umur_bulan" id="inputUmur" class="form-range my-2" min="0" max="60" value="{{ umur_bulan }}" oninput="updateAcuan()">
                    </div>
                    <div class="col-md-4">
                        <label class="form-label fw-semibold small text-secondary">Tinggi / Panjang Badan (cm)</label>
                        <div class="input-group">
                            <input type="number" name="tinggi_badan" class="form-control form-control-lg fs-6" min="45" max="125" step="0.5" value="{{ tinggi_badan }}" required>
                            <span class="input-group-text bg-light text-muted">cm</span>
                        </div>
                    </div>
                </div>

                <!-- INFO ACUAN REALTIME -->
                <div class="acuan-box mt-3 d-flex align-items-center gap-2">
                    <i class="bi bi-info-circle-fill fs-5"></i>
                    <div id="teksAcuan">
                        Acuan Kemenkes RI (<strong>{{ jenis_kelamin }}</strong>, <strong>{{ umur_bulan }} bln</strong>):
                        Ambang batas bawah normal sekitar <strong>{{ batas_stunting }} cm</strong>.
                    </div>
                </div>

                <div class="mt-3">
                    <button type="submit" class="btn btn-teal">
                        <i class="bi bi-search-heart me-1"></i> Prediksi Status Gizi
                    </button>
                </div>
            </form>

            <!-- HASIL PREDIKSI -->
            {% if hasil %}
                <div class="alert {% if hasil == 'Normal' %}alert-success border-success{% else %}alert-warning border-warning{% endif %} mt-4 mb-0 rounded-3 shadow-sm">
                    <div class="d-flex align-items-center gap-3">
                        <i class="bi {% if hasil == 'Normal' %}bi-check-circle-fill text-success{% else %}bi-exclamation-triangle-fill text-warning{% endif %} fs-2"></i>
                        <div>
                            <div class="small text-uppercase fw-semibold opacity-75">Hasil Klasifikasi Model ({{ jenis_kelamin }}, {{ umur_bulan }} bln, {{ tinggi_badan }} cm)</div>
                            <h5 class="fw-bold mb-0">
                                Status Gizi: {{ hasil|upper }}
                                <span class="fw-normal fs-6">
                                    {% if hasil == 'Normal' %}— Pertumbuhan tinggi badan sesuai standar usia.{% else %}— Terindikasi stunting, disarankan konsultasi ke Posyandu/Puskesmas.{% endif %}
                                </span>
                            </h5>
                        </div>
                    </div>
                </div>
            {% endif %}
        </div>
    </div>

    <!-- BAGIAN 2: EVALUASI MODEL -->
    <div class="card card-modern mb-4">
        <div class="card-body p-4">
            <h5 class="fw-bold mb-3 d-flex align-items-center">
                <span class="badge rounded-circle me-2" style="background:#0d9488;">2</span>
                Evaluasi Model & Aturan Pohon Keputusan
            </h5>

            <div class="row g-3 text-center mb-3">
                <div class="col-6">
                    <div class="metric-card">
                        <small class="text-muted fw-semibold d-block mb-1"><i class="bi bi-database-check me-1"></i>Akurasi Data Latih (80%)</small>
                        <h3 class="fw-bold mb-0" style="color: #0f766e;">{{ akurasi_latih }}</h3>
                    </div>
                </div>
                <div class="col-6">
                    <div class="metric-card">
                        <small class="text-muted fw-semibold d-block mb-1"><i class="bi bi-graph-up-arrow me-1"></i>Akurasi Data Uji (20%)</small>
                        <h3 class="fw-bold mb-0" style="color: #0f766e;">{{ akurasi_uji }}</h3>
                    </div>
                </div>
            </div>

            <details class="custom-details mb-2">
                <summary>
                    <span><i class="bi bi-diagram-3-fill text-success me-2"></i>Lihat Visualisasi Pohon Keputusan (Decision Tree)</span>
                    <i class="bi bi-chevron-down text-muted"></i>
                </summary>
                <div class="mt-3 text-center">
                    <img src="data:image/png;base64,{{ pohon_img }}" class="img-fluid rounded border" alt="Decision Tree">
                </div>
            </details>

            <details class="custom-details">
                <summary>
                    <span><i class="bi bi-table text-success me-2"></i>Lihat Confusion Matrix & 10 Sampel Dataset</span>
                    <i class="bi bi-chevron-down text-muted"></i>
                </summary>
                <div class="mt-3">
                    <p class="fw-bold small text-uppercase text-muted mb-2">Confusion Matrix (Data Uji)</p>
                    <table class="table table-bordered table-sm text-center align-middle">
                        <thead class="table-light">
                            <tr><th>Aktual \\ Prediksi</th><th>Tebakan: Normal</th><th>Tebakan: Stunting</th></tr>
                        </thead>
                        <tbody>
                            <tr><th class="table-light">Asli: Normal</th><td class="fw-semibold text-success">{{ cm.0.0 }}</td><td class="text-muted">{{ cm.0.1 }}</td></tr>
                            <tr><th class="table-light">Asli: Stunting</th><td class="text-muted">{{ cm.1.0 }}</td><td class="fw-semibold text-warning">{{ cm.1.1 }}</td></tr>
                        </tbody>
                    </table>

                    <p class="fw-bold small text-uppercase text-muted mt-3 mb-2">10 Sampel Acak Dataset Balita</p>
                    <div class="table-responsive rounded border">{{ sampel_tabel|safe }}</div>
                </div>
            </details>
        </div>
    </div>
</div>

<script>
function updateAcuan() {
    const jk = document.getElementById("inputJK").value;
    const umur = parseInt(document.getElementById("inputUmur").value);
    document.getElementById("lblUmur").innerText = umur + " bulan";
    const dasar = (jk === "Laki-laki") ? 46.1 : 45.4;
    const batas = (dasar + (umur * 0.91)).toFixed(1);
    document.getElementById("teksAcuan").innerHTML =
        `Acuan Kemenkes RI (<strong>${jk}</strong>, <strong>${umur} bln</strong>): Ambang batas bawah normal sekitar <strong>${batas} cm</strong>.`;
}
</script>
</body>
</html>
"""


# 4. VIEW LOGIKA PREDIKSI
def index(request):
    hasil = None
    jenis_kelamin = "Laki-laki"
    umur_bulan = 24
    tinggi_badan = 85.0

    if request.method == "POST":
        jenis_kelamin = request.POST.get("jenis_kelamin", "Laki-laki")
        umur_bulan = int(request.POST.get("umur_bulan", 24))
        tinggi_badan = float(request.POST.get("tinggi_badan", 85.0))

        jk_num = 1 if jenis_kelamin == "Laki-laki" else 0
        data_baru = pd.DataFrame(
            [[jk_num, umur_bulan, tinggi_badan]],
            columns=["jk_num", "Umur (bulan)", "Tinggi Badan (cm)"],
        )
        hasil = model.predict(data_baru)[0]

    if jenis_kelamin == "Laki-laki":
        batas_stunting = round(46.1 + (umur_bulan * 0.91), 1)
    else:
        batas_stunting = round(45.4 + (umur_bulan * 0.91), 1)

    sampel_df = pd.read_csv(
        os.path.join(BASE_DIR, "data_stunting.csv")
    ).sample(10, random_state=42)

    context = {
        "hasil": hasil,
        "jenis_kelamin": jenis_kelamin,
        "umur_bulan": umur_bulan,
        "tinggi_badan": tinggi_badan,
        "batas_stunting": batas_stunting,
        "akurasi_latih": f"{paket['akurasi_latih']:.2%}",
        "akurasi_uji": f"{paket['akurasi_uji']:.2%}",
        "cm": paket["confusion_matrix"],
        "pohon_img": POHON_BASE64,
        "sampel_tabel": sampel_df.to_html(
            classes="table table-sm table-striped table-hover mb-0", index=False
        ),
    }

    template = engines["django"].from_string(HTML_TEMPLATE)
    return HttpResponse(template.render(context, request))


# 5. ROUTING URL & EKSEKUSI SERVER
urlpatterns = [
    path("", index, name="index"),
]

application = get_wsgi_application()
app = application

if __name__ == "__main__":
    execute_from_command_line(sys.argv)