# Bussiness Understanding
## Problem Statements
- Bagaimana mengetahui faktor-faktor yang dapat mempengaruhi popularitas lagu di platform musik seperti Spotify?
- Bagaimana menghasilkan performa model yang seakurat mungkin untuk memprediksi jumlah streaming lagu?
## Goals
- Untuk mengetahui faktor-faktor yang mempengaruhi popularitas lagu, seperti genre, akustik, dan elemen musik lainnya.
- Menghasilkan model machine learning yang dapat memprediksi jumlah streaming lagu dengan seakurat mungkin, yang dapat digunakan sebagai acuan dalam strategi pemasaran dan pengembangan lagu.
## Solution Statements
Untuk mengetahui faktor-faktor yang mempengaruhi popularitas lagu, dilakukan langkah-langkah berikut:
- Melakukan Exploratory Data Analysis (EDA) untuk mendapatkan wawasan informasi yang dibutuhkan. Mencakup visualisasi data fitur terhadap target dalam hal ini, fitur-fitur yang terdapat pada dataset dan jumlah streaming sebagai data target.
- Menggunakan feature importance untuk menilai seberapa berguna fitur dalam memprediksi jumlah streaming.
- Memberikan wawasan tentang data dan membantu dalam pengurangan dimensi serta pemilihan fitur yang meningkatkan efisiensi model.

Untuk menghasilkan model machine learning yang seakurat mungkin dalam memprediksi jumlah streaming, tiga algoritma regresi dapat digunakan:
- Linear Regression: Model statistik sederhana yang digunakan untuk memprediksi dengan menemukan garis terbaik yang memisahkan variabel independen dan dependen. Kelebihannya adalah mudah diimplementasikan, tetapi dapat overfit dan tidak cocok untuk hubungan non-linier.
- Polynomial Regression: Metode yang mencari hubungan non-linier antara variabel, dengan menggunakan fungsi kuadratik. Jika model linear tidak cukup fit, maka pendekatan polinomial bisa dicoba untuk menangkap pola yang lebih kompleks dalam data.
- Random Forest: Algoritma supervised learning yang menggabungkan beberapa pohon keputusan untuk meningkatkan akurasi dan mengurangi overfitting. Kelebihan dari Random Forest termasuk kemampuan untuk menangani dataset besar dan memberikan estimasi pentingnya variabel, tetapi waktu pemrosesan bisa lebih lama dan interpretasinya lebih sulit.

Link Streamlit Deploy : https://spotify-streams.streamlit.app/
