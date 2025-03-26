import math
from collections import Counter

# Term Frequency (TF) ni hisoblash uchun funksiya
def compute_tf(sozlar):
    tf_counter = Counter(sozlar)
    tf = {soz: count / len(sozlar) for soz, count in tf_counter.items()}
    return tf

# Document Frequency (DF) ni hisoblash uchun funksiya
def compute_idf(hujjatlar):
    N = len(hujjatlar)
    idf_values = {}
    
    # Har bir so'z uchun hujjatlardagi chastotani hisoblash
    for hujjat in hujjatlar:
        for soz in set(hujjat):  # set() bilan takrorlanishlardan qochamiz
            idf_values[soz] = idf_values.get(soz, 0) + 1

    # IDF formulasi bo'yicha qiymatlarni hisoblash
    for soz, df in idf_values.items():
        idf_values[soz] = math.log(N / (1 + df))  # IDF formulasi

    return idf_values

# TF-IDF ni hisoblash uchun funksiya
def compute_tf_idf(tf, idf, target_word):
    if target_word in tf and target_word in idf:
        tf_idf_value = tf[target_word] * idf[target_word]
        return tf_idf_value
    else:
        return 0.0  # Agar so'z mavjud bo'lmasa, 0 qaytaradi

# Hujjatlar to'plamini yaratamiz
hujjatlar = [
    "Bu birinchi hujjat",
    "Bu ikkinchi hujjat",
    "Bu uchinchi hujjat takrorlanadi va takrorlanadi"
]

# Hujjatlardagi so'zlarni bo'sh joylar bo'yicha ajratamiz
hujjatlar_ajratilgan = [hujjat.lower().split() for hujjat in hujjatlar]

# Har bir hujjat uchun IDF qiymatini hisoblaymiz
idf = compute_idf(hujjatlar_ajratilgan)

# So'zni aniqlash
target_word = "takrorlanadi"

# Hujjatlar bo'yicha ma'lumotni chiqaramiz
for idx, hujjat in enumerate(hujjatlar_ajratilgan):
    print(f"Hujjat {idx+1}:")
    tf = compute_tf(hujjat)
    tf_idf_value = compute_tf_idf(tf, idf, target_word)
    print(f"'{target_word}' so'zining TF-IDF qiymati: {tf_idf_value}\n")
