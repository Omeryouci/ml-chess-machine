Sıfırdan öğrenen, kendi kendine oynayarak gelişen bir satranç makinesi.
Hiçbir hazır motor, hiçbir hazır açılış kitabı, hiçbir insan satranç bilgisi — sadece bir sinir ağı, bir arama motoru ve seninle oynadıkça öğrenen bir beyin.

📖 İçindekiler
Özellikler

Ekran Görüntüleri

Kurulum

Kullanım

Nasıl Çalışır?

Mimari

Seviye Sistemi

Dosya Yapısı

Testler

Yol Haritası

Lisans

✨ Özellikler
🧠 Yapay Zekâ
Sıfırdan sinir ağı — 773 → 256 → 64 → 1, ReLU + tanh aktivasyonu

Adam optimizer — toplu (batch) vektörize eğitim

Seyrek (sparse) ileri yayılım — sadece dolu karelerin ağırlık satırları toplanır, ~%25 hız artışı

Hibrit değerlendirme — klasik materyal + konum tabloları ile öğrenilmiş sinir ağını harmanlar; ağ olgunlaştıkça (%25.000 adım) tamamen kendi değerlendirmesine geçer

Replay buffer — 30.000 pozisyona kadar geçmiş deneyim havuzu

TD-leaf benzeri hedefler — maç sonucu + arama skoru karışımı

🔍 Arama Motoru
Negamax + alpha-beta ile iteratif derinleştirme

Transposition Table — Zobrist anahtarlı, 400.000 giriş kapasiteli

Quiescence Search — değişim fırtınalarını yakalar, ufuk etkisini azaltır

Killer moves + History heuristic + MVV-LVA hamle sıralaması

Değerlendirme önbelleği — Zobrist anahtarlı, aynı yaprak pozisyonlar tekrar hesaplanmaz

Zaman kontrolü — hamle başına düşünme süresi kullanıcı tarafından seçilir

🎮 Arayüz
Modern kart tasarımı (Catppuccin Mocha temelli koyu tema)

Görsel satranç taşları — Segoe UI Symbol / DejaVu / Symbola sistem fontları üzerinden ♟♞♝♜♛♚

Sürükle-bırak değil, tıkla-tıkla — fare ile hamle yapımı

Yasal hamle hedefleri — nokta (boş kare) / halka (alma)

Şah uyarısı — kırmızı vurgu

Piyon terfi seçimi — vezir / kale / fil / at arasından görsel seçim

Değerlendirme çubuğu — canlı pozisyon değerlendirmesi

Geri alma — yanlış hamleyi tek tuşla geri al

Hamle listesi — fare tekerleği ile kaydırılabilir, SAN notasyonu

Seviye göstergesi — eğitim adımına göre otomatik seviye isimleri

Loss grafiği — öğrenme eğrisi canlı olarak izlenebilir

Ağırlık ısı haritası — sinir ağının ne öğrendiğini görselleştirir

Geçmiş sekmesi — oynanan tüm oyunlar, sonuç, öğrenme hatası

🎯 Eğitim
Self-play eğitimi — makine kendi kendine oynayarak gelişir

3 hız modu — HIZLI (derinlik 2) / NORMAL (derinlik 3) / DERİN (derinlik 4)

Canlı istatistikler — beyaz/siyah/beraberlik, ortalama oyun uzunluğu, son loss

İnsan oyunundan öğrenme — her oyun sonunda otomatik

📸 Ekran Görüntüleri
Ekran görüntülerini oluşturmak için:

bash
python satranc_machine.py --shots screenshots
shots/ klasöründe 7 PNG üretilir:

Dosya	İçerik
1_menu.png	Ana menü
2_game.png	Oyun ekranı (orta oyun)
3_game_selected.png	Seçili taş + yasal hedefler
4_train.png	Self-play eğitim ekranı
5_view0-3.png	İnceleme sekmeleri (Durum / Ağırlıklar / Geçmiş / Son Oyun)
6_gameover.png	Oyun sonu kartı
7_promotion.png	Piyon terfi seçim kartı
🚀 Kurulum
Gereksinimler
Python 3.8+

Pygame 2.0+ (veya pygame-ce)

Adımlar
bash
# 1) Depoyu klonla
git clone https://github.com/kullanici-adi/satranc-machine.git
cd satranc-machine

# 2) Sanal ortam oluştur (önerilir)
python -m venv venv
source venv/bin/activate    # Linux / macOS
# venv\Scripts\activate     # Windows

# 3) Bağımlılıkları kur
pip install chess numpy pygame-ce
# veya:  pip install chess numpy pygame
Not: pygame-ce (Community Edition), pygame'a göre daha hızlı ve daha aktif geliştirilen bir çataldır. İkisi de çalışır.

🎮 Kullanım
Oyun oyna
bash
python satranc_machine.py
Ana menüden:

BEYAZ OYNA — sen beyaz, makine siyah

SİYAH OYNA — makine beyaz, sen siyah

KENDİ KENDİNE ÖĞREN — self-play eğitimi başlat

MAKİNEYİ İNCELE — istatistik + ağırlık görselleştirmesi

Düşünme süresi her hamle için 2s / 5s / 20s / 60s arasından seçilir.

Klavye kısayolları
Tuş	İşlev
F11	Tam ekran
ESC	Çıkış / menüye dön
1-4	İnceleme sekmeleri
↑ / ↓	Geçmiş listesi kaydırma
Fare tekerleği	Hamle listesi kaydırma
Testleri çalıştır
bash
python satranc_machine.py --test
20+ test ile tüm bileşenleri kontrol eder: sinir ağı öğrenmesi, mat bulma, pat, en passant, TT belirleyiciliği, seyrek=yoğun eşitliği, hibrit değerlendirme, terfi yasallığı, geçmiş sınırı, self-play döngüsü...

Ekran görüntüsü üret
bash
python satranc_machine.py --shots screenshots
🔬 Nasıl Çalışır?
1. Başlangıç: Boş Bir Beyin
Makine ilk açıldığında sinir ağının ağırlıkları rastgele başlar. Yani hiçbir satranç bilgisi yoktur — sadece hibrit değerlendirme sayesinde klasik materyal + konum tablolarıyla mantıklı hamleler yapar.

2. Öğrenme Döngüsü
text
Oyun oynanır
   ↓
Her pozisyon için (FEN, arama skoru) toplanır
   ↓
Maç sonucu ile birleştirilir:  target = 0.5 * search_score + 0.5 * outcome
   ↓
Geçmiş tamponuna eklenir (30.000 pozisyon)
   ↓
Rastgele 64'lük batch'lerle sinir ağı eğitilir (Adam)
   ↓
Ağırlıklar diske kaydedilir
3. Hibrit Değerlendirme
Makine olgunlaştıkça klasik değerlendirmeden kendi öğrendiğine geçer:

text
Eğitim adımı 0       → %100 klasik
Eğitim adımı 25.000  → %0 klasik + %100 öğrenilmiş
Formül:

python
w = min(1.0, training_steps / 25000)
value = (1 - w) * classical_eval(board) + w * neural_net(board)
Bu sayede makine hiç öğrenmemiş olsa bile satranç oynayabilir, ama öğrendikçe kendi tarzını geliştirir.

🏗️ Mimari
text
┌─────────────────────────────────────────────────────────┐
│                       Brain                              │
│  ┌───────────────┐    ┌────────────────────────────┐    │
│  │ NeuralNetwork │    │ Replay Buffer (30k FEN)    │    │
│  │ 773→256→64→1  │◄───┤ train_from_buffer()        │    │
│  │  Adam + ReLU  │    └────────────────────────────┘    │
│  └───────┬───────┘                                       │
│          │ forward()                                     │
│          ▼                                               │
│  ┌────────────────────────────────────────────┐          │
│  │ Hibrit Değerlendirme (klasik + öğrenilmiş) │          │
│  └────────────────────┬───────────────────────┘          │
└───────────────────────┼──────────────────────────────────┘
                        │ eval_stm()
                        ▼
┌─────────────────────────────────────────────────────────┐
│                       Engine                             │
│  Negamax + αβ + TT + Quiescence + Killer/History        │
│  İteratif derinleştirme + Zaman kontrolü                 │
└────────────────────┬────────────────────────────────────┘
                     │ en iyi hamle
                     ▼
┌─────────────────────────────────────────────────────────┐
│                    Pygame Arayüzü                        │
│  Tahta · Kartlar · Grafikler · Terfi · Oyun Sonu        │
└─────────────────────────────────────────────────────────┘
Sinir Ağı Girdisi (773 boyut)
Aralık	İçerik
0–63	Beyaz piyonlar (64 kare)
64–127	Beyaz atlar
128–191	Beyaz filler
192–255	Beyaz kaleler
256–319	Beyaz vezirler
320–383	Beyaz şahlar
384–447	Siyah piyonlar
...	...
768	Sıra kimde (+1 beyaz, -1 siyah)
769–772	Rok hakları (4 bit)
Arama Motoru Parametreleri
Parametre	Değer
Maks. derinlik	12
TT kapasitesi	400.000
Eval cache	200.000
Killer moves	2 / katman
Zaman aşımı kontrolü	Her 2048 düğümde
📊 Seviye Sistemi
Eğitim Adımı	Seviye
0	Ham Makine
300	İlk Sinyaller
1.500	Kalibre Ediliyor
4.000	Öğrenen Makine
10.000	Deneyimli Makine
25.000	Gelişmiş Model
60.000	Güçlü Motor
150.000	Ustalaşan Makine
400.000	Elit Model
1.000.000	Süper Makine
📁 Dosya Yapısı
text
satranc-machine/
├── satranc_machine.py        # Ana kod (tek dosya, ~1600 satır)
├── README.md                 # Bu dosya
├── LICENSE                   # MIT lisansı
├── .gitignore                # data/, __pycache__, shots/ hariç tut
└── data/                     # Çalışma zamanında oluşur
    ├── machine_v5.npz        # Sinir ağı ağırlıkları + Adam durumu
    ├── machine_v5_stats.json # İstatistikler (oyun sayısı, loss geçmişi)
    └── history_v5.json       # Oynanan oyunlar (en fazla 500 kayıt)
data/ klasörü .gitignore'da — model dosyaları repoya commit edilmez. Yeni klonlayanlar sıfırdan başlar.

🧪 Testler
--test bayrağı 20+ kontrol çalıştırır:

Kategori	Test
Sinir ağı	Loss düşüyor mu?
Seyrek forward = yoğun forward?
Kaydet/yükle aynı ağırlık ve Adam durumu?
Arama	1 hamlede mat bulunuyor mu? (Qh4#, Rh8#)
Pat pozisyonunda None dönüyor mu?
Yetersiz materyalde skor 0 mı?
En passant pozisyonunda çalışıyor mu?
TT/önbellek belirleyici mi?
Değerlendirme	Klasik: başlangıç dengeli (0)
Klasik: vezir avantajını görür (+900)
Hibrit: hiç öğrenmemiş → tamamen klasik
Hibrit: olgun → tamamen öğrenilmiş
Oyun	Terfi: tüm taşlar yasal
Terfi tetikleyici doğru mu?
Altyapı	Geçmiş 500 kayıtta sınırlanıyor mu?
make_targets formülü doğru mu?
Self-play + öğrenme tamamlanıyor mu?
İnsan oyunundan öğrenme çalışıyor mu?
Örnek çıktı:

text
[PASS] Sinir agi ogreniyor (loss dustu)
[PASS] 1 hamlede mat bulundu (Qh4#)
[PASS] Seyrek forward = yogun forward (fark 3.4e-07)
[PASS] Kale ile 1 hamlede mat (Rh8#)
[PASS] Hibrit: olgun makine tamamen kendi degerlendirmesi
...

🤝 Katkı
Pull request'ler memnuniyetle karşılanır. Büyük değişiklikler için lütfen önce bir issue açıp tartışalım.

bash
git checkout -b ozellik/yeni-arama
# değişikliklerini yap
python satranc_machine.py --test   # testleri geç
git commit -m "feat: yeni arama özelliği"
git push origin ozellik/yeni-arama
