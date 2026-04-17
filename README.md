<div align="right" dir="rtl">

# 🎨 محلّل الصور الأكاديمي المتقدّم

**تحليل تقني شامل للخصائص الفنية والجماليّة للصور**

---

### بحث طالبة الدكتوراه

**عبير قاسم حلف**
الجامعة المستنصرية &nbsp;•&nbsp; كلية التربية الأساسية

</div>

---

## 🖼️ Overview

An academic-grade image analyzer built in Python + Streamlit that examines an image through **nine analytical stages** — from raw pixel metrics to semiotic interpretation. Designed as a research instrument for Fine Arts thesis work: it pairs the precision of computer vision with the interpretive vocabulary of visual culture.

<div align="right" dir="rtl">

تطبيقٌ لتحليل الصور بصورة أكاديميّة، يُمرّر كلّ صورةٍ عبر تسع مراحلَ متسلسلة تبدأ بالقياسات الرقميّة وتنتهي بتفسيرٍ سيميائيّ للعناصر الرمزيّة.

</div>

---

## 🧭 Pipeline — Nine Stages

| # | Stage | Arabic | Algorithms |
|---|-------|--------|------------|
| 01 | Upload | رفع الصورة | PIL decode → NumPy matrix |
| 02 | Basic Metrics | القياسات الأساسيّة | dimensions · aspect ratio · resolution class |
| 03 | Color Analysis | تحليل الألوان | **K-Means** (k=5) · HSV statistics · color temperature |
| 04 | Lighting | الإضاءة | histogram statistics · exposure balance |
| 05 | Composition | التكوين الفني | symmetry · rule of thirds · entropy |
| 06 | Quality | الجودة | **Laplacian variance** · **Sobel gradient** · noise estimate |
| 07 | Typography (OCR) | الخطوط والنصوص | **MSER** + **EasyOCR** (ar · en) |
| 08 | Semiotic Analysis | التحليل السيميائي | **Hough** transforms · contour polygons · saliency |
| 09 | Report | التقرير | aggregate · format · downloadable .txt |

A full visual diagram is included as [`system_diagram_figma.png`](system_diagram_figma.png) and [`system_diagram.pdf`](system_diagram.pdf).

---

## 🚀 Quick Start

```bash
# 1. clone the repo
git clone https://github.com/alihusseinfadel/academic-image-analyzer.git
cd academic-image-analyzer

# 2. create a virtual environment (recommended)
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS / Linux:
source .venv/bin/activate

# 3. install dependencies
pip install -r requirements.txt

# 4. run the app
streamlit run image_analyzer_app_1.py
```

The app opens automatically at `http://localhost:8501`.

---

## 📦 Project Structure

```
academic-image-analyzer/
├── image_analyzer_app_1.py    ← main Streamlit application
├── image_analyzer_app.py      ← earlier revision (kept for reference)
├── requirements.txt
├── README.md
├── شرح_النظام.docx             ← academic explanation (Arabic thesis appendix)
├── design_philosophy.md
├── system_diagram.png          ← manuscript-style plate
├── system_diagram.pdf
├── system_diagram_figma.svg    ← editable vector for Figma
├── system_diagram_figma.png
├── simple_flow.svg
├── simple_flow.png
├── generate_diagram.py         ← scripts that reproduce the diagrams
├── generate_figma_diagram.py
├── render_figma_png.py
├── simple_flow.py
└── generate_word_doc.py
```

---

## 🛠️ Built With

- **[Streamlit](https://streamlit.io/)** — interactive web UI
- **[OpenCV](https://opencv.org/)** — core computer-vision operations
- **[Pillow / PIL](https://python-pillow.org/)** — image I/O
- **[NumPy](https://numpy.org/) & [SciPy](https://scipy.org/)** — numerical computing
- **[scikit-image](https://scikit-image.org/)** — filters, entropy, morphology
- **[EasyOCR](https://github.com/JaidedAI/EasyOCR)** — Arabic & English text recognition
- **[pandas](https://pandas.pydata.org/)** — report tabulation
- **[arabic-reshaper](https://github.com/mpcabd/python-arabic-reshaper) + [python-bidi](https://github.com/MeirKriheli/python-bidi)** — correct Arabic rendering in diagrams

---

## 🎓 Academic Context

<div align="right" dir="rtl">

أُعدّت هذه الأداة ضمن بحث الدكتوراه في **كلية التربية الأساسية – الجامعة المستنصرية**، للباحثة **عبير قاسم حلف**. تهدف إلى تقديم قراءة مزدوجة للصورة: قراءةٌ رقميّةٌ دقيقةٌ من جهة، وقراءةٌ تأويليّةٌ سيميائيّة من جهةٍ أخرى — بما يخدم الدراسات الجماليّة والفنون البصريّة.

</div>

This tool was developed as part of PhD research at the **College of Basic Education – Al-Mustansiriya University**. It offers a dual reading of each image: a precise quantitative reading, alongside an interpretive semiotic reading — serving aesthetic studies and visual-arts research.

---

## 📄 License

Released under the MIT License — see [`LICENSE`](LICENSE) for details.

---

<div align="center">
<sub>Built with patience, care, and the tradition of careful looking.</sub>
</div>
