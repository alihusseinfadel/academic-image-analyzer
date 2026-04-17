import streamlit as st
import cv2
import numpy as np
from PIL import Image
from skimage import filters
import pandas as pd
import re
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    print("⚠️ EasyOCR not installed. Text analysis will be limited.")

# إعدادات الصفحة
st.set_page_config(
    page_title="محلل الصور الأكاديمي",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS مخصص للتصميم العربي
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;900&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Tajawal', sans-serif;
        direction: rtl;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        padding: 1.5rem;
        border-radius: 10px;
        border-right: 5px solid #c59849;
        margin-bottom: 1rem;
    }
    
    .section-header {
        background: #2c5282;
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 2rem;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: 700;
    }
    
    .analysis-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    
    .color-box {
        display: inline-block;
        width: 60px;
        height: 60px;
        border-radius: 8px;
        margin: 5px;
        border: 2px solid #ddd;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #f7fafc 0%, white 100%);
        padding: 1rem;
        border-radius: 8px;
        border: 2px solid #e2e8f0;
    }
</style>
""", unsafe_allow_html=True)


class ImageAnalyzer:
    """فئة تحليل الصور الأكاديمي"""
    
    def __init__(self, image):
        """تهيئة المحلل مع الصورة"""
        self.pil_image = image
        self.cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        self.gray_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2GRAY)
        self.width, self.height = image.size
        self.pixels = self.width * self.height
        
    def get_basic_info(self):
        """الحصول على المعلومات الأساسية"""
        info = {
            'العرض': f'{self.width} بكسل',
            'الارتفاع': f'{self.height} بكسل',
            'نسبة العرض للارتفاع': self._calculate_aspect_ratio(),
            'عدد البكسلات': f'{self.pixels:,}',
            'نوع الصورة': self.pil_image.mode,
            'تصنيف الدقة': self._classify_resolution()
        }
        return info
    
    def _calculate_aspect_ratio(self):
        """حساب نسبة العرض للارتفاع"""
        from math import gcd
        divisor = gcd(self.width, self.height)
        return f'{self.width//divisor}:{self.height//divisor}'
    
    def _classify_resolution(self):
        """تصنيف الدقة"""
        if self.pixels < 500000:
            return '📱 دقة منخفضة (< 0.5 ميغابكسل)'
        elif self.pixels < 2000000:
            return '💻 دقة متوسطة (0.5-2 ميغابكسل)'
        elif self.pixels < 8000000:
            return '📷 دقة عالية HD (2-8 ميغابكسل)'
        elif self.pixels < 20000000:
            return '🎬 دقة عالية جداً Full HD (8-20 ميغابكسل)'
        else:
            return '🎥 دقة فائقة 4K+ (> 20 ميغابكسل)'
    
    def analyze_colors(self):
        """تحليل الألوان المتقدم"""
        # تحويل إلى RGB
        rgb_image = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2RGB)
        pixels = rgb_image.reshape(-1, 3)
        
        # الألوان السائدة باستخدام K-means
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        k = 5
        _, labels, palette = cv2.kmeans(
            pixels.astype(np.float32), k, None, criteria, 10, 
            cv2.KMEANS_RANDOM_CENTERS
        )
        
        # حساب النسب
        _, counts = np.unique(labels, return_counts=True)
        percentages = counts / len(labels) * 100
        
        dominant_colors = []
        for color, percentage in zip(palette, percentages):
            dominant_colors.append({
                'rgb': tuple(color.astype(int)),
                'hex': '#{:02x}{:02x}{:02x}'.format(*color.astype(int)),
                'percentage': percentage
            })
        
        # ترتيب حسب النسبة
        dominant_colors = sorted(dominant_colors, key=lambda x: x['percentage'], reverse=True)
        
        # تحليل التشبع والسطوع
        hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        color_analysis = {
            'dominant_colors': dominant_colors,
            'avg_hue': float(np.mean(h)),
            'avg_saturation': float(np.mean(s)) / 255 * 100,
            'avg_brightness': float(np.mean(v)) / 255 * 100,
            'color_variance': float(np.std(rgb_image)),
            'color_temperature': self._estimate_color_temperature(rgb_image)
        }
        
        return color_analysis
    
    def _estimate_color_temperature(self, rgb_image):
        """تقدير درجة حرارة اللون"""
        r = np.mean(rgb_image[:, :, 0])
        b = np.mean(rgb_image[:, :, 2])
        
        if r > b * 1.2:
            return 'دافئة (Warm) - درجات الأحمر والبرتقالي'
        elif b > r * 1.2:
            return 'باردة (Cool) - درجات الأزرق والأخضر'
        else:
            return 'محايدة (Neutral) - متوازنة'
    
    def analyze_lighting(self):
        """تحليل الإضاءة والتباين"""
        # السطوع
        brightness = np.mean(self.gray_image)
        
        # التباين
        contrast = np.std(self.gray_image)
        
        # الهيستوجرام
        hist = cv2.calcHist([self.gray_image], [0], None, [256], [0, 256])
        
        # التعرض
        dark_pixels = np.sum(self.gray_image < 50) / self.pixels * 100
        bright_pixels = np.sum(self.gray_image > 200) / self.pixels * 100
        midtone_pixels = 100 - dark_pixels - bright_pixels
        
        # Dynamic Range
        dynamic_range = float(np.max(self.gray_image) - np.min(self.gray_image))
        
        lighting = {
            'brightness': brightness,
            'brightness_category': self._categorize_brightness(brightness),
            'contrast': contrast,
            'contrast_category': self._categorize_contrast(contrast),
            'dark_pixels': dark_pixels,
            'bright_pixels': bright_pixels,
            'midtone_pixels': midtone_pixels,
            'dynamic_range': dynamic_range,
            'exposure_balance': self._evaluate_exposure(dark_pixels, bright_pixels)
        }
        
        return lighting
    
    def _categorize_brightness(self, brightness):
        """تصنيف السطوع"""
        if brightness < 60:
            return 'مظلمة جداً (Very Dark)'
        elif brightness < 100:
            return 'مظلمة (Dark)'
        elif brightness < 150:
            return 'متوسطة (Medium)'
        elif brightness < 200:
            return 'مشرقة (Bright)'
        else:
            return 'مشرقة جداً (Very Bright)'
    
    def _categorize_contrast(self, contrast):
        """تصنيف التباين"""
        if contrast < 30:
            return 'منخفض جداً (Very Low)'
        elif contrast < 50:
            return 'منخفض (Low)'
        elif contrast < 70:
            return 'متوسط (Medium)'
        elif contrast < 90:
            return 'عالي (High)'
        else:
            return 'عالي جداً (Very High)'
    
    def _evaluate_exposure(self, dark, bright):
        """تقييم التعرض"""
        if dark > 40:
            return 'تعرض ناقص (Underexposed)'
        elif bright > 40:
            return 'تعرض زائد (Overexposed)'
        else:
            return 'متوازن (Well Balanced)'
    
    def analyze_composition(self):
        """تحليل التكوين والبناء الفني"""
        # كشف الحواف
        edges = cv2.Canny(self.gray_image, 100, 200)
        edge_density = np.sum(edges > 0) / self.pixels * 100
        
        # التماثل الأفقي والعمودي
        horizontal_symmetry = self._calculate_symmetry(self.gray_image, axis=0)
        vertical_symmetry = self._calculate_symmetry(self.gray_image, axis=1)
        
        # قاعدة الأثلاث (Rule of Thirds)
        thirds_analysis = self._analyze_rule_of_thirds()
        
        # التعقيد البصري
        complexity = self._calculate_complexity()
        
        composition = {
            'edge_density': edge_density,
            'edges_category': self._categorize_edges(edge_density),
            'horizontal_symmetry': horizontal_symmetry,
            'vertical_symmetry': vertical_symmetry,
            'symmetry_level': self._evaluate_symmetry(horizontal_symmetry, vertical_symmetry),
            'thirds_score': thirds_analysis,
            'complexity': complexity,
            'complexity_category': self._categorize_complexity(complexity)
        }
        
        return composition
    
    def _calculate_symmetry(self, image, axis):
        """حساب التماثل"""
        if axis == 0:  # أفقي
            top = image[:image.shape[0]//2, :]
            bottom = np.flipud(image[image.shape[0]//2:, :])
            min_height = min(top.shape[0], bottom.shape[0])
            diff = np.mean(np.abs(top[:min_height] - bottom[:min_height]))
        else:  # عمودي
            left = image[:, :image.shape[1]//2]
            right = np.fliplr(image[:, image.shape[1]//2:])
            min_width = min(left.shape[1], right.shape[1])
            diff = np.mean(np.abs(left[:, :min_width] - right[:, :min_width]))
        
        symmetry_score = max(0, 100 - (diff / 255 * 100))
        return symmetry_score
    
    def _evaluate_symmetry(self, h_sym, v_sym):
        """تقييم مستوى التماثل"""
        avg_sym = (h_sym + v_sym) / 2
        if avg_sym > 80:
            return 'متماثل جداً (Highly Symmetrical)'
        elif avg_sym > 60:
            return 'متماثل (Symmetrical)'
        elif avg_sym > 40:
            return 'متماثل جزئياً (Partially Symmetrical)'
        else:
            return 'غير متماثل (Asymmetrical)'
    
    def _analyze_rule_of_thirds(self):
        """تحليل قاعدة الأثلاث"""
        h, w = self.gray_image.shape
        
        # نقاط التقاطع
        points = [
            (h//3, w//3), (h//3, 2*w//3),
            (2*h//3, w//3), (2*h//3, 2*w//3)
        ]
        
        # حساب الكثافة عند نقاط التقاطع
        total_interest = 0
        for y, x in points:
            region = self.gray_image[max(0, y-20):min(h, y+20), 
                                     max(0, x-20):min(w, x+20)]
            interest = np.std(region)
            total_interest += interest
        
        # نقاط الاهتمام في الأثلاث
        score = min(100, total_interest / 4)
        return score
    
    def _calculate_complexity(self):
        """حساب التعقيد البصري"""
        # استخدام entropy كمقياس للتعقيد
        entropy = filters.rank.entropy(self.gray_image, np.ones((9, 9)))
        complexity_score = np.mean(entropy) * 20
        return min(100, complexity_score)
    
    def _categorize_complexity(self, complexity):
        """تصنيف التعقيد"""
        if complexity < 30:
            return 'بسيط جداً (Very Simple)'
        elif complexity < 50:
            return 'بسيط (Simple)'
        elif complexity < 70:
            return 'متوسط التعقيد (Moderate)'
        elif complexity < 85:
            return 'معقد (Complex)'
        else:
            return 'معقد جداً (Very Complex)'
    
    def _categorize_edges(self, edge_density):
        """تصنيف كثافة الحواف"""
        if edge_density < 5:
            return 'ناعم جداً (Very Smooth)'
        elif edge_density < 10:
            return 'ناعم (Smooth)'
        elif edge_density < 15:
            return 'متوسط التفاصيل (Medium Detail)'
        elif edge_density < 20:
            return 'كثير التفاصيل (Detailed)'
        else:
            return 'كثير التفاصيل جداً (Highly Detailed)'
    
    def analyze_quality(self):
        """تحليل جودة الصورة"""
        # الحدة (Sharpness)
        laplacian_var = cv2.Laplacian(self.gray_image, cv2.CV_64F).var()
        
        # التشويش (Noise)
        noise_level = self._estimate_noise()
        
        # الوضوح (Clarity)
        clarity = self._calculate_clarity()
        
        quality = {
            'sharpness': laplacian_var,
            'sharpness_category': self._categorize_sharpness(laplacian_var),
            'noise_level': noise_level,
            'noise_category': self._categorize_noise(noise_level),
            'clarity': clarity,
            'overall_quality': self._calculate_overall_quality(laplacian_var, noise_level, clarity)
        }
        
        return quality
    
    def _estimate_noise(self):
        """تقدير مستوى التشويش"""
        # استخدام High-pass filter
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])
        filtered = cv2.filter2D(self.gray_image, -1, kernel)
        noise = np.std(filtered)
        return noise
    
    def _calculate_clarity(self):
        """حساب الوضوح"""
        # استخدام gradient magnitude
        sobelx = cv2.Sobel(self.gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(self.gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        clarity_score = np.mean(gradient_magnitude)
        return clarity_score
    
    def _categorize_sharpness(self, sharpness):
        """تصنيف الحدة"""
        if sharpness < 100:
            return '🔴 ضبابية (Blurry)'
        elif sharpness < 500:
            return '🟡 حدة منخفضة (Low Sharpness)'
        elif sharpness < 1000:
            return '🟢 حدة متوسطة (Medium Sharpness)'
        elif sharpness < 2000:
            return '🟢 حدة عالية (Sharp)'
        else:
            return '🟢 حدة عالية جداً (Very Sharp)'
    
    def _categorize_noise(self, noise):
        """تصنيف التشويش"""
        if noise < 10:
            return '🟢 نظيفة جداً (Very Clean)'
        elif noise < 20:
            return '🟢 نظيفة (Clean)'
        elif noise < 30:
            return '🟡 تشويش خفيف (Slight Noise)'
        elif noise < 40:
            return '🟡 تشويش متوسط (Moderate Noise)'
        else:
            return '🔴 تشويش عالي (High Noise)'
    
    def _calculate_overall_quality(self, sharpness, noise, clarity):
        """حساب الجودة الإجمالية"""
        # تطبيع القيم
        sharp_score = min(100, sharpness / 20)
        noise_score = max(0, 100 - noise)
        clarity_score = min(100, clarity / 2)
        
        overall = (sharp_score * 0.4 + noise_score * 0.3 + clarity_score * 0.3)
        
        if overall > 80:
            return '⭐⭐⭐⭐⭐ ممتازة'
        elif overall > 60:
            return '⭐⭐⭐⭐ جيدة جداً'
        elif overall > 40:
            return '⭐⭐⭐ جيدة'
        elif overall > 20:
            return '⭐⭐ مقبولة'
        else:
            return '⭐ ضعيفة'
    
    def analyze_typography(self):
        """تحليل الخطوط والنصوص في الصورة"""
        typography_analysis = {
            'has_text': False,
            'text_detected': [],
            'languages': [],
            'readability_score': 0,
            'text_regions': 0,
            'contrast_quality': 'غير متاح',
            'average_text_size': 0,
            'text_clarity': 'غير متاح'
        }
        
        # كشف مناطق النصوص باستخدام MSER
        text_regions = self._detect_text_regions()
        typography_analysis['text_regions'] = len(text_regions)
        
        if len(text_regions) > 0:
            typography_analysis['has_text'] = True
        
        # إذا كان EasyOCR متاحاً، استخدمه للتحليل المتقدم
        if EASYOCR_AVAILABLE and len(text_regions) > 0:
            try:
                # تهيئة EasyOCR للعربية والإنجليزية
                reader = easyocr.Reader(['ar', 'en'], gpu=False)
                
                # قراءة النصوص
                results = reader.readtext(self.cv_image)
                
                if results:
                    typography_analysis['has_text'] = True
                    
                    # تحليل النصوص المكتشفة
                    arabic_texts = []
                    english_texts = []
                    all_texts = []
                    text_sizes = []
                    confidences = []
                    
                    for (bbox, text, confidence) in results:
                        all_texts.append(text)
                        confidences.append(confidence)
                        
                        # تحليل حجم النص
                        box_height = self._calculate_bbox_height(bbox)
                        text_sizes.append(box_height)
                        
                        # كشف اللغة
                        if self._contains_arabic(text):
                            arabic_texts.append(text)
                        if self._contains_english(text):
                            english_texts.append(text)
                    
                    # تحديد اللغات
                    if arabic_texts:
                        typography_analysis['languages'].append('عربي')
                    if english_texts:
                        typography_analysis['languages'].append('إنجليزي')
                    
                    # النصوص المكتشفة
                    typography_analysis['text_detected'] = all_texts[:10]  # أول 10 نصوص
                    
                    # متوسط حجم النص
                    if text_sizes:
                        typography_analysis['average_text_size'] = np.mean(text_sizes)
                    
                    # درجة القابلية للقراءة (بناءً على الثقة)
                    if confidences:
                        typography_analysis['readability_score'] = np.mean(confidences) * 100
                    
                    # تحليل وضوح النص
                    typography_analysis['text_clarity'] = self._evaluate_text_clarity(
                        confidences, text_sizes
                    )
                    
                    # تحليل التباين
                    typography_analysis['contrast_quality'] = self._analyze_text_contrast(results)
            
            except Exception as e:
                print(f"خطأ في OCR: {e}")
        
        # إذا لم يكن EasyOCR متاحاً، استخدم التحليل الأساسي
        elif len(text_regions) > 0:
            typography_analysis['has_text'] = True
            typography_analysis['text_clarity'] = 'تحليل أساسي - مناطق نصية مكتشفة'
            typography_analysis['contrast_quality'] = self._analyze_basic_contrast()
        
        return typography_analysis
    
    def _detect_text_regions(self):
        """كشف مناطق النصوص باستخدام MSER"""
        try:
            # تحويل إلى grayscale
            gray = self.gray_image.copy()
            
            # MSER detector
            mser = cv2.MSER_create()
            regions, _ = mser.detectRegions(gray)
            
            # تصفية المناطق
            text_regions = []
            for region in regions:
                if len(region) > 10 and len(region) < 5000:
                    text_regions.append(region)
            
            return text_regions
        except:
            return []
    
    def _contains_arabic(self, text):
        """كشف إذا كان النص يحتوي على أحرف عربية"""
        arabic_pattern = re.compile(r'[\u0600-\u06FF]')
        return bool(arabic_pattern.search(text))
    
    def _contains_english(self, text):
        """كشف إذا كان النص يحتوي على أحرف إنجليزية"""
        english_pattern = re.compile(r'[a-zA-Z]')
        return bool(english_pattern.search(text))
    
    def _calculate_bbox_height(self, bbox):
        """حساب ارتفاع صندوق النص"""
        points = np.array(bbox)
        height = np.max(points[:, 1]) - np.min(points[:, 1])
        return height
    
    def _evaluate_text_clarity(self, confidences, text_sizes):
        """تقييم وضوح النص"""
        if not confidences:
            return 'غير متاح'
        
        avg_confidence = np.mean(confidences)
        avg_size = np.mean(text_sizes) if text_sizes else 0
        
        # تقييم بناءً على الثقة والحجم
        if avg_confidence > 0.9 and avg_size > 20:
            return '🟢 واضح جداً ومقروء'
        elif avg_confidence > 0.75 and avg_size > 15:
            return '🟢 واضح ومقروء'
        elif avg_confidence > 0.6 and avg_size > 10:
            return '🟡 مقروء بشكل متوسط'
        elif avg_confidence > 0.4:
            return '🟡 صعب القراءة قليلاً'
        else:
            return '🔴 غير واضح'
    
    def _analyze_text_contrast(self, ocr_results):
        """تحليل تباين النص مع الخلفية"""
        if not ocr_results:
            return 'غير متاح'
        
        contrasts = []
        
        for (bbox, text, confidence) in ocr_results:
            # استخراج منطقة النص
            points = np.array(bbox, dtype=np.int32)
            x_min, y_min = points.min(axis=0)
            x_max, y_max = points.max(axis=0)
            
            # التأكد من أن الإحداثيات ضمن حدود الصورة
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(self.width, x_max)
            y_max = min(self.height, y_max)
            
            if x_max > x_min and y_max > y_min:
                text_region = self.gray_image[y_min:y_max, x_min:x_max]
                
                if text_region.size > 0:
                    # حساب التباين
                    contrast = np.std(text_region)
                    contrasts.append(contrast)
        
        if contrasts:
            avg_contrast = np.mean(contrasts)
            
            if avg_contrast > 60:
                return '🟢 تباين ممتاز (نص واضح على الخلفية)'
            elif avg_contrast > 40:
                return '🟢 تباين جيد'
            elif avg_contrast > 25:
                return '🟡 تباين متوسط'
            else:
                return '🔴 تباين ضعيف (صعوبة في القراءة)'
        
        return 'غير متاح'
    
    def _analyze_basic_contrast(self):
        """تحليل أساسي للتباين عندما OCR غير متاح"""
        contrast = np.std(self.gray_image)
        
        if contrast > 60:
            return '🟢 تباين عالي (جيد للنصوص)'
        elif contrast > 40:
            return '🟢 تباين متوسط'
        else:
            return '🟡 تباين منخفض'
    
    def analyze_semiotics(self):
        """التحليل السيميائي وكشف الرموز البصرية"""
        semiotic_analysis = {
            'geometric_shapes': {},
            'circles': [],
            'triangles': [],
            'rectangles': [],
            'squares': [],
            'lines': [],
            'patterns': {},
            'focal_points': [],
            'visual_hierarchy': {},
            'symmetry_type': '',
            'repetition_score': 0,
            'symbolic_elements': []
        }
        
        # كشف الأشكال الهندسية
        shapes = self._detect_geometric_shapes()
        semiotic_analysis['geometric_shapes'] = shapes
        
        # كشف الدوائر (رموز الكمال، الأبدية، الوحدة)
        circles = self._detect_circles()
        semiotic_analysis['circles'] = circles
        
        # كشف المثلثات (رموز القوة، الاستقرار، الديناميكية)
        triangles = self._detect_triangles()
        semiotic_analysis['triangles'] = triangles
        
        # كشف المستطيلات والمربعات (رموز الثبات، النظام)
        rectangles, squares = self._detect_rectangles()
        semiotic_analysis['rectangles'] = rectangles
        semiotic_analysis['squares'] = squares
        
        # كشف الخطوط المهيمنة (رموز الاتجاه، الحركة)
        lines = self._detect_dominant_lines()
        semiotic_analysis['lines'] = lines
        
        # تحليل الأنماط والتكرار
        patterns = self._analyze_patterns()
        semiotic_analysis['patterns'] = patterns
        semiotic_analysis['repetition_score'] = patterns.get('repetition_intensity', 0)
        
        # تحليل نقاط التركيز البصرية
        focal_points = self._detect_focal_points()
        semiotic_analysis['focal_points'] = focal_points
        
        # تحليل الهرمية البصرية
        hierarchy = self._analyze_visual_hierarchy()
        semiotic_analysis['visual_hierarchy'] = hierarchy
        
        # تحليل نوع التماثل
        symmetry_type = self._determine_symmetry_type()
        semiotic_analysis['symmetry_type'] = symmetry_type
        
        # استنتاج العناصر الرمزية
        symbolic = self._infer_symbolic_elements(semiotic_analysis)
        semiotic_analysis['symbolic_elements'] = symbolic
        
        return semiotic_analysis
    
    def _detect_geometric_shapes(self):
        """كشف الأشكال الهندسية الأساسية"""
        # تحسين الصورة للكشف
        blurred = cv2.GaussianBlur(self.gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # إيجاد الكونتورات
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        shapes_count = {
            'total_shapes': len(contours),
            'simple_shapes': 0,
            'complex_shapes': 0,
            'organic_shapes': 0
        }
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 100:  # تجاهل الأشكال الصغيرة جداً
                continue
            
            # تبسيط الشكل
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            vertices = len(approx)
            
            if vertices <= 4:
                shapes_count['simple_shapes'] += 1
            elif vertices <= 8:
                shapes_count['complex_shapes'] += 1
            else:
                shapes_count['organic_shapes'] += 1
        
        return shapes_count
    
    def _detect_circles(self):
        """كشف الدوائر (رمز الكمال والأبدية)"""
        circles_data = []
        
        # استخدام Hough Circle Transform
        blurred = cv2.GaussianBlur(self.gray_image, (9, 9), 2)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=100,
            param2=30,
            minRadius=10,
            maxRadius=min(self.width, self.height) // 4
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :]:
                x, y, r = circle
                # تحليل موقع الدائرة
                position = self._categorize_position(x, y)
                circles_data.append({
                    'center': (int(x), int(y)),
                    'radius': int(r),
                    'position': position,
                    'size_category': self._categorize_size(r)
                })
        
        return circles_data
    
    def _detect_triangles(self):
        """كشف المثلثات (رمز القوة والديناميكية)"""
        triangles_data = []
        
        # تحسين الصورة
        blurred = cv2.GaussianBlur(self.gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # إيجاد الكونتورات
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue
            
            # تبسيط الشكل
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # إذا كان لديه 3 رؤوس فهو مثلث
            if len(approx) == 3:
                # حساب المركز
                M = cv2.moments(contour)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    
                    position = self._categorize_position(cx, cy)
                    orientation = self._determine_triangle_orientation(approx)
                    
                    triangles_data.append({
                        'vertices': len(approx),
                        'center': (cx, cy),
                        'area': area,
                        'position': position,
                        'orientation': orientation
                    })
        
        return triangles_data
    
    def _detect_rectangles(self):
        """كشف المستطيلات والمربعات (رمز الثبات والنظام)"""
        rectangles_data = []
        squares_data = []
        
        # تحسين الصورة
        blurred = cv2.GaussianBlur(self.gray_image, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # إيجاد الكونتورات
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 500:
                continue
            
            # تبسيط الشكل
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # إذا كان لديه 4 رؤوس
            if len(approx) == 4:
                # حساب نسبة العرض للارتفاع
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = float(w) / h if h != 0 else 0
                
                # حساب المركز
                cx = x + w // 2
                cy = y + h // 2
                position = self._categorize_position(cx, cy)
                
                shape_data = {
                    'center': (cx, cy),
                    'width': w,
                    'height': h,
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'position': position
                }
                
                # تمييز المربعات (نسبة قريبة من 1)
                if 0.85 <= aspect_ratio <= 1.15:
                    squares_data.append(shape_data)
                else:
                    rectangles_data.append(shape_data)
        
        return rectangles_data, squares_data
    
    def _detect_dominant_lines(self):
        """كشف الخطوط المهيمنة (رمز الاتجاه والحركة)"""
        lines_data = {
            'horizontal': 0,
            'vertical': 0,
            'diagonal': 0,
            'dominant_direction': '',
            'line_count': 0
        }
        
        # كشف الخطوط باستخدام Hough Transform
        edges = cv2.Canny(self.gray_image, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi/180,
            threshold=100,
            minLineLength=min(self.width, self.height) // 10,
            maxLineGap=10
        )
        
        if lines is not None:
            lines_data['line_count'] = len(lines)
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # حساب الزاوية
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                
                # تصنيف الخط
                if angle < 30 or angle > 150:
                    lines_data['horizontal'] += 1
                elif 60 < angle < 120:
                    lines_data['vertical'] += 1
                else:
                    lines_data['diagonal'] += 1
            
            # تحديد الاتجاه المهيمن
            max_dir = max(
                lines_data['horizontal'],
                lines_data['vertical'],
                lines_data['diagonal']
            )
            
            if max_dir == lines_data['horizontal']:
                lines_data['dominant_direction'] = 'أفقي (استقرار، هدوء)'
            elif max_dir == lines_data['vertical']:
                lines_data['dominant_direction'] = 'عمودي (قوة، صعود)'
            else:
                lines_data['dominant_direction'] = 'قطري (حركة، ديناميكية)'
        
        return lines_data
    
    def _analyze_patterns(self):
        """تحليل الأنماط والتكرار"""
        patterns = {
            'has_repetition': False,
            'repetition_intensity': 0,
            'pattern_type': '',
            'regularity': 0
        }
        
        # استخدام autocorrelation للكشف عن الأنماط
        # تقسيم الصورة إلى شبكة
        grid_size = 8
        h_step = self.height // grid_size
        w_step = self.width // grid_size
        
        cells = []
        for i in range(grid_size):
            for j in range(grid_size):
                cell = self.gray_image[
                    i*h_step:(i+1)*h_step,
                    j*w_step:(j+1)*w_step
                ]
                if cell.size > 0:
                    cells.append(np.mean(cell))
        
        if len(cells) > 0:
            # حساب التباين بين الخلايا
            cell_variance = np.var(cells)
            cell_std = np.std(cells)
            
            # كشف التكرار
            if cell_std < 30:  # قيم متشابهة تشير إلى تكرار
                patterns['has_repetition'] = True
                patterns['repetition_intensity'] = max(0, 100 - cell_std * 2)
                patterns['pattern_type'] = 'نمط منتظم'
                patterns['regularity'] = min(100, 100 - cell_variance / 10)
            elif cell_std < 60:
                patterns['has_repetition'] = True
                patterns['repetition_intensity'] = max(0, 70 - cell_std)
                patterns['pattern_type'] = 'نمط شبه منتظم'
                patterns['regularity'] = min(100, 80 - cell_variance / 10)
            else:
                patterns['pattern_type'] = 'نمط عشوائي/عضوي'
                patterns['regularity'] = 30
        
        return patterns
    
    def _detect_focal_points(self):
        """كشف نقاط التركيز البصرية"""
        focal_points = []
        
        # استخدام Saliency Detection
        saliency = cv2.saliency.StaticSaliencyFineGrained_create()
        success, saliency_map = saliency.computeSaliency(self.cv_image)
        
        if success:
            # تحويل إلى 8-bit
            saliency_map = (saliency_map * 255).astype("uint8")
            
            # إيجاد النقاط البارزة
            _, thresh = cv2.threshold(saliency_map, 200, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 100:
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        
                        position = self._categorize_position(cx, cy)
                        focal_points.append({
                            'center': (cx, cy),
                            'area': area,
                            'position': position,
                            'importance': min(100, area / 100)
                        })
            
            # ترتيب حسب الأهمية
            focal_points = sorted(focal_points, key=lambda x: x['importance'], reverse=True)[:5]
        
        return focal_points
    
    def _analyze_visual_hierarchy(self):
        """تحليل الهرمية البصرية"""
        hierarchy = {
            'primary_level': 0,
            'secondary_level': 0,
            'tertiary_level': 0,
            'hierarchy_strength': 0
        }
        
        # تحليل توزيع السطوع
        hist = cv2.calcHist([self.gray_image], [0], None, [256], [0, 256])
        
        # إيجاد القمم في الهيستوجرام
        peaks = []
        for i in range(10, 246):
            if hist[i] > hist[i-10] and hist[i] > hist[i+10]:
                peaks.append((i, hist[i][0]))
        
        # ترتيب القمم
        peaks = sorted(peaks, key=lambda x: x[1], reverse=True)
        
        if len(peaks) >= 1:
            hierarchy['primary_level'] = peaks[0][0]
        if len(peaks) >= 2:
            hierarchy['secondary_level'] = peaks[1][0]
        if len(peaks) >= 3:
            hierarchy['tertiary_level'] = peaks[2][0]
        
        # حساب قوة الهرمية
        if len(peaks) >= 2:
            diff = abs(peaks[0][1] - peaks[1][1])
            hierarchy['hierarchy_strength'] = min(100, diff / 1000)
        
        return hierarchy
    
    def _determine_symmetry_type(self):
        """تحديد نوع التماثل"""
        h_sym = self._calculate_symmetry(self.gray_image, axis=0)
        v_sym = self._calculate_symmetry(self.gray_image, axis=1)
        
        if h_sym > 70 and v_sym > 70:
            return 'تماثل رباعي (Radial Symmetry) - رمز الكمال'
        elif h_sym > 60:
            return 'تماثل أفقي (Horizontal) - رمز الهدوء والاستقرار'
        elif v_sym > 60:
            return 'تماثل عمودي (Vertical) - رمز القوة والارتفاع'
        elif h_sym > 40 or v_sym > 40:
            return 'تماثل جزئي (Partial) - توازن ديناميكي'
        else:
            return 'غير متماثل (Asymmetric) - حرية وحركة'
    
    def _infer_symbolic_elements(self, analysis):
        """استنتاج العناصر الرمزية من التحليل"""
        symbolic = []
        
        # تحليل الدوائر
        if len(analysis['circles']) > 0:
            symbolic.append({
                'element': 'الدائرة',
                'count': len(analysis['circles']),
                'meaning': 'الكمال، الأبدية، الوحدة، الاستمرارية',
                'cultural': 'رمز عالمي للكمال والتناغم'
            })
        
        # تحليل المثلثات
        if len(analysis['triangles']) > 0:
            symbolic.append({
                'element': 'المثلث',
                'count': len(analysis['triangles']),
                'meaning': 'القوة، الثبات، التوازن، الثالوث',
                'cultural': 'يرمز للقوة والديناميكية حسب الاتجاه'
            })
        
        # تحليل المربعات
        if len(analysis['squares']) > 0:
            symbolic.append({
                'element': 'المربع',
                'count': len(analysis['squares']),
                'meaning': 'الثبات، النظام، الأرض، المادية',
                'cultural': 'رمز الاستقرار والأساس المتين'
            })
        
        # تحليل المستطيلات
        if len(analysis['rectangles']) > 0:
            symbolic.append({
                'element': 'المستطيل',
                'count': len(analysis['rectangles']),
                'meaning': 'التنظيم، الإطار، الحدود، البناء',
                'cultural': 'يمثل النظام والتخطيط'
            })
        
        # تحليل الخطوط
        if analysis['lines'].get('line_count', 0) > 10:
            direction = analysis['lines'].get('dominant_direction', '')
            if 'أفقي' in direction:
                symbolic.append({
                    'element': 'الخطوط الأفقية',
                    'count': analysis['lines']['horizontal'],
                    'meaning': 'الهدوء، الاستقرار، الأفق، الراحة',
                    'cultural': 'توحي بالسكون والسلام'
                })
            elif 'عمودي' in direction:
                symbolic.append({
                    'element': 'الخطوط العمودية',
                    'count': analysis['lines']['vertical'],
                    'meaning': 'القوة، الصعود، الطموح، الروحانية',
                    'cultural': 'توحي بالعلو والارتفاع'
                })
            elif 'قطري' in direction:
                symbolic.append({
                    'element': 'الخطوط القطرية',
                    'count': analysis['lines']['diagonal'],
                    'meaning': 'الحركة، الديناميكية، التغيير، الطاقة',
                    'cultural': 'توحي بالحركة والنشاط'
                })
        
        # تحليل التكرار
        if analysis['patterns']['has_repetition']:
            symbolic.append({
                'element': 'التكرار والنمط',
                'count': analysis['patterns']['repetition_intensity'],
                'meaning': 'الإيقاع، التناغم، النظام، الوحدة',
                'cultural': 'يخلق إحساساً بالتماسك والانسجام'
            })
        
        # تحليل التماثل
        if 'رباعي' in analysis['symmetry_type']:
            symbolic.append({
                'element': 'التماثل الرباعي',
                'count': 1,
                'meaning': 'الكمال المطلق، التوازن الشامل، المركزية',
                'cultural': 'أعلى درجات التناغم والكمال'
            })
        
        return symbolic
    
    def _categorize_position(self, x, y):
        """تصنيف موقع العنصر في الصورة"""
        # تقسيم الصورة إلى 9 مناطق (شبكة 3×3)
        h_third = self.height // 3
        w_third = self.width // 3
        
        if y < h_third:
            v_pos = 'أعلى'
        elif y < 2 * h_third:
            v_pos = 'وسط'
        else:
            v_pos = 'أسفل'
        
        if x < w_third:
            h_pos = 'يسار'
        elif x < 2 * w_third:
            h_pos = 'وسط'
        else:
            h_pos = 'يمين'
        
        return f'{v_pos} {h_pos}'
    
    def _categorize_size(self, radius):
        """تصنيف حجم العنصر"""
        max_dimension = max(self.width, self.height)
        relative_size = (radius * 2) / max_dimension * 100
        
        if relative_size > 30:
            return 'كبير (مهيمن)'
        elif relative_size > 15:
            return 'متوسط (بارز)'
        else:
            return 'صغير (ثانوي)'
    
    def _determine_triangle_orientation(self, approx):
        """تحديد اتجاه المثلث"""
        points = approx.reshape(-1, 2)
        
        # إيجاد أعلى وأسفل نقطة
        top_point = points[np.argmin(points[:, 1])]
        bottom_point = points[np.argmax(points[:, 1])]
        
        # حساب المركز
        center_y = np.mean(points[:, 1])
        
        if top_point[1] < center_y - 10:
            return 'مثلث صاعد (طموح، نمو)'
        elif bottom_point[1] > center_y + 10:
            return 'مثلث هابط (استقرار، أساس)'
        else:
            return 'مثلث أفقي (توازن)'


def main():
    """الوظيفة الرئيسية للتطبيق"""
    
    # العنوان الرئيسي
    st.markdown("""
    <div class="main-header">
        <h1>🎨 محلل الصور الأكاديمي المتقدم</h1>
        <p>تحليل تقني شامل للخصائص الفنية والجمالية للصور</p>
        <p style="font-size: 0.9rem; opacity: 0.9;">معالجة الصور • الرؤية الحاسوبية • التحليل الأكاديمي</p>
        <div style="margin-top: 1.2rem; padding-top: 1rem; border-top: 1px solid rgba(255,255,255,0.25);
                    font-size: 0.95rem; line-height: 1.9;">
            <div style="font-weight: 700; letter-spacing: 1px; color: #f0d78a;">
                بحث طالبة الدكتوراه
            </div>
            <div style="font-size: 1.2rem; font-weight: 700; margin-top: 0.3rem;">
                عبير قاسم حلف
            </div>
            <div style="font-size: 0.9rem; opacity: 0.9; margin-top: 0.4rem;">
                الجامعة المستنصرية &nbsp;•&nbsp; كلية التربية الأساسية
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # الشريط الجانبي
    with st.sidebar:
        # بطاقة الباحثة الأكاديمية
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
                    color: white; padding: 1.2rem; border-radius: 10px;
                    border-right: 4px solid #c59849; margin-bottom: 1.2rem;
                    text-align: center;">
            <div style="font-size: 0.8rem; letter-spacing: 2px; color: #f0d78a;
                        font-weight: 700; margin-bottom: 0.5rem;">
                بحث طالبة الدكتوراه
            </div>
            <div style="font-size: 1.25rem; font-weight: 700; line-height: 1.4;">
                عبير قاسم حلف
            </div>
            <div style="height: 1px; background: rgba(255,255,255,0.3);
                        margin: 0.8rem 0;"></div>
            <div style="font-size: 0.95rem; font-weight: 500;">
                الجامعة المستنصرية
            </div>
            <div style="font-size: 0.85rem; opacity: 0.9; margin-top: 0.3rem;">
                كلية التربية الأساسية
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📤 رفع الصورة")
        uploaded_file = st.file_uploader(
            "اختر صورة للتحليل",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            help="يدعم جميع صيغ الصور الشائعة"
        )

        st.markdown("---")
        st.markdown("### ℹ️ عن التطبيق")
        st.info("""
        **محلل الصور الأكاديمي** يستخدم تقنيات متقدمة في:
        - معالجة الصور الرقمية
        - الرؤية الحاسوبية
        - التحليل الإحصائي
        - نظرية الألوان
        
        **بدون استخدام الذكاء الاصطناعي**
        """)
        
        st.markdown("---")
        st.markdown("### 🔬 التقنيات المستخدمة")
        st.markdown("""
        - OpenCV
        - PIL/Pillow
        - NumPy
        - SciPy
        - Scikit-image
        """)
    
    # المحتوى الرئيسي
    if uploaded_file is not None:
        # قراءة الصورة
        image = Image.open(uploaded_file)
        
        # إنشاء المحلل
        analyzer = ImageAnalyzer(image)
        
        # عرض الصورة
        st.markdown('<div class="section-header">📸 الصورة المحللة</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, use_container_width=True)
        
        # المعلومات الأساسية
        st.markdown('<div class="section-header">📊 المعلومات الأساسية</div>', unsafe_allow_html=True)
        basic_info = analyzer.get_basic_info()
        
        cols = st.columns(3)
        info_items = list(basic_info.items())
        for idx, (key, value) in enumerate(info_items):
            with cols[idx % 3]:
                st.metric(label=key, value=value)
        
        # تحليل الألوان
        st.markdown('<div class="section-header">🎨 تحليل الألوان المتقدم</div>', unsafe_allow_html=True)
        with st.spinner('جارٍ تحليل الألوان...'):
            color_analysis = analyzer.analyze_colors()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🎯 الألوان السائدة")
            for i, color_info in enumerate(color_analysis['dominant_colors'][:5], 1):
                rgb = color_info['rgb']
                hex_color = color_info['hex']
                percentage = color_info['percentage']
                
                st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 10px;">
                    <div style="width: 50px; height: 50px; background-color: {hex_color}; 
                                border-radius: 8px; margin-left: 15px; border: 2px solid #ddd;">
                    </div>
                    <div>
                        <strong>اللون {i}</strong><br>
                        RGB: {rgb[0]}, {rgb[1]}, {rgb[2]}<br>
                        النسبة: {percentage:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 📈 إحصائيات اللون")
            st.metric("متوسط التشبع", f"{color_analysis['avg_saturation']:.1f}%")
            st.metric("متوسط السطوع", f"{color_analysis['avg_brightness']:.1f}%")
            st.metric("تباين الألوان", f"{color_analysis['color_variance']:.2f}")
            st.info(f"**درجة حرارة اللون:** {color_analysis['color_temperature']}")
        
        # تحليل الإضاءة
        st.markdown('<div class="section-header">💡 تحليل الإضاءة والتباين</div>', unsafe_allow_html=True)
        with st.spinner('جارٍ تحليل الإضاءة...'):
            lighting = analyzer.analyze_lighting()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("السطوع", f"{lighting['brightness']:.1f}/255")
            st.caption(lighting['brightness_category'])
        
        with col2:
            st.metric("التباين", f"{lighting['contrast']:.1f}")
            st.caption(lighting['contrast_category'])
        
        with col3:
            st.metric("النطاق الديناميكي", f"{lighting['dynamic_range']:.0f}")
            st.caption(lighting['exposure_balance'])
        
        # توزيع النغمات
        st.markdown("#### 📊 توزيع النغمات (Tonal Distribution)")
        tone_data = pd.DataFrame({
            'النوع': ['نغمات مظلمة', 'نغمات متوسطة', 'نغمات مشرقة'],
            'النسبة': [lighting['dark_pixels'], lighting['midtone_pixels'], lighting['bright_pixels']]
        })
        st.bar_chart(tone_data.set_index('النوع'))
        
        # تحليل التكوين
        st.markdown('<div class="section-header">🎭 تحليل التكوين والبناء الفني</div>', unsafe_allow_html=True)
        with st.spinner('جارٍ تحليل التكوين...'):
            composition = analyzer.analyze_composition()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📐 التماثل والتوازن")
            st.metric("التماثل الأفقي", f"{composition['horizontal_symmetry']:.1f}%")
            st.metric("التماثل العمودي", f"{composition['vertical_symmetry']:.1f}%")
            st.info(f"**التقييم:** {composition['symmetry_level']}")
        
        with col2:
            st.markdown("#### 🎯 خصائص التكوين")
            st.metric("كثافة الحواف", f"{composition['edge_density']:.1f}%")
            st.caption(composition['edges_category'])
            st.metric("نقاط الاهتمام (قاعدة الأثلاث)", f"{composition['thirds_score']:.1f}/100")
            st.metric("التعقيد البصري", f"{composition['complexity']:.1f}/100")
            st.caption(composition['complexity_category'])
        
        # تحليل الجودة
        st.markdown('<div class="section-header">⭐ تحليل الجودة والوضوح</div>', unsafe_allow_html=True)
        with st.spinner('جارٍ تحليل الجودة...'):
            quality = analyzer.analyze_quality()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("الحدة", f"{quality['sharpness']:.1f}")
            st.caption(quality['sharpness_category'])
        
        with col2:
            st.metric("مستوى التشويش", f"{quality['noise_level']:.1f}")
            st.caption(quality['noise_category'])
        
        with col3:
            st.metric("الوضوح", f"{quality['clarity']:.1f}")
        
        # تحليل الخطوط والنصوص
        st.markdown('<div class="section-header">📝 تحليل الخطوط والنصوص (Typography)</div>', unsafe_allow_html=True)
        
        if EASYOCR_AVAILABLE:
            with st.spinner('جارٍ تحليل النصوص والخطوط... (قد يستغرق بضع ثوانٍ)'):
                typography = analyzer.analyze_typography()
        else:
            st.warning('⚠️ مكتبة EasyOCR غير مثبتة. سيتم استخدام التحليل الأساسي فقط.')
            with st.spinner('جارٍ التحليل الأساسي للنصوص...'):
                typography = analyzer.analyze_typography()
        
        if typography['has_text']:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🔤 معلومات النصوص")
                st.metric("مناطق نصية مكتشفة", typography['text_regions'])
                
                if typography['languages']:
                    st.info(f"**اللغات المكتشفة:** {' + '.join(typography['languages'])}")
                
                if typography['average_text_size'] > 0:
                    st.metric("متوسط حجم الخط", f"{typography['average_text_size']:.1f} بكسل")
                
                if typography['readability_score'] > 0:
                    st.metric("درجة القابلية للقراءة", f"{typography['readability_score']:.1f}%")
            
            with col2:
                st.markdown("#### 📊 جودة الخطوط")
                
                if typography['text_clarity'] != 'غير متاح':
                    st.info(f"**وضوح الخط:** {typography['text_clarity']}")
                
                if typography['contrast_quality'] != 'غير متاح':
                    st.info(f"**التباين:** {typography['contrast_quality']}")
            
            # عرض النصوص المكتشفة
            if typography['text_detected']:
                st.markdown("#### 📄 نماذج من النصوص المكتشفة")
                
                # عرض النصوص في صناديق
                for i, text in enumerate(typography['text_detected'][:5], 1):
                    if text.strip():  # تجاهل النصوص الفارغة
                        # تحديد اتجاه النص
                        is_arabic = any('\u0600' <= c <= '\u06FF' for c in text)
                        direction = "rtl" if is_arabic else "ltr"
                        
                        st.markdown(f"""
                        <div style="background: #f7fafc; padding: 1rem; border-radius: 8px; 
                                    border-right: 4px solid #c59849; margin-bottom: 0.5rem;
                                    direction: {direction};">
                            <strong>نص {i}:</strong> {text}
                        </div>
                        """, unsafe_allow_html=True)
            
            # تقييم عام للخطوط
            st.markdown("#### 🎓 التقييم الأكاديمي للخطوط")
            
            assessment = []
            
            if typography['readability_score'] > 80:
                assessment.append("✅ الخطوط واضحة جداً ومقروءة بسهولة")
            elif typography['readability_score'] > 60:
                assessment.append("✅ الخطوط مقروءة بشكل جيد")
            elif typography['readability_score'] > 40:
                assessment.append("⚠️ الخطوط مقروءة بشكل متوسط")
            else:
                assessment.append("❌ الخطوط صعبة القراءة")
            
            if typography['languages']:
                if 'عربي' in typography['languages'] and 'إنجليزي' in typography['languages']:
                    assessment.append("✅ تدعم اللغتين العربية والإنجليزية")
                elif 'عربي' in typography['languages']:
                    assessment.append("✅ خطوط عربية واضحة")
                elif 'إنجليزي' in typography['languages']:
                    assessment.append("✅ خطوط إنجليزية واضحة")
            
            if typography['average_text_size'] > 25:
                assessment.append("✅ حجم الخط كبير ومريح للقراءة")
            elif typography['average_text_size'] > 15:
                assessment.append("✅ حجم الخط مناسب")
            elif typography['average_text_size'] > 0:
                assessment.append("⚠️ حجم الخط صغير نسبياً")
            
            if '🟢' in typography['contrast_quality']:
                assessment.append("✅ تباين ممتاز بين النص والخلفية")
            elif '🟡' in typography['contrast_quality']:
                assessment.append("⚠️ تباين متوسط - يمكن تحسينه")
            elif '🔴' in typography['contrast_quality']:
                assessment.append("❌ تباين ضعيف - يحتاج تحسين")
            
            for item in assessment:
                st.markdown(f"- {item}")
        
        else:
            st.info("📋 لم يتم اكتشاف نصوص في هذه الصورة")
            st.caption("الصورة قد لا تحتوي على نصوص، أو النصوص غير واضحة بما يكفي للكشف")
        
        # التحليل السيميائي وكشف الرموز
        st.markdown('<div class="section-header">🔮 التحليل السيميائي وكشف الرموز</div>', unsafe_allow_html=True)
        st.caption("تحليل الرموز البصرية والعلامات والدلالات الثقافية")
        
        with st.spinner('جارٍ التحليل السيميائي المتقدم...'):
            semiotics = analyzer.analyze_semiotics()
        
        # عرض الأشكال الهندسية
        st.markdown("#### 🔷 الأشكال الهندسية المكتشفة")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🔴 دوائر", len(semiotics['circles']))
            if len(semiotics['circles']) > 0:
                st.caption("رمز الكمال والأبدية")
        
        with col2:
            st.metric("🔺 مثلثات", len(semiotics['triangles']))
            if len(semiotics['triangles']) > 0:
                st.caption("رمز القوة والديناميكية")
        
        with col3:
            st.metric("⬜ مربعات", len(semiotics['squares']))
            if len(semiotics['squares']) > 0:
                st.caption("رمز الثبات والنظام")
        
        with col4:
            st.metric("▭ مستطيلات", len(semiotics['rectangles']))
            if len(semiotics['rectangles']) > 0:
                st.caption("رمز التنظيم والإطار")
        
        # تفاصيل الأشكال
        if len(semiotics['circles']) > 0 or len(semiotics['triangles']) > 0 or len(semiotics['squares']) > 0:
            with st.expander("📊 تفاصيل الأشكال المكتشفة"):
                
                # الدوائر
                if len(semiotics['circles']) > 0:
                    st.markdown("**🔴 الدوائر:**")
                    for i, circle in enumerate(semiotics['circles'][:3], 1):
                        st.markdown(f"""
                        - **دائرة {i}**: نصف القطر = {circle['radius']} بكسل، 
                          الموقع: {circle['position']}, 
                          الحجم: {circle['size_category']}
                        """)
                
                # المثلثات
                if len(semiotics['triangles']) > 0:
                    st.markdown("**🔺 المثلثات:**")
                    for i, triangle in enumerate(semiotics['triangles'][:3], 1):
                        st.markdown(f"""
                        - **مثلث {i}**: المساحة = {triangle['area']:.0f} بكسل², 
                          الموقع: {triangle['position']}, 
                          الاتجاه: {triangle['orientation']}
                        """)
                
                # المربعات
                if len(semiotics['squares']) > 0:
                    st.markdown("**⬜ المربعات:**")
                    for i, square in enumerate(semiotics['squares'][:3], 1):
                        st.markdown(f"""
                        - **مربع {i}**: الحجم = {square['width']}×{square['height']} بكسل, 
                          الموقع: {square['position']}
                        """)
        
        # تحليل الخطوط
        st.markdown("#### 📏 تحليل الخطوط والاتجاهات")
        
        if semiotics['lines']['line_count'] > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("عدد الخطوط المكتشفة", semiotics['lines']['line_count'])
                
                # رسم توزيع الخطوط
                line_data = pd.DataFrame({
                    'النوع': ['أفقية', 'عمودية', 'قطرية'],
                    'العدد': [
                        semiotics['lines']['horizontal'],
                        semiotics['lines']['vertical'],
                        semiotics['lines']['diagonal']
                    ]
                })
                st.bar_chart(line_data.set_index('النوع'))
            
            with col2:
                if semiotics['lines']['dominant_direction']:
                    st.info(f"**الاتجاه المهيمن:** {semiotics['lines']['dominant_direction']}")
                
                # التفسير الرمزي
                if semiotics['lines']['horizontal'] > semiotics['lines']['vertical']:
                    st.markdown("✨ **الدلالة:** الخطوط الأفقية توحي بالاستقرار والهدوء")
                elif semiotics['lines']['vertical'] > semiotics['lines']['horizontal']:
                    st.markdown("✨ **الدلالة:** الخطوط العمودية توحي بالقوة والصعود")
                else:
                    st.markdown("✨ **الدلالة:** توازن بين الاستقرار والحركة")
        else:
            st.caption("لم يتم اكتشاف خطوط واضحة")
        
        # تحليل الأنماط والتكرار
        st.markdown("#### 🔁 تحليل الأنماط والتكرار")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if semiotics['patterns']['has_repetition']:
                st.success(f"✅ **يوجد تكرار:** {semiotics['patterns']['pattern_type']}")
                st.metric("شدة التكرار", f"{semiotics['patterns']['repetition_intensity']:.0f}%")
                st.progress(semiotics['patterns']['repetition_intensity'] / 100)
            else:
                st.info("📋 **نمط عشوائي:** لا يوجد تكرار واضح")
        
        with col2:
            st.metric("الانتظام", f"{semiotics['patterns']['regularity']:.0f}%")
            
            if semiotics['patterns']['regularity'] > 70:
                st.markdown("✨ **الدلالة:** نمط منتظم يخلق إيقاعاً بصرياً قوياً")
            elif semiotics['patterns']['regularity'] > 40:
                st.markdown("✨ **الدلالة:** نمط شبه منتظم يوازن بين النظام والحرية")
            else:
                st.markdown("✨ **الدلالة:** نمط عضوي يعكس العفوية والطبيعية")
        
        # نقاط التركيز البصرية
        st.markdown("#### 🎯 نقاط التركيز البصرية")
        
        if len(semiotics['focal_points']) > 0:
            st.success(f"تم اكتشاف {len(semiotics['focal_points'])} نقطة تركيز رئيسية")
            
            for i, fp in enumerate(semiotics['focal_points'][:3], 1):
                st.markdown(f"""
                <div style="background: #f7fafc; padding: 1rem; border-radius: 8px; 
                            border-right: 4px solid #c59849; margin-bottom: 0.5rem;">
                    <strong>نقطة التركيز {i}:</strong><br>
                    📍 الموقع: {fp['position']}<br>
                    ⭐ الأهمية: {fp['importance']:.0f}/100<br>
                    💡 الدلالة: هذه منطقة تجذب انتباه المشاهد بقوة
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📋 لا توجد نقاط تركيز بارزة - توزيع متساوٍ للانتباه")
        
        # التماثل والتوازن
        st.markdown("#### ⚖️ التماثل والتوازن الرمزي")
        
        st.info(f"**نوع التماثل:** {semiotics['symmetry_type']}")
        
        # التفسير الرمزي للتماثل
        if 'رباعي' in semiotics['symmetry_type']:
            st.success("✨ **التفسير:** التماثل الرباعي يرمز للكمال المطلق والتوازن الشامل، وهو أعلى درجات التناغم في التصميم")
        elif 'أفقي' in semiotics['symmetry_type']:
            st.success("✨ **التفسير:** التماثل الأفقي يوحي بالهدوء والاستقرار والسكينة")
        elif 'عمودي' in semiotics['symmetry_type']:
            st.success("✨ **التفسير:** التماثل العمودي يرمز للقوة والارتفاع والطموح")
        elif 'جزئي' in semiotics['symmetry_type']:
            st.info("✨ **التفسير:** التماثل الجزئي يخلق توازناً ديناميكياً بين النظام والحرية")
        else:
            st.info("✨ **التفسير:** عدم التماثل يعكس الحرية والحركة والطبيعية")
        
        # العناصر الرمزية المستنتجة
        st.markdown("#### 🎨 العناصر الرمزية والدلالات الثقافية")
        
        if len(semiotics['symbolic_elements']) > 0:
            st.success(f"تم استنتاج {len(semiotics['symbolic_elements'])} عنصر رمزي")
            
            for symbol in semiotics['symbolic_elements']:
                with st.expander(f"🔸 {symbol['element']} ({symbol['count']} عنصر)", expanded=True):
                    st.markdown(f"""
                    **المعنى الرمزي:** {symbol['meaning']}
                    
                    **الدلالة الثقافية:** {symbol['cultural']}
                    """)
        else:
            st.info("📋 لا توجد عناصر رمزية واضحة في هذه الصورة")
        
        # التقييم السيميائي الشامل
        st.markdown("#### 🎓 التقييم الأكاديمي السيميائي")
        
        semiotic_assessment = []
        
        # تقييم الأشكال
        total_shapes = len(semiotics['circles']) + len(semiotics['triangles']) + len(semiotics['squares']) + len(semiotics['rectangles'])
        if total_shapes > 5:
            semiotic_assessment.append("✅ غنية بالأشكال الهندسية - تكوين رمزي قوي")
        elif total_shapes > 0:
            semiotic_assessment.append("✅ تحتوي على عناصر هندسية - رمزية واضحة")
        
        # تقييم التكرار
        if semiotics['patterns']['has_repetition']:
            semiotic_assessment.append(f"✅ نمط تكراري ({semiotics['patterns']['pattern_type']}) - إيقاع بصري منتظم")
        
        # تقييم التماثل
        if 'رباعي' in semiotics['symmetry_type'] or 'أفقي' in semiotics['symmetry_type']:
            semiotic_assessment.append("✅ تماثل قوي - توازن رمزي واضح")
        
        # تقييم نقاط التركيز
        if len(semiotics['focal_points']) > 0:
            semiotic_assessment.append(f"✅ {len(semiotics['focal_points'])} نقاط تركيز - هرمية بصرية واضحة")
        
        # تقييم الرموز
        if len(semiotics['symbolic_elements']) >= 3:
            semiotic_assessment.append("✅ متعددة الرموز - غنية بالدلالات الثقافية")
        elif len(semiotics['symbolic_elements']) > 0:
            semiotic_assessment.append("✅ تحتوي على رموز واضحة - دلالات ثقافية موجودة")
        
        # عرض التقييم
        if semiotic_assessment:
            st.markdown("**الخلاصة السيميائية:**")
            for assessment in semiotic_assessment:
                st.markdown(f"- {assessment}")
        else:
            st.info("الصورة ذات طابع طبيعي/عضوي بدون رمزية هندسية واضحة")
        
        # التقييم الإجمالي
        st.markdown("---")
        st.markdown("### 🎓 التقييم الأكاديمي الشامل")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"""
            <div class="analysis-box">
                <h3 style="color: #2c5282;">التقييم النهائي</h3>
                <h2 style="color: #c59849; text-align: center; font-size: 2.5rem; margin: 20px 0;">
                    {quality['overall_quality']}
                </h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("#### 📋 ملخص النتائج")
            st.success("✅ التحليل مكتمل")
            st.info(f"📊 تم تحليل {analyzer.pixels:,} بكسل")
        
        # التوصيات
        st.markdown("### 💡 التوصيات والاقتراحات")
        recommendations = []
        
        if quality['sharpness'] < 500:
            recommendations.append("🔸 تحسين الحدة: الصورة تحتاج إلى تحسين الحدة (Sharpening)")
        
        if quality['noise_level'] > 30:
            recommendations.append("🔸 تقليل التشويش: يُنصح بتطبيق فلتر تقليل التشويش (Noise Reduction)")
        
        if lighting['brightness'] < 80:
            recommendations.append("🔸 تحسين الإضاءة: الصورة مظلمة نسبياً، يُنصح بزيادة السطوع")
        elif lighting['brightness'] > 180:
            recommendations.append("🔸 تخفيف الإضاءة: الصورة ساطعة جداً، يُنصح بتقليل التعرض")
        
        if lighting['contrast'] < 40:
            recommendations.append("🔸 زيادة التباين: يُنصح بزيادة التباين لتحسين الوضوح")
        
        if color_analysis['avg_saturation'] < 30:
            recommendations.append("🔸 تحسين الألوان: الصورة باهتة، يُنصح بزيادة التشبع")
        
        if composition['thirds_score'] < 40:
            recommendations.append("🔸 تحسين التكوين: يُنصح بإعادة تأطير الصورة وفق قاعدة الأثلاث")
        
        # توصيات الخطوط
        if typography['has_text']:
            if typography['readability_score'] > 0 and typography['readability_score'] < 60:
                recommendations.append("🔸 تحسين وضوح الخط: الخطوط غير واضحة بما يكفي، يُنصح بزيادة الحدة أو استخدام خطوط أوضح")
            
            if typography['average_text_size'] > 0 and typography['average_text_size'] < 15:
                recommendations.append("🔸 زيادة حجم الخط: حجم الخط صغير جداً، يُنصح بزيادته لتحسين القابلية للقراءة")
            
            if '🔴' in typography['contrast_quality'] or '🟡' in typography['contrast_quality']:
                recommendations.append("🔸 تحسين تباين الخط: التباين بين النص والخلفية ضعيف، يُنصح باستخدام ألوان أكثر تبايناً")
            
            if typography['languages'] and 'عربي' in typography['languages']:
                if typography['readability_score'] < 70:
                    recommendations.append("🔸 الخط العربي: يُنصح باستخدام خطوط عربية واضحة مثل (Traditional Arabic، Tahoma، Arial)")
            
            if typography['languages'] and 'إنجليزي' in typography['languages']:
                if typography['readability_score'] < 70:
                    recommendations.append("🔸 الخط الإنجليزي: يُنصح باستخدام خطوط واضحة مثل (Arial، Helvetica، Calibri)")
        
        if recommendations:
            for rec in recommendations:
                st.warning(rec)
        else:
            st.success("✨ الصورة بجودة ممتازة ولا تحتاج إلى تحسينات كبيرة")
        
        # زر تنزيل التقرير
        st.markdown("---")
        
        # إعداد معلومات الخطوط للتقرير
        typography_info = ""
        if typography['has_text']:
            typography_info = f"""
        تحليل الخطوط والنصوص:
        - مناطق نصية: {typography['text_regions']}
        - اللغات المكتشفة: {', '.join(typography['languages']) if typography['languages'] else 'غير محدد'}
        - وضوح الخط: {typography['text_clarity']}
        - تباين الخط: {typography['contrast_quality']}
        - متوسط حجم الخط: {typography['average_text_size']:.1f} بكسل
        - درجة القابلية للقراءة: {typography['readability_score']:.1f}%"""
        else:
            typography_info = """
        تحليل الخطوط والنصوص:
        - لم يتم اكتشاف نصوص في الصورة"""
        
        # إعداد معلومات التحليل السيميائي
        semiotics_info = f"""
        التحليل السيميائي وكشف الرموز:
        - الدوائر: {len(semiotics['circles'])} (رمز الكمال والأبدية)
        - المثلثات: {len(semiotics['triangles'])} (رمز القوة والديناميكية)
        - المربعات: {len(semiotics['squares'])} (رمز الثبات والنظام)
        - المستطيلات: {len(semiotics['rectangles'])} (رمز التنظيم)
        - الخطوط المكتشفة: {semiotics['lines']['line_count']}
        - الاتجاه المهيمن: {semiotics['lines'].get('dominant_direction', 'غير محدد')}
        - نوع النمط: {semiotics['patterns']['pattern_type']}
        - شدة التكرار: {semiotics['patterns']['repetition_intensity']:.0f}%
        - نقاط التركيز: {len(semiotics['focal_points'])}
        - نوع التماثل: {semiotics['symmetry_type']}
        - العناصر الرمزية: {len(semiotics['symbolic_elements'])} عنصر"""
        
        if len(semiotics['symbolic_elements']) > 0:
            semiotics_info += "\n        \n        الرموز المكتشفة:"
            for symbol in semiotics['symbolic_elements'][:5]:
                semiotics_info += f"\n        - {symbol['element']}: {symbol['meaning']}"
        
        report = f"""
        تقرير التحليل الأكاديمي الشامل للصورة
        ================================================
        
        المعلومات الأساسية:
        - العرض: {basic_info['العرض']}
        - الارتفاع: {basic_info['الارتفاع']}
        - نسبة العرض للارتفاع: {basic_info['نسبة العرض للارتفاع']}
        - الدقة: {basic_info['تصنيف الدقة']}
        
        تحليل الألوان:
        - التشبع: {color_analysis['avg_saturation']:.1f}%
        - السطوع: {color_analysis['avg_brightness']:.1f}%
        - درجة الحرارة: {color_analysis['color_temperature']}
        
        تحليل الإضاءة:
        - السطوع: {lighting['brightness_category']}
        - التباين: {lighting['contrast_category']}
        - التعرض: {lighting['exposure_balance']}
        
        تحليل التكوين:
        - التماثل: {composition['symmetry_level']}
        - التعقيد: {composition['complexity_category']}
        - الحواف: {composition['edges_category']}
        
        تحليل الجودة:
        - الحدة: {quality['sharpness_category']}
        - التشويش: {quality['noise_category']}
        - التقييم الإجمالي: {quality['overall_quality']}
        {typography_info}
        {semiotics_info}
        
        التوصيات:
        {chr(10).join('- ' + r for r in recommendations) if recommendations else '- لا توجد توصيات، الصورة بجودة ممتازة'}
        
        ================================================
        بحث طالبة الدكتوراه: عبير قاسم حلف
        الجامعة المستنصرية - كلية التربية الأساسية
        ================================================
        تم إنشاء هذا التقرير بواسطة: محلل الصور الأكاديمي
        التاريخ: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        st.download_button(
            label="📥 تحميل التقرير الكامل",
            data=report,
            file_name="image_analysis_report.txt",
            mime="text/plain"
        )
    
    else:
        # رسالة الترحيب
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: white; 
                    border-radius: 15px; margin: 2rem 0;">
            <h2 style="color: #2c5282;">مرحباً بك في محلل الصور الأكاديمي! 👋</h2>
            <p style="font-size: 1.1rem; color: #4a5568; margin-top: 1rem;">
                ابدأ برفع صورة من الشريط الجانبي للحصول على تحليل شامل ومتقدم
            </p>
            <div style="margin-top: 2rem; padding: 2rem; background: #f7fafc; 
                        border-radius: 10px; border-right: 5px solid #c59849;">
                <h3 style="color: #2c5282; margin-bottom: 1rem;">ماذا يقدم التطبيق؟</h3>
                <div style="text-align: right;">
                    <p>📊 <strong>المعلومات الأساسية:</strong> الأبعاد، الدقة، النسب</p>
                    <p>🎨 <strong>تحليل الألوان:</strong> الألوان السائدة، التشبع، درجة الحرارة</p>
                    <p>💡 <strong>تحليل الإضاءة:</strong> السطوع، التباين، التعرض</p>
                    <p>🎭 <strong>تحليل التكوين:</strong> التماثل، قاعدة الأثلاث، التعقيد</p>
                    <p>⭐ <strong>تحليل الجودة:</strong> الحدة، التشويش، الوضوح</p>
                    <p>📝 <strong>تحليل الخطوط:</strong> كشف النصوص العربية والإنجليزية، وضوح الخط، التباين</p>
                    <p>🔮 <strong>التحليل السيميائي:</strong> كشف الرموز، الأشكال الهندسية، الدلالات الثقافية</p>
                    <p>💡 <strong>التوصيات:</strong> اقتراحات أكاديمية للتحسين</p>
                </div>
            </div>
            <div style="margin-top: 2rem; padding: 1.5rem;
                        background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
                        color: white; border-radius: 10px;
                        border-top: 3px solid #c59849;">
                <div style="font-size: 0.85rem; letter-spacing: 2px; color: #f0d78a;
                            font-weight: 700; margin-bottom: 0.6rem;">
                    بحث طالبة الدكتوراه
                </div>
                <div style="font-size: 1.4rem; font-weight: 700;">
                    عبير قاسم حلف
                </div>
                <div style="font-size: 1rem; margin-top: 0.6rem; opacity: 0.95;">
                    الجامعة المستنصرية &nbsp;•&nbsp; كلية التربية الأساسية
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
