import pandas as pd
import re
from konlpy.tag import Okt
import time
from multiprocessing import Pool, cpu_count
import numpy as np

# 1. 데이터 불러오기 및 컬럼 확인
df = pd.read_csv('./cleaned_data/supplements.csv', quotechar='"', encoding='utf-8-sig')

# 컬럼명 확인 및 정리
print("원본 데이터 구조:")
print(f"컬럼명: {df.columns.tolist()}")
print(f"데이터 형태: {df.shape}")
print(df.head())

# 컬럼명 표준화 (순서: supplements, product, ingredient, review, url)
expected_columns = ['supplements', 'product', 'ingredient', 'review', 'url']
if len(df.columns) == 5:
    df.columns = expected_columns
    print(f"\n컬럼명 표준화 완료: {df.columns.tolist()}")

# 데이터 타입 정리
for col in ['supplements', 'product', 'ingredient', 'review']:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()

print(f"\n데이터 정리 후:")
print(df.info())



# 2. 개선된 비타민/영양성분 패턴 정의
vitamin_patterns = [
    # 1. 영어 vitamin 패턴들 (숫자 포함)
    (re.compile(r'[Vv]itamin\s*([AaBbCcDdEeKk])(\d+)', re.IGNORECASE),
     lambda m: f'비타민{m.group(1).upper()}{m.group(2)}'),
    (re.compile(r'[Vv]itamin\s*([AaBbCcDdEeKk])', re.IGNORECASE),
     lambda m: f'비타민{m.group(1).upper()}'),

    # 2. 한글 음역 패턴들 (숫자 포함)
    (re.compile(r'비타민\s*비\s*(\d+)', re.IGNORECASE), r'비타민B\1'),
    (re.compile(r'비타민\s*비(\d+)', re.IGNORECASE), r'비타민B\1'),
    (re.compile(r'비타민\s*디\s*(\d+)', re.IGNORECASE), r'비타민D\1'),
    (re.compile(r'비타민\s*디(\d+)', re.IGNORECASE), r'비타민D\1'),

    # 3. 한글 음역어들
    (re.compile(r'비타민\s*비원', re.IGNORECASE), '비타민B1'),
    (re.compile(r'비타민\s*씨', re.IGNORECASE), '비타민C'),
    (re.compile(r'비타민\s*디(?!\d)', re.IGNORECASE), '비타민D'),
    (re.compile(r'비타민\s*에이', re.IGNORECASE), '비타민A'),
    (re.compile(r'비타민\s*이', re.IGNORECASE), '비타민E'),
    (re.compile(r'비타민\s*케이', re.IGNORECASE), '비타민K'),

    # 4. 일반적인 비타민 + 알파벳 패턴
    (re.compile(r'비타민\s*([a-kA-K])(\d*)', re.IGNORECASE),
     lambda m: f'비타민{m.group(1).upper()}{m.group(2)}'),

    # 5. 미네랄 및 기타 영양소 패턴
    (re.compile(r'마그네슘|마그네시움', re.IGNORECASE), '마그네슘'),
    (re.compile(r'칼슘|칼시움', re.IGNORECASE), '칼슘'),
    (re.compile(r'아연|징크', re.IGNORECASE), '아연'),
    (re.compile(r'철분|철', re.IGNORECASE), '철분'),
    (re.compile(r'오메가\s*3|omega\s*3', re.IGNORECASE), '오메가3'),
    (re.compile(r'오메가\s*6|omega\s*6', re.IGNORECASE), '오메가6'),
    (re.compile(r'dha|DHA', re.IGNORECASE), 'DHA'),
    (re.compile(r'epa|EPA', re.IGNORECASE), 'EPA'),
    (re.compile(r'코엔자임\s*q10|coenzyme\s*q10|코큐텐', re.IGNORECASE), '코엔자임Q10'),
    (re.compile(r'루테인', re.IGNORECASE), '루테인'),
    (re.compile(r'프로바이오틱스|유산균', re.IGNORECASE), '프로바이오틱스'),
    (re.compile(r'콜라겐', re.IGNORECASE), '콜라겐'),
    (re.compile(r'엽산|폴산', re.IGNORECASE), '엽산'),
    (re.compile(r'비오틴', re.IGNORECASE), '비오틴'),
]


def normalize_nutrients_comprehensive(text):
    """포괄적인 영양소 표기 정규화 함수"""
    if pd.isna(text) or text == '':
        return text

    text = str(text)

    # 기본 정규화 패턴 적용
    for pattern, replacement in vitamin_patterns:
        if callable(replacement):
            text = pattern.sub(replacement, text)
        else:
            text = pattern.sub(replacement, text)

    # 비타민 뒤의 알파벳 패턴 처리
    def process_vitamin_context(match):
        full_match = match.group(0)
        vitamin_part = re.sub(r'\b([a-k])\b', lambda m: m.group(1).upper(), full_match, flags=re.IGNORECASE)
        return vitamin_part

    text = re.sub(r'비타민\s+([a-k\s\d]+)', process_vitamin_context, text, flags=re.IGNORECASE)

    # 연속된 비타민 알파벳 + 숫자 조합 처리
    text = re.sub(r'비타민\s*([A-K])\s*([A-K])\s*(\d+)', r'비타민\1\2\3', text)
    text = re.sub(r'비타민\s*([A-K])\s*(\d+)', r'비타민\1\2', text)
    text = re.sub(r'비타민\s*([A-K])', r'비타민\1', text)

    # b6, b12, d3, 등 알파벳+숫자 패턴 후처리
    text = re.sub(r'\b([abcdekl])(\d{1,2})\b', lambda m: f'비타민{m.group(1).upper()}{m.group(2)}', text)

    text = re.sub(r'vitamin\s*([a-zA-Z]\d*)', lambda m: f'비타민{m.group(1).upper()}', text, flags=re.IGNORECASE)
    text = re.sub(r'비타민\s*([a-zA-Z]\d*)', lambda m: f'비타민{m.group(1).upper()}', text)

    # 단독 알파벳 또는 b군 비타민이 소문자로만 등장할 경우 대문자로 변환
    # 예: 'a', 'b2', 'e', 'd3' → 'A', 'B2', ...
    def fix_single_vitamin(match):
        val = match.group(1)
        return val.upper()

    # 단어 경계에서 a~k 한글자 또는 b군 조합
    text = re.sub(r'\b([a-df-hj-kA-DF-HJ-K])\b', fix_single_vitamin, text)  # 단일 알파벳 대문자화
    text = re.sub(r'\b(b[1-9]\d?)\b', fix_single_vitamin, text, flags=re.IGNORECASE)  # b1~b12 등

    return text

    return text

brand_keywords = [
    # 한글 브랜드
    '센트룸', '솔가', '나우푸드', '닥터스베스트', '뉴트리라이트', '스완슨',
    '라이프익스텐션', '내추럴팩터스', '네이처메이드', '칼슨랩스', '오쏘몰',

    # 영문 브랜드
    'Centrum', 'Solgar', 'Now Foods', 'Doctor\'s Best', 'Nutrilite', 'Swanson',
    'Life Extension', 'Natural Factors', 'Nature Made', 'Carlson Labs', 'Orthomol',

    # 기타
    'GNC', '21st Century', 'Jarrow', 'California Gold Nutrition', 'Nature\'s Way'
]

def normalize_product_name(text):
    """제품명 전용 정규화 + 브랜드 제거"""
    if pd.isna(text) or text == '':
        return text

    text = str(text)
    text = normalize_nutrients_comprehensive(text)

    # 브랜드 제거
    for brand in brand_keywords:
        text = re.sub(re.escape(brand), '', text, flags=re.IGNORECASE)

    # 특수 문자 제거 및 다중 공백 정리
    text = re.sub(r'[^\w\s가-힣A-Za-z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    # 비타민 단어 보정
    words = text.split()
    processed_words = []
    for i, word in enumerate(words):
        if i > 0 and ('비타민' in words[i - 1] or '비타민' in word):
            if word.lower() in ['a', 'b', 'c', 'd', 'e', 'k']:
                processed_words.append(word.upper())
            elif re.match(r'^[a-k]\d*$', word, re.IGNORECASE):
                processed_words.append(word.upper())
            else:
                processed_words.append(word)
        else:
            processed_words.append(word)

    text = ' '.join(processed_words)
    text = re.sub(r'비타민\s+([A-K]\d*)', r'비타민\1', text)

    return text


# 3. 제품명과 영양성분 정규화 적용
print("\n=== 제품명 및 영양성분 정규화 ===")
print("정규화 전 제품명 샘플:")
print(df['product'].head())

df['cleaned_product'] = df['product'].apply(normalize_product_name)
df['ingredient'] = df['ingredient'].apply(normalize_nutrients_comprehensive)

print("\n정규화 후 제품명 샘플:")
print(df['product'].head())

print("\n정규화 후 영양성분 샘플:")
print(df['ingredient'].head())

# 4. 영양제 종류 정제 (첫 번째 컬럼)
print("\n=== 영양제 종류 정제 ===")
print("정제 전:", df['supplements'].unique())

df['supplements'] = df['supplements'].str.replace(' ', '', regex=False)

# 명칭 통일
supplement_mapping = {
    '남성용비타민': '남성종합비타민',
    '남성용종합비타민': '남성종합비타민',
    '여성용비타민': '여성종합비타민',
    '여성용종합비타민': '여성종합비타민',
    '멀티비타민': '종합비타민',
    '남성멀티비타민': '남성종합비타민',
    '여성멀티비타민': '여성종합비타민',
    '임산부종합비타민': '임산부종합비타민',
}

df['supplements'] = df['supplements'].replace(supplement_mapping)
print("정제 후:", df['supplements'].unique())


# 5. 제품명과 영양성분을 리뷰에 증폭 (기존 500배 대신 개선된 방식)
def amplify_nutrients_in_review(df, repeat_product= 500):
    """제품명과 영양성분을 리뷰에 증폭"""
    df = df.copy()

    print(f"리뷰 증폭 시작: 제품명 {repeat_product}회")

    def create_amplified_review(row):
        # 정규화된 제품명과 영양성분을 반복해서 추가
        product_amplified = (str(row['product']) + ' ') * repeat_product
        original_review = str(row['review'])

        return product_amplified + original_review

    df['review'] = df.apply(create_amplified_review, axis=1)
    return df


df = amplify_nutrients_in_review(df, repeat_product=500)

# 6. 불용어 및 보존 단어 설정
stop_words = set([
    # 일반 동사/형용사
    '하다', '되다', '있다', '없다', '같다', '보다', '주다', '먹다', '좋다', '나쁘다',
    '많다', '적다', '크다', '작다', '들다', '꾸준하다', '훌륭하다', '탁월하다',
    '만족하다', '넘다', '들어서다', '안되다', '좋아하다', '되어다', '깔끔하다',
    '해보다', '해주다', '먹이다', '먹어주다', '아니다', '재다', '넘어가다', '사다',
    '빠르다', '켜지다', '다만', '비싸다', '기다리다', '특별하다', '꼼꼼하다',
    '이다', '어렵다', '않다', '먹기', '부분', '챙기다', '맛있다',
    # 명사
    '제품', '영양제', '구매', '구입', '사용', '가격', '포장', '배송', '리뷰',
    '후기', '품절', '유기농', '직구', '함량', '냄새', '부담', '구미', '젤리',
    # 수량, 시간, 기타
    '하루', '개월', '이상', '정도', '전', '후', '지금', '조금', '다시', '계속',
    '처음', '최근', '현재', '매일', '늘', '함께',
    # 신체/감정 추상어
    '느낌', '기분', '만족', '불만',
])

preserve_words = set([
    # 비타민 관련
    'A', 'B', 'C', 'D', 'E', 'K', 'a', 'b', 'c', 'd', 'e', 'k',
    '비타민A', '비타민B', '비타민C', '비타민D', '비타민E', '비타민K',
    '비타민B1', '비타민B2', '비타민B6', '비타민B12', '비타민D3',
    # 미네랄
    '마그네슘', '칼슘', '아연', '철분', '셀레늄', '크롬', '망간',
    # 기타 영양소
    '오메가3', '오메가6', 'DHA', 'EPA', '코엔자임Q10', '루테인',
    '프로바이오틱스', '콜라겐', '엽산', '비오틴',
    # 신체 부위
    '손', '발', '목', '몸', '눈', '팔', '간', '장', '뇌', '뼈', '귀', '코', '위', '폐', '피',
    # 숫자
    '1', '2', '3', '6', '12',
])

# 7. 리뷰 전처리 함수
special_char_pattern = re.compile('[^가-힣A-Za-z0-9]')


def process_reviews_batch(reviews_batch):
    """배치 단위로 리뷰 처리"""
    results = []
    okt = Okt()

    for idx, review in reviews_batch:
        if pd.isna(review):
            results.append((idx, ''))
            continue

        try:
            review = str(review)
            review = re.sub(r'[|,]', ' ', review)

            # 영양소 정규화 (소문자 변환 전)
            review = normalize_nutrients_comprehensive(review)
            review = review.lower()

            # 소문자 변환 후 다시 영양소 정규화
            review = normalize_nutrients_comprehensive(review)
            review = special_char_pattern.sub(' ', review)

            if not review.strip():
                results.append((idx, ''))
                continue

            tokens = okt.pos(review, stem=True)
            words = []

            for w, cls in tokens:
                w_normalized = normalize_nutrients_comprehensive(w)

                # 보존 단어 체크
                if w_normalized in preserve_words:
                    words.append(w_normalized)
                # 영양소 패턴 체크
                elif re.match(r'(비타민[A-K]\d*|마그네슘|칼슘|아연|철분|오메가\d|DHA|EPA)', w_normalized, re.IGNORECASE):
                    words.append(w_normalized)
                # Alpha 클래스 처리
                elif cls == 'Alpha':
                    if w.lower() in ['a', 'b', 'c', 'd', 'e', 'k']:
                        words.append(w.upper())
                    continue
                # 숫자 처리
                elif cls == 'Number' and w in ['1', '2', '3', '6', '12']:
                    words.append(w)
                # 일반 단어 처리
                elif cls in ['Noun', 'Adjective', 'Verb'] and len(w) > 1 and w not in stop_words:
                    words.append(w)

            cleaned = ' '.join(words)
            results.append((idx, cleaned))

            if idx % 50 == 0:
                print(f"배치 처리 중... {idx}")

        except Exception as e:
            print(f"오류 발생 (인덱스 {idx}): {e}")
            results.append((idx, ''))

    return results


# 8. 메인 처리 로직
print(f"\n총 {len(df)}개의 리뷰 처리 시작...")
start_time = time.time()

batch_size = 50
total_reviews = len(df)
batches = []

# 배치 생성
for i in range(0, total_reviews, batch_size):
    batch = [(idx, review) for idx, review in enumerate(df['review'].iloc[i:i + batch_size], i)]
    batches.append(batch)

print(f"총 {len(batches)}개 배치로 분할")

# 배치별 처리
all_results = []
for batch_idx, batch in enumerate(batches):
    print(f"\n배치 {batch_idx + 1}/{len(batches)} 처리 중...")
    batch_results = process_reviews_batch(batch)
    all_results.extend(batch_results)

    progress = (batch_idx + 1) / len(batches) * 100
    elapsed_time = time.time() - start_time
    print(f"진행률: {progress:.1f}% | 경과시간: {elapsed_time:.1f}초")

# 결과 적용
all_results.sort(key=lambda x: x[0])
cleaned_reviews = [result[1] for result in all_results]

# 9. 최종 데이터 구성 (기존 컬럼 유지, 리뷰만 정제된 것으로 교체)
final_df = df.copy()
final_df['review'] = cleaned_reviews

# 전처리된 제품명을 별도 컬럼으로 보관
final_df['cleaned_product'] = final_df['product'].apply(normalize_product_name)

# 중복 제거 (리뷰 기준)
print(f"\n중복 제거 전: {len(final_df)}개")
final_df.drop_duplicates(subset=['review'], inplace=True)
print(f"중복 제거 후: {len(final_df)}개")

# 새로운 컬럼 순서로 정리
column_order = ['supplements', 'product', 'ingredient', 'review', 'url', 'cleaned_product']
final_df = final_df[column_order]
# 10. 저장
output_path = './cleaned_data/cleaned_supplements.csv'
final_df.to_csv(output_path, index=False, encoding='utf-8-sig')

end_time = time.time()
total_time = end_time - start_time
print(f"\n리뷰 전처리 완료!")
print(f"총 소요시간: {total_time:.1f}초")
print(f"평균 처리속도: {len(final_df) / total_time:.2f}개/초")
print(f"저장 경로: {output_path}")

# 11. 결과 확인
print("\n=== 최종 결과 확인 ===")
print(f"최종 데이터 형태: {final_df.shape}")
print(f"컬럼 순서: {final_df.columns.tolist()}")
print("\n샘플 데이터:")
print(final_df.head(3).to_string())

# 12. 영양소 정규화 테스트
print("\n=== 영양소 정규화 테스트 ===")
# test_cases = [
#     "면역 비타민 C d 3 아연 캡슐",
#     "마그네슘 비타민 B 복합체",
#     "오메가3 DHA EPA 루테인",
#     "코엔자임 q10 콜라겐 비오틴",
#     "종합비타민 미네랄 프로바이오틱스",
# ]

test_cases = [
    "vitamin c와 d3를 함께 먹고 있어요",
    "요즘 B1, b2, B6, b12를 챙겨요",
    "a랑 e가 피부에 좋다길래 시작했어요",
    "비타민 D와 K2는 같이 먹는 게 좋대",
    "b complex 중에 b6하고 b12가 중요하다더라",
    "d3를 먹으면서 칼슘 흡수를 돕고 있어요",
    "c랑 e는 항산화 효과가 있어서 챙기고 있어",
    "B2 b3 같이 먹으니 피로가 덜해요",
    "a d는 눈이랑 뼈 건강에 좋다고 들었어요",
    "E랑 k를 같이 복용하면 피부에 도움이 된대요",
]

for test in test_cases:
    normalized = normalize_nutrients_comprehensive(test)
    print(f"'{test}' → '{normalized}'")

print("\n전처리 완료!")