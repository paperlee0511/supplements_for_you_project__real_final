import pandas as pd

# 파일 경로
stopwords_path = "./models/auto_stopwords_candidates.csv"

# 유지할 단어들
keywords_to_keep = set([
    '추천', '건강', '감소', '증가', '균형', '긍정', '건강 유지', '효과 느끼다', '작용', '이점', '여성',
    '수치', '비타민 섭취', '좋다', '효과품질', '스트레스', '비타민 복용', '종합비타민', '항산화제',
    '보호', '수면', '엄마', '오메가', '손톱', '증진', '영향 미치다', '향상 시키다', '체내', '유지 도움',
    '치료', '머리카락', '유익하다', '항산화', '복용', '강화하다', '대사', '다이어트', '촉진', '완화',
    '세포', '비타민 성분', '건강 개선', '임신', '신경계', '자극', '칼슘', '근육', '유발', '철분',
    '염증', '면역 체계', '성분 효과', '결핍 보충', '부모님', '감기', '질환', '어린이', '비타민 보충',
    '호르몬', '성능', '품질 비타민', '알레르기', '비타민 비타민', '모발', '질병', '효과 나타나다',
    '관절', '머리', '통증'
])

# 데이터 불러오기
df_stop = pd.read_csv(stopwords_path)

# 제거할 불용어만 추출
df_stop_filtered = df_stop[~df_stop['word'].isin(keywords_to_keep)].copy()

# 파일로 저장
filtered_path = "./models/filtered_stopwords.csv"
df_stop_filtered.to_csv(filtered_path, index=False, encoding="utf-8-sig")
