# 💊 자연어 처리를 통한 리뷰 기반 영양제 추천

## 시연 영상



## 개요
영양제 구매 시 제품이름이나 성분에 대한 정보 부족으로 선택이 어려움
사용자 증상에 기반한 자연어 처리 추천 시스템 구축
제품 리뷰 데이터를 학습하여 증상과 유사한 의미를 가진 영양제를 추천한다.

- 팀명 : 깻잎
- 팀장 : 이종희
- 팀원 : 이윤성, 안진홍

### 역할
  - 웹크롤링, 데이터전처리(CSV파일 병합 및 표준화)
  - 텍스트 전처리(형태소 분석, 불용어 제거, 비타민 표기 정규화)
  - TF-IDF 벡터화, Word2Vec

## 프로젝트 핵심 기능
  - 자동데이터 수집 : IHerb 사이트에서 건강보조식품 정보 및 리뷰 크롤링
  - 사용자 증상 입력 시 유사한 리뷰를 가진 제품 추천
  - TF-IDF 코사인 유사도 기반 맞춤형 제품 추천

## 사용 기법
### 자연어 처리
  - KoNLPy(Okt) : 한국어 형태소 분석
  - 정규표현식 : 비타민 표기 정규화(vitamin c -> 비타민C)
  - 불용어 탐지 : IDF값 기반 불용어 추출
 
### 머신러닝
- TF-IDF 벡터화 : 문서-단어 중요도 행렬 생성
- Word2Vec(Skip-gram) : 단어 의미 임베딩 학습
- 코사인 유사도 : 제품간 유사성 계산

### word cloud
<img width="400" alt="image" src="https://github.com/user-attachments/assets/ed240d93-d81b-4dd3-a4a5-425bed72fa9a" />
종합비타민

<img width="400" alt="image" src="https://github.com/user-attachments/assets/c23ea3af-bf85-4ae2-97ea-4afc0fbfb23b" />
칼슘

<img width="400" alt="image" src="https://github.com/user-attachments/assets/bfa0d53d-3789-48a5-b796-99b7c1bfdb7c" />
비타민D

<img width="400" alt="image" src="https://github.com/user-attachments/assets/5d4cb0fb-540c-42fa-8f9d-6e753a6bcac5" />
아연

### 벡터 시각화
<img width="400" alt="image" src="https://github.com/user-attachments/assets/b40d9a5e-2860-422a-9688-2558c22d3573" />



## 데이터셋 정보
-  수집 대상 : iHerb사이트
-  카테고리 : 11개(종합비타민, 비타민A~E, 아연, 셀레늄 등)
-  수집 규모 : 48개 제품 x 최대 50개 페이지 리뷰
-  데이터 형태 : 제품명, 영양성분, 고객 리뷰, URL



  

