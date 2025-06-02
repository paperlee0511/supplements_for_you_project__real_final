import sys
import pandas as pd
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon
from PyQt5 import uic
from supplement_recommendation_app2 import ModalLikeWindow
from recommendation import preprocess_data_optimized, recommend_products_hybrid
from gensim.models import Word2Vec

form_window = uic.loadUiType('./Ui/supplement_recommendation_app.ui')[0]

def keyword_score(text, keywords):
    return sum(text.lower().count(kw.lower()) for kw in keywords)

class SupplementApp(QMainWindow, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        print("CSV Load 시도")
        # Word2Vec 모델 불러오기 (경로는 실제 위치로!)
        self.embedding_model = Word2Vec.load('models/word2vec_supplements_okt.model')
        self.products_df = pd.read_csv('models/tfidf_products.csv')
        self.products_df['ingredient'] = self.products_df['ingredient'].fillna("정보 없음")
        # 아래 한 줄이 중요! (토큰화, 추천시 필요)
        self.products_df = preprocess_data_optimized(self.products_df)
        print("CSV Loaded:", self.products_df.shape)

        self.pushButton.clicked.connect(self.show_modal)
        self.pushButton_2.clicked.connect(self.reset_inputs)

    def reset_inputs(self):
        self.lineEdit.clear()
        for cb in self.findChildren(QCheckBox):
            cb.setChecked(False)

    def show_modal(self):
        print("show_modal 진입")
        search_word = self.lineEdit.text().strip()
        checked_names = []
        for i in range(1, 104):
            checkbox = getattr(self, f'checkBox_{i}', None)
            if checkbox and checkbox.isChecked():
                checked_names.append(checkbox.text())

        # 검색어 + 체크박스 조합
        if search_word:
            symptom_query = search_word
        elif checked_names:
            symptom_query = ','.join(checked_names)
        else:
            symptom_query = ""

        # Word2Vec 추천 함수 호출
        if symptom_query:
            df_result = recommend_products_hybrid(
                self.products_df, symptom_query, self.embedding_model, topn=10
            )
            # 확장 키워드 표시(옵션)
            expanded_tokens = df_result.attrs.get("expanded_tokens", symptom_query)
            symptom = symptom_query
            keywords = expanded_tokens
        else:
            df_result = pd.DataFrame()
            symptom = "추천 결과 없음"
            keywords = []

        # 제품 리스트 생성
        products = []
        for i, row in df_result.iterrows():
            products.append({
                'product': row['product'],
                'score': row['temp_score'] if 'temp_score' in row else 0,
                'ingredient': row['ingredient'],
                'url': row['url']
            })

        # summary_text 생성
        if not df_result.empty:
            lines = []
            lines.append(f"✅ [{symptom}] 추천 결과:")
            lines.append(f"🔍 확장된 키워드: {keywords}")
            for p in products:
                lines.append(f"- {p['product']} (점수: {p['score']})")
            summary_text = '\n'.join(lines)
            ingredient_info = str(df_result.iloc[0]['ingredient'])
            url = str(df_result.iloc[0]['url'])
        else:
            summary_text = "검색 결과 없음"
            ingredient_info = ""
            url = ""

        print("products 리스트:", products)
        print("summary_text, ingredient_info, url 생성 완료")

        dialog = ModalLikeWindow(self)
        print("ModalLikeWindow 생성 완료")
        dialog.set_result(summary_text, products)
        print("set_result 호출 완료")
        dialog.exec_()
        print("exec_() 호출 완료")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SupplementApp()
    window.show()
    sys.exit(app.exec_())



#되는 코드
# import sys
# import pandas as pd
# from PyQt5.QtWidgets import *
# from PyQt5.QtCore import Qt
# from PyQt5.QtGui import QIcon
# from PyQt5 import uic
# from supplement_recommendation_app2 import ModalLikeWindow
#
# form_window = uic.loadUiType('./Ui/supplement_recommendation_app.ui')[0]
#
# class SupplementApp(QMainWindow, form_window):
#     def __init__(self):
#         super().__init__()
#         self.setupUi(self)
#
#         print("CSV Load 시도")
#         self.products_df = pd.read_csv('models/tfidf_products.csv')
#         # ← 여기 바로 아래에 넣어주시면 됩니다!
#         self.products_df['ingredient'] = self.products_df['ingredient'].fillna("정보 없음")
#         print("CSV Loaded:", self.products_df.shape)
#
#         self.pushButton.clicked.connect(self.show_modal)
#
#     def show_modal(self):
#         print("show_modal 진입")
#         search_word = self.lineEdit.text().strip()
#         checked_names = []
#         for i in range(1, 104):
#             checkbox = getattr(self, f'checkBox_{i}', None)
#             if checkbox and checkbox.isChecked():
#                 checked_names.append(checkbox.text())
#         print("검색어/체크박스 추출 완료")
#
#         if search_word:
#             df_result = self.products_df[
#                 self.products_df['review'].astype(str).str.contains(search_word, na=False, case=False) |
#                 self.products_df['product'].astype(str).str.contains(search_word, na=False, case=False)
#             ]
#             symptom = search_word  # 검색어를 증상명으로 사용
#         elif checked_names:
#             import re
#             query = '|'.join([re.escape(kw) for kw in checked_names])
#             df_result = self.products_df[
#                 self.products_df['review'].astype(str).str.contains(query, na=False, case=False) |
#                 self.products_df['product'].astype(str).str.contains(query, na=False, case=False)
#             ]
#             symptom = ', '.join(checked_names)  # 체크박스 선택값을 증상명으로 사용
#         else:
#             df_result = self.products_df.iloc[0:0]
#             symptom = "추천 결과 없음"
#
#         # 예시 키워드(실제 키워드 추출 로직이 있다면 거기서 받아오세요)
#         keywords = "키워드 예시"  # 실제 확장 키워드로 변경
#         # 추천 제품 리스트
#         products = []
#         for i, row in df_result.head(7).iterrows():
#             products.append({
#                 'product': row['product'],
#                 'score': 10,  # 필요시 점수 로직
#                 'ingredient': row['ingredient'],  # ★ 반드시 포함!
#                 'url': row['url']  # ★ 반드시 포함!
#             })
#
#         # summary_text 생성
#         if not df_result.empty:
#             lines = []
#             lines.append(f"✅ [{symptom}] 추천 결과:")
#             lines.append(f"🔍 확장된 키워드: {keywords}")
#             for p in products:
#                 lines.append(f"- {p['product']} (점수: {p['score']})")
#             summary_text = '\n'.join(lines)
#
#             ingredient_info = str(df_result.iloc[0]['ingredient'])
#             url = str(df_result.iloc[0]['url'])
#         else:
#             summary_text = "검색 결과 없음"
#             ingredient_info = ""
#             url = ""
#
#         print("products 리스트:", products)
#         print("summary_text, ingredient_info, url 생성 완료")
#
#         dialog = ModalLikeWindow(self)
#         print("ModalLikeWindow 생성 완료")
#         dialog.set_result(summary_text, products)
#         print("set_result 호출 완료")
#         dialog.exec_()
#         print("exec_() 호출 완료")
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = SupplementApp()
#     window.show()
#     sys.exit(app.exec_())

#그전에 내가 Qt Designer에서 리셋버튼을 만들어 보려고 하는데

# import sys
# import pandas as pd
# from PyQt5.QtWidgets import *
# from PyQt5.QtCore import Qt
# from PyQt5.QtGui import QIcon
# from PyQt5 import uic
# from supplement_recommendation_app2 import ModalLikeWindow
#
# form_window = uic.loadUiType('./Ui/supplement_recommendation_app.ui')[0]
#
# class SupplementApp(QMainWindow, form_window):
#     def __init__(self):
#         super().__init__()
#         self.setupUi(self)
#
#         print("CSV Load 시도")
#         self.products_df = pd.read_csv('models/tfidf_products.csv')
#         # ← 여기 바로 아래에 넣어주시면 됩니다!
#         self.products_df['ingredient'] = self.products_df['ingredient'].fillna("정보 없음")
#         print("CSV Loaded:", self.products_df.shape)
#
#         self.pushButton.clicked.connect(self.show_modal)
#
#     def show_modal(self):
#         search_word = self.lineEdit.text().strip()
#         checked_names_review = []
#         checked_names_ingredient = []
#         for i in range(1, 104):  # 전체 체크박스
#             checkbox = getattr(self, f'checkBox_{i}', None)
#             if checkbox and checkbox.isChecked():
#                 # 3~17번만 영양성분 기준
#                 if 3 <= i <= 17:
#                     checked_names_ingredient.append(checkbox.text())
#                 else:
#                     checked_names_review.append(checkbox.text())
#
#         import re
#
#         # 기본값
#         symptom = ""
#         df_result = self.products_df.iloc[0:0]  # 빈 결과
#
#         # 1. 증상 검색어(검색창)이 있는 경우
#         if search_word:
#             df_result = self.products_df[
#                 self.products_df['review'].astype(str).str.contains(search_word, na=False, case=False) |
#                 self.products_df['product'].astype(str).str.contains(search_word, na=False, case=False)
#                 ]
#             symptom = search_word
#
#         # 2. 체크박스(영양성분 OR 리뷰/제품명)로만 검색하는 경우
#         else:
#             mask = pd.Series([False] * len(self.products_df))
#
#             # 2-1. 리뷰/제품명 기반
#             if checked_names_review:
#                 query = '|'.join([re.escape(kw) for kw in checked_names_review])
#                 mask_review = (
#                         self.products_df['review'].astype(str).str.contains(query, na=False, case=False) |
#                         self.products_df['product'].astype(str).str.contains(query, na=False, case=False)
#                 )
#                 mask = mask | mask_review  # OR 조건 누적
#
#             # 2-2. 영양성분 기반
#             if checked_names_ingredient:
#                 query_ing = '|'.join([re.escape(kw) for kw in checked_names_ingredient])
#                 mask_ing = self.products_df['ingredient'].astype(str).str.contains(query_ing, na=False, case=False)
#                 mask = mask | mask_ing  # OR 조건 누적
#
#             df_result = self.products_df[mask]
#             # 증상 이름
#             symptom = ', '.join(checked_names_review + checked_names_ingredient) if (
#                         checked_names_review or checked_names_ingredient) else "추천 결과 없음"
#
# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     window = SupplementApp()
#     window.show()
#     sys.exit(app.exec_())