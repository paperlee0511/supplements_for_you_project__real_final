from PyQt5.QtWidgets import QDialog
from PyQt5.QtCore import Qt
from Ui.ui_modal import Ui_Form
import webbrowser
import pandas as pd
import re

def safe_text(val):
    if pd.isna(val):
        return "정보 없음"
    s = str(val)
    s = s.replace('\n', ' ').replace('\r', ' ')
    s = re.sub(r'[^\x20-\x7E가-힣]', '', s)
    if len(s) > 200:
        s = s[:200] + "...(생략)"
    if s.strip().startswith("nan;"):
        return s.replace("nan;", "", 1).strip()
    if s.strip().lower() == "nan":
        return "정보 없음"
    return s.strip()

class ModalLikeWindow(QDialog, Ui_Form):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("추천 결과")
        self.current_url = ""
        self.products = []
        self.comboBox.currentIndexChanged.connect(self.update_product_info)
        self.pushButton_url.clicked.connect(self.open_url)

    def set_result(self, summary_text, products):
        print("set_result 진입")
        print("summary_text:", summary_text)
        print("products:", products)
        self.label.setPlainText(summary_text)
        self.label.setReadOnly(True)
        self.products = products
        self.comboBox.clear()
        for p in products:
            print("제품 추가:", p['product'])
            self.comboBox.addItem(p['product'])
        print("ComboBox 아이템 추가 완료")
        self.update_product_info(0)

    def update_product_info(self, index):
        if not self.products or index < 0 or index >= len(self.products):
            return
        product_info = self.products[index]
        self.label_2.setPlainText(str(product_info['ingredient']))
        self.label_2.setReadOnly(True)
        url = product_info.get('url', "")
        self.current_url = url.strip() if url else ""
        print("update_product_info, current_url:", self.current_url)  # 디버깅
        if self.current_url and self.current_url.lower() != "정보 없음":
            # 버튼에 url을 직접 표시!
            self.pushButton_url.setText(self.current_url)
            self.pushButton_url.setEnabled(True)
        else:
            self.pushButton_url.setText("URL 정보 없음")
            self.pushButton_url.setEnabled(False)

    def open_url(self):
        if self.current_url:
            import webbrowser
            webbrowser.open(self.current_url)

