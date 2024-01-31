Side Project - License Plate Recognition
====
### 번호판 인식 사이드 프로젝트
#### 목표 : 영상의 자동차 번호판을 인식하여 그 내용을 영상에 표시해줌
## Codes
* main.py
  + YOLO model과 [SORT](https://github.com/Jh-jaehyuk/ObjectDetection/tree/main/EasyOCR/sort)를 불러와 영상의 번호판 바운딩박스를 입력받음.
* util.py
  + main.py에서 입력받은 바운딩박스를 기준으로 EasyOCR를 이용하여 문자를 인식하여 csv로 저장.
* add_missing_data.py
  + util.py에서 저장된 csv에서 비어있는 프레임의 값을 보간법을 이용하여 보완해줌.
* visualize.py
  + add_missing_data.py에서 보완된 csv를 기준으로 영상에 번호판과 그 내용을 표시해줌.
