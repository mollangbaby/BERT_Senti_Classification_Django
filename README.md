# BERT_Senti_Classification_Django
## 🤷‍♀️ 프로젝트 설명 
    * 자세한 내용은 model/templates 폴더의 data.html, info.html 참고

감정 분석 장고 웹앱
- flask, mlflow 등을 사용하지 않고 django 프로젝트 내부에 모델을 등록하여 serving 
- info 페이지에서 모델정보, data 페이지에서 상세 데이터(데이터전처리 및 설계) 정보 확인 가능
- train code : Beomi_kcbert_base_(Tensorflow)TrainValidCode.ipynb
- input 페이지에서 감정을 분석하고자 하는 문장을 입력하면, predict 페이지에서 모델이 추론한 감정을 보여줌 


모델
- Beomi-KcBert-base 모델 (5개의 Bert모델 중 가장 좋은 성능 기록)
- tensorflow에서 학습(huggingface의 tfbertclassification을 사용하고자)


데이터셋 
- 학습 데이터셋 : `20만개` (AI hub 데이터셋 2개 수집 )
    - 감정 정보 제공
    - 수집한 두 개의 데이터셋은 다른 감정 분류
    - 감정 분류를 통일하는 작업 진행 


- 모델 일반화 테스트 데이터셋 : `1024개`
    - 국립국어원 모두의 말뭉치 - 일기 데이터셋
    - 감정 정보가 제공되지 않으며, 일기 텍스트 데이터만 제공됨
    - 학습용 데이터셋으로 사용하기엔 무리가 있어서 모델 일반화 테스트 데이터로 사용

감정 분류
- 로버트 플루칙의 이론에 근거하여 감정을 8개로 분류
- 분노, 평온, 기대, 혐오, 기쁨, 슬픔, 충격, 두려움 (angry, calmness, expect, loathing, pleasure, sadness, shock, terror)
- 데이터셋의 80%의 경우 직접 데이터 확인하여 전처리
- 일부 데이터 허깅페이스 파이프라인 사용하여 `오토라벨링` 후 `정교화` 작업 수행
- `클래스 불균형`이 심한 데이터셋 > `데이터 증강`을 통한 클래스 불균형 해소


# 💁‍♀️ 페이지 
### home.html 
![home](https://user-images.githubusercontent.com/119670827/221406899-dd8197e3-2952-4b55-933a-61980d4812ca.png)

### data.html
![data](https://user-images.githubusercontent.com/119670827/221406913-28ab8326-3e90-4512-b43a-575543c22790.png)

### info.html
![info](https://user-images.githubusercontent.com/119670827/221406906-c1ea1320-9eaf-4000-9611-f996982f5160.png)

### code.html 
![code2](https://user-images.githubusercontent.com/119670827/221406917-37b96ee5-06fc-43c0-b44b-8181c940c1cd.png)

### input.html 
![input](https://user-images.githubusercontent.com/119670827/221406924-21c91446-4177-4c8b-a840-3fbec55c42b0.png)

### predict.html 
![predict](https://user-images.githubusercontent.com/119670827/221406928-c59b6685-95d3-4f3b-8c0f-041eb2a7fca0.png)


# 사용 방법
(1) 모델 파일 다운로드
- https://drive.google.com/drive/folders/10LA8mzUOMkcFcoCImyNlL_3_8jxQjTrv?usp=sharing 공유 드라이브 이동하여 DT5_beomi_kcbert_base_model.h5 모델 파일 다운로드
- 다운로드한 모델 파일을 장고 프로젝트 내부로 이동 
- 장고 base project인 Diary의 settings.py로 이동

(2) Diary > settings.py 이동 
- settings.py 상단의 related bert import
- settings.py의 맨 하단의 각주 처리된 코드 > 각주 해제 
```python
# BEST_MODEL_NAME = 'C:\django\DL_senti_classification\DT5_beomi_kcbert_base_model.h5'
# # 최고 성능의 모델 불러오기
# MODEL = tf.keras.models.load_model(BEST_MODEL_NAME,
#                                                   custom_objects = {'TFBertForSequenceClassification': TFBertForSequenceClassification})
# # tokenizer 
# TOKENIZER = BertTokenizer.from_pretrained("beomi/kcbert-base")
# print("##### settings loaded >> tokenizer ")
# print(TOKENIZER)

# ENCODER  = LabelEncoder()
# labels = pd.read_csv(r"C:\django\DL_senti_classification\labels.csv")
# ENCODER.fit_transform(np.array(labels['감정'])) 
```

- 각주 해제 후 다운로드하여 이동시킨 모델의 절대 경로를 BERT_MODEL_NAME 상수 변수에 재할당
- labels변수에 labels.csv 파일의 절대경로 재할당
    - encoding 용도

(3) 장고 프로젝트의 하위 앱 model >  views.py 이동
- 각주 처리된 convert_data , bertPredict 함수 각주 해제
- 맨 하단의 predict 함수형뷰 코드로 이동하여 각주 처리된 부분의 코드 각주해제
- runserver 용도 코드 각주 밑 코드 context = {'input_submit' : input_submit} 삭제 


```python
def predict(request):
    start = time.time()
    if request.method == 'POST' :
        form = request.POST
        input_submit = form['input_diary']
        print("input_submit > ", input_submit, len(input_submit))
        pred = bertPredict(input_submit)
        context = {'input_submit' : pred}
        
        # runserver 용도 코드 
        # context = {'input_submit' : input_submit} >> ******* 삭제 

    end = time.time()
    print("총 소요시간 >> ", end-start )
    return render(request, 'model/predict.html', context)
```

(4) runserver 실행
