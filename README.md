# BERT_Senti_Classification_Django
## ๐คทโโ๏ธ ํ๋ก์ ํธ ์ค๋ช 
    * ์์ธํ ๋ด์ฉ์ model/templates ํด๋์ data.html, info.html ์ฐธ๊ณ 

๊ฐ์  ๋ถ์ ์ฅ๊ณ  ์น์ฑ
- flask, mlflow ๋ฑ์ ์ฌ์ฉํ์ง ์๊ณ  django ํ๋ก์ ํธ ๋ด๋ถ์ ๋ชจ๋ธ์ ๋ฑ๋กํ์ฌ serving 
- info ํ์ด์ง์์ ๋ชจ๋ธ์ ๋ณด, data ํ์ด์ง์์ ์์ธ ๋ฐ์ดํฐ(๋ฐ์ดํฐ์ ์ฒ๋ฆฌ ๋ฐ ์ค๊ณ) ์ ๋ณด ํ์ธ ๊ฐ๋ฅ
- train code : Beomi_kcbert_base_(Tensorflow)TrainValidCode.ipynb
- input ํ์ด์ง์์ ๊ฐ์ ์ ๋ถ์ํ๊ณ ์ ํ๋ ๋ฌธ์ฅ์ ์๋ ฅํ๋ฉด, predict ํ์ด์ง์์ ๋ชจ๋ธ์ด ์ถ๋ก ํ ๊ฐ์ ์ ๋ณด์ฌ์ค 


๋ชจ๋ธ
- Beomi-KcBert-base ๋ชจ๋ธ (5๊ฐ์ Bert๋ชจ๋ธ ์ค ์งง์ ๋ฌธ์ฅ์์ ๊ฐ์ฅ ์ข์ ์ฑ๋ฅ ๊ธฐ๋ก)
- tensorflow์์ ํ์ต(huggingface์ tfbertclassification์ ์ฌ์ฉํ๊ณ ์)


๋ฐ์ดํฐ์ 
- ํ์ต ๋ฐ์ดํฐ์ : `20๋ง๊ฐ` (AI hub ๋ฐ์ดํฐ์ 2๊ฐ ์์ง )
    - ๊ฐ์  ์ ๋ณด ์ ๊ณต
    - ์์งํ ๋ ๊ฐ์ ๋ฐ์ดํฐ์์ ๋ค๋ฅธ ๊ฐ์  ๋ถ๋ฅ
    - ๊ฐ์  ๋ถ๋ฅ๋ฅผ ํต์ผํ๋ ์์ ์งํ 


- ๋ชจ๋ธ ์ผ๋ฐํ ํ์คํธ ๋ฐ์ดํฐ์ : `1024๊ฐ`
    - ๊ตญ๋ฆฝ๊ตญ์ด์ ๋ชจ๋์ ๋ง๋ญ์น - ์ผ๊ธฐ ๋ฐ์ดํฐ์
    - ๊ฐ์  ์ ๋ณด๊ฐ ์ ๊ณต๋์ง ์์ผ๋ฉฐ, ์ผ๊ธฐ ํ์คํธ ๋ฐ์ดํฐ๋ง ์ ๊ณต๋จ
    - ํ์ต์ฉ ๋ฐ์ดํฐ์์ผ๋ก ์ฌ์ฉํ๊ธฐ์ ๋ฌด๋ฆฌ๊ฐ ์์ด์ ๋ชจ๋ธ ์ผ๋ฐํ ํ์คํธ ๋ฐ์ดํฐ๋ก ์ฌ์ฉ

๊ฐ์  ๋ถ๋ฅ
- ๋ก๋ฒํธ ํ๋ฃจ์น์ ์ด๋ก ์ ๊ทผ๊ฑฐํ์ฌ ๊ฐ์ ์ 8๊ฐ๋ก ๋ถ๋ฅ
- ๋ถ๋ธ, ํ์จ, ๊ธฐ๋, ํ์ค, ๊ธฐ์จ, ์ฌํ, ์ถฉ๊ฒฉ, ๋๋ ค์ (angry, calmness, expect, loathing, pleasure, sadness, shock, terror)
- ๋ฐ์ดํฐ์์ 80%์ ๊ฒฝ์ฐ ์ง์  ๋ฐ์ดํฐ ํ์ธํ์ฌ ์ ์ฒ๋ฆฌ
- ์ผ๋ถ ๋ฐ์ดํฐ ํ๊นํ์ด์ค ํ์ดํ๋ผ์ธ ์ฌ์ฉํ์ฌ `์คํ ๋ผ๋ฒจ๋ง` ํ `์ ๊ตํ` ์์ ์ํ
- `ํด๋์ค ๋ถ๊ท ํ`์ด ์ฌํ ๋ฐ์ดํฐ์ > `๋ฐ์ดํฐ ์ฆ๊ฐ`์ ํตํ ํด๋์ค ๋ถ๊ท ํ ํด์


# ๐โโ๏ธ ํ์ด์ง 
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


# ์ฌ์ฉ ๋ฐฉ๋ฒ
(1) ๋ชจ๋ธ ํ์ผ ๋ค์ด๋ก๋
- https://drive.google.com/drive/folders/10LA8mzUOMkcFcoCImyNlL_3_8jxQjTrv?usp=sharing ๊ณต์  ๋๋ผ์ด๋ธ ์ด๋ํ์ฌ DT5_beomi_kcbert_base_model.h5 ๋ชจ๋ธ ํ์ผ ๋ค์ด๋ก๋
- ๋ค์ด๋ก๋ํ ๋ชจ๋ธ ํ์ผ์ ์ฅ๊ณ  ํ๋ก์ ํธ ๋ด๋ถ๋ก ์ด๋ 
- ์ฅ๊ณ  base project์ธ Diary์ settings.py๋ก ์ด๋

(2) Diary > settings.py ์ด๋ 
- settings.py ์๋จ์ related bert import
- settings.py์ ๋งจ ํ๋จ์ ๊ฐ์ฃผ ์ฒ๋ฆฌ๋ ์ฝ๋ > ๊ฐ์ฃผ ํด์  
```python
# BEST_MODEL_NAME = 'C:\django\DL_senti_classification\DT5_beomi_kcbert_base_model.h5'
# # ์ต๊ณ  ์ฑ๋ฅ์ ๋ชจ๋ธ ๋ถ๋ฌ์ค๊ธฐ
# MODEL = tf.keras.models.load_model(BEST_MODEL_NAME,
#                                                   custom_objects = {'TFBertForSequenceClassification': TFBertForSequenceClassification})
# # tokenizer 
# TOKENIZER = BertTokenizer.from_pretrained("beomi/kcbert-base")
# print("##### settings loaded >> tokenizer ")
# print(TOKENIZER)

# ENCODER  = LabelEncoder()
# labels = pd.read_csv(r"C:\django\DL_senti_classification\labels.csv")
# ENCODER.fit_transform(np.array(labels['๊ฐ์ '])) 
```

- ๊ฐ์ฃผ ํด์  ํ ๋ค์ด๋ก๋ํ์ฌ ์ด๋์ํจ ๋ชจ๋ธ์ ์ ๋ ๊ฒฝ๋ก๋ฅผ BERT_MODEL_NAME ์์ ๋ณ์์ ์ฌํ ๋น
- labels๋ณ์์ labels.csv ํ์ผ์ ์ ๋๊ฒฝ๋ก ์ฌํ ๋น
    - encoding ์ฉ๋

(3) ์ฅ๊ณ  ํ๋ก์ ํธ์ ํ์ ์ฑ model >  views.py ์ด๋
- ๊ฐ์ฃผ ์ฒ๋ฆฌ๋ convert_data , bertPredict ํจ์ ๊ฐ์ฃผ ํด์ 
- ๋งจ ํ๋จ์ predict ํจ์ํ๋ทฐ ์ฝ๋๋ก ์ด๋ํ์ฌ ๊ฐ์ฃผ ์ฒ๋ฆฌ๋ ๋ถ๋ถ์ ์ฝ๋ ๊ฐ์ฃผํด์ 
- runserver ์ฉ๋ ์ฝ๋ ๊ฐ์ฃผ ๋ฐ ์ฝ๋ context = {'input_submit' : input_submit} ์ญ์  


```python
def predict(request):
    start = time.time()
    if request.method == 'POST' :
        form = request.POST
        input_submit = form['input_diary']
        print("input_submit > ", input_submit, len(input_submit))
        pred = bertPredict(input_submit)
        context = {'input_submit' : pred}
        
        # runserver ์ฉ๋ ์ฝ๋ 
        # context = {'input_submit' : input_submit} >> ******* ์ญ์  

    end = time.time()
    print("์ด ์์์๊ฐ >> ", end-start )
    return render(request, 'model/predict.html', context)
```

(4) runserver ์คํ
