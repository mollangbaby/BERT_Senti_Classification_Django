from django.shortcuts import render
from django.views.generic import TemplateView
import time

import numpy as np 
from django.conf import settings

# def convert_data(input_data) : 
#     TOKENIZER = getattr(settings, 'TOKENIZER', 'None')
#     print("convert data Function >>> tokenizer >>", TOKENIZER)
#     out = TOKENIZER(input_data, truncation = True, padding = 'max_length', max_length = 64)
#     tokens = np.array([out['input_ids']])
#     masks = np.array([out['attention_mask']])
#     segments = np.array([out['token_type_ids']])
#     return [tokens, masks, segments]
    
# def bertPredict(input):
#     converted = convert_data(input)
#     MODEL = getattr(settings, 'MODEL', 'None')
#     print("function BertPREDICT >> ", MODEL)
#     ENCODER = getattr(settings, 'ENCODER', 'None')
#     val = MODEL.predict(converted)
#     pred = ENCODER.inverse_transform(np.argmax(val, axis = 1 ))[0]
#     return pred

class HomeView(TemplateView):
    template_name = 'model/home.html'

def info(request):
    return render(request, 'model/info.html')

def data(request):
    return render(request, 'model/data.html')

def code(request):
    return render(request, 'model/code.html')

def input(request):
    return render(request, 'model/input.html', {'data' : "log your day"})

def predict(request):
    start = time.time()
    if request.method == 'POST' :
        form = request.POST
        input_submit = form['input_diary']
        # print("input_submit > ", input_submit, len(input_submit))
        # pred = bertPredict(input_submit)
        # context = {'input_submit' : pred}
        
        # runserver용도
        context = {'input_submit' : input_submit}

    end = time.time()
    print("총 소요시간 >> ", end-start )
    return render(request, 'model/predict.html', context)
