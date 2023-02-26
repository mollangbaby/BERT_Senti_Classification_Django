from django.urls import path
from model import views
app_name = "model"

urlpatterns = [
    path('input/', views.input, name = 'input'),
    path('predict/', views.predict, name = 'predict'),
    path('info/', views.info, name = 'info'),
    path('data/', views.data, name = 'data'),
    path('code/', views.code, name = 'code'),
]
