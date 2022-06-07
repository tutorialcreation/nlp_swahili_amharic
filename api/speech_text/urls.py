from django.urls import path
from . import views
app_name='stt'

urlpatterns=[
    path('create/',views.FetchAudio,name='fetch'),
    path('lang/',views.FetchLanguage,name='lang'),
    path('predict/',views.PredictView,name='predict'),
]