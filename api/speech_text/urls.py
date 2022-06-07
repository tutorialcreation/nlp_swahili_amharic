from django.urls import path
from . import views
app_name='stt'

urlpatterns=[
    path('create/',views.FetchAudio.as_view(),name='fetch'),
    path('lang/',views.FetchLanguage.as_view(),name='lang'),
    path('predict/',views.PredictView.as_view(),name='predict'),
]