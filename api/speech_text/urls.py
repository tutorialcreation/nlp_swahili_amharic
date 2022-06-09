from django.urls import path,include
from rest_framework import routers
from . import views
app_name='stt'

router = routers.DefaultRouter()
router.register('audio', views.AudioLoader)


urlpatterns=[
    path('', include(router.urls)),
    path('lang/',views.FetchLanguage.as_view(),name='lang'),
    path('predict/',views.PredictView.as_view(),name='predict'),
    path('sr/',views.SPR.as_view(),name='spr'),
    path('sta/',views.Stt.as_view(),name='sta'),
]