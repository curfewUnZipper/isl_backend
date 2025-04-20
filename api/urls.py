from django.urls import path
from .views import double_dict

urlpatterns = [
    path('double/', double_dict),
]
