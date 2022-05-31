from django.contrib import admin
from django.urls import path
from QQ import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('main/', views.index),
    path('data/', views.Data, name='data'),
]
