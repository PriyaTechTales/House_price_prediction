from django.contrib import admin
from django.urls import path
from home import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.index, name='home'),
    path('contact', views.contact, name='contact'),
    path('help', views.help, name='help'),
    path('login', views.login, name='login'),
    path('signup', views.signup, name='signup'),
    path('about', views.about, name='about')
]

 