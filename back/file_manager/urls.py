# file_manager/urls.py
"""URL configuration for file_manager app."""
from django.urls import path
from . import views

app_name = 'file_manager'

urlpatterns = [
    path('', views.HomeView.as_view(), name='home'),
    path('ping/', views.DebugPingView.as_view(), name='ping'),
    path('tasks/', views.HomeView.as_view(), name='tasks_list'),  # FIX: Frontend espera lista de tareas ML, no archivos
    path('files/', views.FileListView.as_view(), name='file_list'),  # Movido a /files/
    path('tasks/<slug:task_id>/', views.TaskDetailView.as_view(), name='task_upload'),
    path('tasks/<slug:task_id>/result/<int:pk>/', views.TaskResultView.as_view(), name='task_result'),
    path('files/<int:pk>/', views.FileDisplayView.as_view(), name='file_detail'),
]
