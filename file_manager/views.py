# file_manager/views.py
"""
Views for file upload application.
Following Clean Code - Thin Controllers principle.
Controllers coordinate, services contain business logic.
"""
from django.shortcuts import render, redirect, get_object_or_404
from django.views import View
from django.views.generic import ListView, DetailView
from django.contrib import messages
from django.conf import settings
from .models import UploadedFile
from .forms import FileUploadForm
from .services import FileHandlerFactory
from .ml_utils import MLManager


class FileUploadView(View):
    """
    Handle file upload functionality.
    GET: Display upload form
    POST: Process file upload
    """
    
    template_name = 'file_manager/upload.html'
    form_class = FileUploadForm
    
    def get(self, request):
        """Display upload form."""
        form = self.form_class()
        context = {
            'form': form,
            'allowed_extensions': settings.ALLOWED_FILE_EXTENSIONS,
            'max_size_mb': settings.MAX_UPLOAD_SIZE // (1024 * 1024),
        }
        return render(request, self.template_name, context)
    
    def post(self, request):
        """Process file upload."""
        form = self.form_class(request.POST, request.FILES)
        
        if form.is_valid():
            uploaded_file = form.save()
            messages.success(
                request,
                f'Archivo "{uploaded_file.filename}" subido exitosamente '
                f'({uploaded_file.get_file_size_display()})'
            )
            return redirect('file_manager:file_detail', pk=uploaded_file.pk)
        
        context = {
            'form': form,
            'allowed_extensions': settings.ALLOWED_FILE_EXTENSIONS,
            'max_size_mb': settings.MAX_UPLOAD_SIZE // (1024 * 1024),
        }
        return render(request, self.template_name, context)


class FileListView(ListView):
    """Display list of uploaded files."""
    model = UploadedFile
    template_name = 'file_manager/file_list.html'
    context_object_name = 'files'
    paginate_by = 20
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['total_files'] = UploadedFile.objects.count()
        return context


class FileDisplayView(DetailView):
    """
    Display uploaded file content.
    Uses FileHandlerFactory to get appropriate handler.
    """
    model = UploadedFile
    template_name = 'file_manager/display.html'
    context_object_name = 'file'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        uploaded_file = self.object
        file_path = uploaded_file.file.path
        
        handler = FileHandlerFactory.get_handler(uploaded_file.file_type)
        if handler:
            try:
                file_content = handler.process_file(file_path)
                context['file_content'] = file_content
                context['processing_success'] = True
            except Exception as e:
                context['processing_success'] = False
                context['error_message'] = str(e)
        else:
            context['processing_success'] = False
            context['error_message'] = f'No hay handler disponible para archivos .{uploaded_file.file_type}'
        
        return context


class HomeView(View):
    """Home page with 6 ML functional paths."""
    def get(self, request):
        tasks = [
            {
                'id': '05_spam',
                'name': '05: Regresión Logística (Spam)',
                'icon': 'bi-envelope-exclamation',
                'description': 'Detección de Spam. Acepta cualquier formato y archivos sin extensión.',
                'ext': 'Cualquier extensión / index',
                'color': 'primary'
            },
            {
                'id': '06_viz',
                'name': '06: Visualización de DataSet',
                'icon': 'bi-bar-chart-line',
                'description': 'Visualización del dataset NSL-KDD.',
                'ext': '.arff',
                'color': 'info'
            },
            {
                'id': '07_split',
                'name': '07: División del DataSet',
                'icon': 'bi-columns-gap',
                'description': 'Técnicas de división de datos (Train/Test).',
                'ext': '.arff',
                'color': 'warning'
            },
            {
                'id': '08_prep',
                'name': '08: Preparación del DataSet',
                'icon': 'bi-gear-wide-connected',
                'description': 'Limpieza y preparación inicial del dataset.',
                'ext': '.arff',
                'color': 'secondary'
            },
            {
                'id': '09_pipelines',
                'name': '09: Pipelines Personalizados',
                'icon': 'bi-diagram-3',
                'description': 'Transformadores y pipelines de scikit-learn.',
                'ext': '.arff',
                'color': 'dark'
            },
            {
                'id': '10_eval',
                'name': '10: Evaluación de Resultados',
                'icon': 'bi-check-all',
                'description': 'Evaluación de modelos y métricas de rendimiento.',
                'ext': '.arff',
                'color': 'success'
            }
        ]
        context = {
            'tasks': tasks,
            'recent_files': UploadedFile.objects.all()[:5],
        }
        return render(request, 'file_manager/home.html', context)


class TaskDetailView(View):
    """View for a specific ML task upload and instructions."""
    
    @staticmethod
    def get_task_info(task_id):
        """Helper to get task metadata."""
        task_data = {
            '05_spam': {'name': '05: Regresión Logística (Spam)', 'exts': settings.ALLOWED_FILE_EXTENSIONS},
            '06_viz': {'name': '06: Visualización de DataSet', 'exts': ['arff']},
            '07_split': {'name': '07: División del DataSet', 'exts': ['arff']},
            '08_prep': {'name': '08: Preparación del DataSet', 'exts': ['arff']},
            '09_pipelines': {'name': '09: Pipelines Personalizados', 'exts': ['arff']},
            '10_eval': {'name': '10: Evaluación de Resultados', 'exts': ['arff']}
        }
        return task_data.get(task_id, {'name': 'Tarea Desconocida', 'exts': settings.ALLOWED_FILE_EXTENSIONS})

    def get(self, request, task_id):
        task_info = self.get_task_info(task_id)
        context = {
            'task_id': task_id,
            'task_name': task_info['name'],
            'form': FileUploadForm(),
            'allowed_extensions': task_info['exts'],
            'max_size_mb': settings.MAX_UPLOAD_SIZE // (1024 * 1024),
        }
        return render(request, 'file_manager/task_upload.html', context)

    def post(self, request, task_id):
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.save()
            return redirect('file_manager:task_result', task_id=task_id, pk=uploaded_file.pk)
        
        task_info = self.get_task_info(task_id)
        context = {
            'task_id': task_id,
            'task_name': task_info['name'],
            'form': form,
            'allowed_extensions': task_info['exts'],
            'max_size_mb': settings.MAX_UPLOAD_SIZE // (1024 * 1024),
        }
        return render(request, 'file_manager/task_upload.html', context)


class TaskResultView(DetailView):
    """View to display ML results for a specific task and file."""
    model = UploadedFile
    template_name = 'file_manager/task_result.html'
    context_object_name = 'file'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        task_id = self.kwargs.get('task_id')
        file_path = self.object.file.path
        
        # Mapping task_id to MLManager methods
        analysis_map = {
            '05_spam': MLManager.analyze_05_spam,
            '06_viz': MLManager.analyze_06_viz,
            '07_split': MLManager.analyze_07_split,
            '08_prep': MLManager.analyze_08_prep,
            '09_pipelines': MLManager.analyze_09_pipelines,
            '10_eval': MLManager.analyze_10_eval,
        }
        
        analysis_func = analysis_map.get(task_id)
        result = analysis_func(file_path) if analysis_func else {"success": False, "error": "Tarea no reconocida"}
            
        context['ml_result'] = result
        context['task_id'] = task_id
        return context
