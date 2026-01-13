# file_manager/views.py
from django.shortcuts import render, redirect, get_object_or_404
from django.views import View
from django.views.generic import ListView, DetailView
from django.contrib import messages
from django.conf import settings
from django.http import JsonResponse
from .models import UploadedFile
from .forms import FileUploadForm
from .services import FileHandlerFactory
from .ml_utils import MLManager
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import logging
import threading

logger = logging.getLogger(__name__)

@method_decorator(csrf_exempt, name='dispatch')
class DebugPingView(View):
    def get(self, request):
        return JsonResponse({"status": "pong", "method": "GET"})
    def post(self, request):
        return JsonResponse({"status": "pong", "method": "POST"})

@method_decorator(csrf_exempt, name='dispatch')
class FileUploadView(View):
    """Maneja la subida de archivos."""
    form_class = FileUploadForm
    
    def get(self, request):
        """Configuración de subida permitida."""
        return JsonResponse({
            'allowed_extensions': settings.ALLOWED_FILE_EXTENSIONS,
            'max_size_mb': settings.MAX_UPLOAD_SIZE // (1024 * 1024),
        })
    
    def post(self, request):
        """Procesa el archivo subido."""
        form = self.form_class(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = form.save()
            return JsonResponse({
                'status': 'success',
                'file_id': uploaded_file.pk,
                'filename': uploaded_file.filename,
                'file_size': uploaded_file.get_file_size_display()
            })
        return JsonResponse({'status': 'error', 'errors': form.errors}, status=400)

class FileListView(ListView):
    """Lista de archivos subidos recientemente."""
    def get(self, request, *args, **kwargs):
        try:
            files = UploadedFile.objects.all().order_by('-uploaded_at')[:20]
            return JsonResponse({
                'total_files': UploadedFile.objects.count(),
                'files': [
                    {
                        'id': f.id,
                        'filename': f.filename,
                        'file_type': f.file_type,
                        'size': f.get_file_size_display(),
                        'uploaded_at': f.uploaded_at.isoformat()
                    } for f in files
                ]
            })
        except Exception as e:
            logger.error(f"Error en FileListView: {str(e)}")
            return JsonResponse({'error': 'Error al obtener la lista', 'ok': False}, status=500)

class HomeView(View):
    """Lista las tareas de Machine Learning disponibles."""
    def get(self, request):
        tasks = [
            {'id': '05_spam', 'name': '05: Regresión Logística (Spam)', 'icon': 'bi-envelope-exclamation', 'description': 'Detección de Spam.', 'ext': 'ZIP / index', 'color': 'primary'},
            {'id': '06_viz', 'name': '06: Visualización de DataSet', 'icon': 'bi-bar-chart-line', 'description': 'Gráficos del dataset NSL-KDD.', 'ext': '.arff', 'color': 'info'},
            {'id': '07_split', 'name': '07: División del DataSet', 'icon': 'bi-columns-gap', 'description': 'División Train/Test.', 'ext': '.arff', 'color': 'warning'},
            {'id': '08_prep', 'name': '08: Preparación del DataSet', 'icon': 'bi-gear-wide-connected', 'description': 'Limpieza e imputación.', 'ext': '.arff', 'color': 'secondary'},
            {'id': '09_pipelines', 'name': '09: Pipelines Personalizados', 'icon': 'bi-diagram-3', 'description': 'Transformadores scikit-learn.', 'ext': '.arff', 'color': 'dark'},
            {'id': '10_eval', 'name': '10: Evaluación de Resultados', 'icon': 'bi-check-all', 'description': 'Métricas de rendimiento.', 'ext': '.arff', 'color': 'success'}
        ]
        return JsonResponse({'tasks': tasks})

class FileDisplayView(DetailView):
    """Visualiza el contenido procesado de un archivo."""
    model = UploadedFile
    
    def get(self, request, *args, **kwargs):
        obj = self.get_object()
        handler = FileHandlerFactory.get_handler(obj.file_type)
        if not handler:
            return JsonResponse({'error': 'No hay handler para este tipo'}, status=400)
        try:
            content = handler.process_file(obj.file.path)
            return JsonResponse({'status': 'success', 'content': content})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

@method_decorator(csrf_exempt, name='dispatch')
class TaskDetailView(View):
    """Maneja la ejecución de una tarea específica de ML."""

    def get(self, request, task_id):
        return JsonResponse({'task_id': task_id, 'max_size_mb': settings.MAX_UPLOAD_SIZE // (1024 * 1024)})

    def post(self, request, task_id):
        """Inicia el análisis en segundo plano."""
        form = FileUploadForm(request.POST, request.FILES)
        if not form.is_valid():
            return JsonResponse({'status': 'error', 'errors': form.errors}, status=400)
        
        uploaded_file = form.save()
        
        def run_analysis(file_obj_id, t_id):
            try:
                from .models import UploadedFile
                file_obj = UploadedFile.objects.get(pk=file_obj_id)
                file_obj.analysis_status = 'PROCESSING'
                file_obj.save()
                
                analysis_map = {
                    '05_spam': MLManager.analyze_05_spam,
                    '06_viz': MLManager.analyze_06_viz,
                    '07_split': MLManager.analyze_07_split,
                    '08_prep': MLManager.analyze_08_prep,
                    '09_pipelines': MLManager.analyze_09_pipelines,
                    '10_eval': MLManager.analyze_10_eval,
                }
                
                analysis_func = analysis_map.get(t_id)
                if not analysis_func: raise ValueError(f"Tarea desconocida: {t_id}")

                # Ejecutar análisis
                result = analysis_func(file_obj.file.path)
                
                file_obj.analysis_result = result
                file_obj.analysis_status = 'FAILED' if result.get('success') == False else 'COMPLETED'
                file_obj.save()
                
            except Exception as e:
                logger.error(f"Error en hilo de análisis: {str(e)}")
                try:
                    fo = UploadedFile.objects.get(pk=file_obj_id)
                    fo.analysis_status = 'FAILED'
                    fo.analysis_result = {'error': str(e)}
                    fo.save()
                except: pass

        thread = threading.Thread(target=run_analysis, args=(uploaded_file.pk, task_id))
        thread.daemon = True
        thread.start()
        
        return JsonResponse({'status': 'accepted', 'file_id': uploaded_file.pk}, status=202)

class TaskResultView(DetailView):
    """Polling para obtener resultados del análisis."""
    model = UploadedFile
    
    def get(self, request, *args, **kwargs):
        try:
            obj = self.get_object()
            data = {'status': obj.analysis_status, 'file_id': obj.pk, 'filename': obj.filename}
            if obj.analysis_status == 'COMPLETED':
                data['ml_result'] = obj.analysis_result
            elif obj.analysis_status == 'FAILED':
                data['error'] = obj.analysis_result
            return JsonResponse(data)
        except:
            return JsonResponse({'status': 'error', 'message': 'No encontrado'}, status=404)
