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
from django.http import JsonResponse
from .models import UploadedFile
from .forms import FileUploadForm
from .services import FileHandlerFactory
from .ml_utils import MLManager
from django.utils.decorators import method_decorator
from django.views.decorators.csrf import csrf_exempt
import logging

logger = logging.getLogger(__name__)


@method_decorator(csrf_exempt, name='dispatch')
class DebugPingView(View):
    def get(self, request):
        return JsonResponse({"status": "pong", "method": "GET"})
    def post(self, request):
        return JsonResponse({"status": "pong", "method": "POST"})

@method_decorator(csrf_exempt, name='dispatch')
class FileUploadView(View):
    """
    Handle file upload functionality.
    GET: Display upload form
    POST: Process file upload
    """
    
    template_name = 'file_manager/upload.html'
    form_class = FileUploadForm
    
    def get(self, request):
        """Get upload configuration."""
        data = {
            'allowed_extensions': settings.ALLOWED_FILE_EXTENSIONS,
            'max_size_mb': settings.MAX_UPLOAD_SIZE // (1024 * 1024),
        }
        return JsonResponse(data)
    
    def post(self, request):
        """Process file upload."""
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
    """Display list of uploaded files."""
    def get(self, request, *args, **kwargs):
        try:
            files = UploadedFile.objects.all().order_by('-uploaded_at')[:20]
            data = {
                'total_files': UploadedFile.objects.count(),
                'files': [
                    {
                        'id': f.id,
                        'filename': f.filename,
                        'file_type': f.file_type,
                        'size': f.get_file_size_display(),
                        'uploaded_at': f.uploaded_at.isoformat()  # FIX: era created_at
                    } for f in files
                ]
            }
            return JsonResponse(data)
        except Exception as e:
            logger.error(f"Error en FileListView: {str(e)}")
            return JsonResponse({
                'error': 'Error al obtener la lista de archivos',
                'ok': False
            }, status=500)


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
    """API view to get available ML tasks."""
    def get(self, request):
        try:
            tasks = [
            # ... (mismo contenido que antes, omitido por brevedad en el chunk, pero lo pondré completo)
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
            return JsonResponse({'tasks': tasks})
        except Exception as e:
            logger.error(f"Error en HomeView: {str(e)}")
            return JsonResponse({
                'error': 'Error al cargar las tareas disponibles',
                'ok': False
            }, status=500)


@method_decorator(csrf_exempt, name='dispatch')
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
        return JsonResponse({
            'task_id': task_id,
            'task_name': task_info['name'],
            'allowed_extensions': task_info['exts'],
            'max_size_mb': settings.MAX_UPLOAD_SIZE // (1024 * 1024),
        })

    def post(self, request, task_id):
        import time
        import threading
        
        try:
            # 1. Validar formulario y guardar archivo
            form = FileUploadForm(request.POST, request.FILES)
            if not form.is_valid():
                return JsonResponse({'status': 'error', 'errors': form.errors}, status=400)
            
            uploaded_file = form.save()
            print(f"[DEBUG] Archivo aceptado (ID: {uploaded_file.pk}). Iniciando Thread...")
            
            # 2. Definir función worker para el Thread
            def run_analysis(file_obj_id, t_id):
                try:
                    # Re-instanciar modelo para evitar problemas de hilo
                    from .models import UploadedFile
                    file_obj = UploadedFile.objects.get(pk=file_obj_id)
                    
                    file_obj.analysis_status = 'PROCESSING'
                    file_obj.save()
                    
                    file_path = file_obj.file.path
                    
                    analysis_map = {
                        '05_spam': MLManager.analyze_05_spam,
                        '06_viz': MLManager.analyze_06_viz,
                        '07_split': MLManager.analyze_07_split,
                        '08_prep': MLManager.analyze_08_prep,
                        '09_pipelines': MLManager.analyze_09_pipelines,
                        '10_eval': MLManager.analyze_10_eval,
                    }
                    
                    analysis_func = analysis_map.get(t_id)
                    if not analysis_func:
                        raise ValueError(f"Tarea desconocida: {t_id}")

                    # Ejecutar análisis (lento)
                    result = analysis_func(file_path)
                    
                    # Guardar éxito o fallo controlado
                    file_obj.analysis_result = result
                    if isinstance(result, dict) and not result.get('success', True):
                        file_obj.analysis_status = 'FAILED'
                    else:
                        file_obj.analysis_status = 'COMPLETED'
                        
                    file_obj.save()
                    print(f"[THREAD] Análisis completado para ID {file_obj_id} (Status: {file_obj.analysis_status})")
                    
                except Exception as e:
                    import traceback
                    print(f"[THREAD ERROR] {str(e)}")
                    try:
                        file_obj = UploadedFile.objects.get(pk=file_obj_id)
                        file_obj.analysis_status = 'FAILED'
                        file_obj.analysis_result = {'error': str(e), 'traceback': traceback.format_exc()}
                        file_obj.save()
                    except:
                        pass

            # 3. Lanzar hilo y retornar INMEDIATAMENTE
            thread = threading.Thread(target=run_analysis, args=(uploaded_file.pk, task_id))
            thread.daemon = True # Para que no bloquee shutdown
            thread.start()
            
            # 4. Respuesta Rápida
            return JsonResponse({
                'status': 'accepted',
                'task_id': task_id,
                'file_id': uploaded_file.pk,
                'message': 'Procesamiento iniciado en segundo plano'
            }, status=202)
            
        except Exception as e:
            logger.error(f"Error en TaskDetailView.post: {str(e)}")
            return JsonResponse({
                'status': 'error',
                'error': str(e),
                'ok': False
            }, status=500)


class TaskResultView(DetailView):
    """API View to check status and get results (POLLING)."""
    model = UploadedFile
    
    def get(self, request, *args, **kwargs):
        try:
            self.object = self.get_object()
            
            # Retornar Status + Resultado si existe
            data = {
                'status': self.object.analysis_status,
                'file_id': self.object.pk,
                'filename': self.object.filename,
            }
            
            if self.object.analysis_status == 'COMPLETED':
                data['ml_result'] = self.object.analysis_result
            elif self.object.analysis_status == 'FAILED':
                data['error'] = self.object.analysis_result
                
            return JsonResponse(data)
            
        except UploadedFile.DoesNotExist:
            return JsonResponse({'status': 'error', 'message': 'Archivo no encontrado'}, status=404)
