import math
import json
import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from matplotlib.patches import Polygon as MplPolygon
from matplotlib import animation
from IPython.display import HTML, display

# QGIS импорты
from qgis.core import (
    QgsProject, QgsVectorLayer, QgsFeature, QgsGeometry,
    QgsPointXY, QgsWkbTypes, QgsCoordinateReferenceSystem,
    QgsDistanceArea, QgsMessageLog, Qgis
)
from qgis.gui import QgsMapTool, QgsRubberBand
from qgis.PyQt.QtWidgets import (
    QAction, QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QFileDialog, QMessageBox,
    QGroupBox, QFormLayout, QLineEdit, QTextEdit, QSpinBox
)
from qgis.PyQt.QtCore import Qt, pyqtSignal
from qgis.PyQt.QtGui import QColor

import numpy as np
from scipy import spatial
from shapely.geometry import Point, Polygon, LineString, MultiPoint
from shapely.ops import nearest_points


# Типы преобразований образа ситуации
class TransformationType(Enum):
    PERMISSIBLE = "допустимое"      # H_a(p) - допустимые преобразования
    IMPERMISSIBLE = "недопустимое"  # H_i(p) - недопустимые преобразования
    POSSIBLE = "возможное"          # H_c(p) - возможные преобразования

# Типы контекстов логистических ситуаций
class ContextType(Enum):
    URBAN = "городской"
    HIGHWAY = "магистральный"
    RURAL = "сельский"
    MOUNTAIN = "горный"
    EMERGENCY = "экстренный"

# Картографический объект в образе ситуации
@dataclass
class SpatialObject:
    geometry: QgsGeometry
    attributes: Dict[str, float]
    object_type: str  # "точка", "полигон", "линия"
    def distance_to(self, other: 'SpatialObject') -> float:
        # Вычисление расстояния между объектами
        if self.geometry is None or other.geometry is None:
            return float('inf')
        return self.geometry.distance(other.geometry)

# Преобразование образа ситуации
@dataclass
class Transformation:
    type: TransformationType
    geometry: QgsGeometry  # Область преобразования
    affected_objects: List[int]  # Индексы затронутых объектов
    constraints: Dict[str, float]  # Ограничения преобразования
    weight: float = 1.0  # Вес преобразования в метрике
    
    def is_valid_for(self, target_geometry: QgsGeometry) -> bool:
        # Проверка допустимости преобразования в целевой области
        if self.type == TransformationType.IMPERMISSIBLE:
            return False
        if self.geometry is None:
            return True
        return self.geometry.intersects(target_geometry)

@dataclass
class Image:
    # Двухкомпонентная модель образа пространственной ситуации:
    # - center: базовый прецедент (наблюдавшаяся ситуация)
    # - neighborhood: образ пространственной окрестности
    # - transformations: множества допустимых, недопустимых и возможных преобразований
    id: str
    context_type: ContextType
    center: List[SpatialObject]  # Основные объекты ситуации
    neighborhood: List[SpatialObject]  # Объекты окрестности
    transformations: Dict[TransformationType, List[Transformation]]
    solution: Dict[str, any]  # Решение для данной ситуации
    metadata: Dict[str, any] = field(default_factory=dict)
    
    def get_all_objects(self) -> List[SpatialObject]:
        # Получить все объекты образа
        return self.center + self.neighborhood
    
    def get_permissible_transformations(self) -> List[Transformation]:
        # Получить допустимые преобразования
        return self.transformations.get(TransformationType.PERMISSIBLE, [])
    
    def get_impermissible_transformations(self) -> List[Transformation]:
        # Получить недопустимые преобразования
        return self.transformations.get(TransformationType.IMPERMISSIBLE, [])

@dataclass
class ProblemSituation:
    # Проблемная ситуация, для которой ищется решение
    start_point: QgsGeometry
    end_point: QgsGeometry
    context_type: Optional[ContextType]
    additional_params: Dict[str, float]
    target_area: QgsGeometry  # Целевая область для трансформирования

# БАЗА ЗНАНИЙ ОБРАЗОВ
class KnowledgeBase:
    # База знаний образов логистических маршрутов.
    # Хранит прецеденты в виде двухкомпонентных образов ситуаций, организованных по контекстам.    
    def __init__(self):
        self.images: Dict[str, Image] = {}
        self.context_images: Dict[ContextType, List[str]] = defaultdict(list)
        self._crs = QgsCoordinateReferenceSystem("EPSG:4326")
    
    def add_image(self, image: Image):
        # Добавить образ в базу знаний
        self.images[image.id] = image
        self.context_images[image.context_type].append(image.id)
        QgsMessageLog.logMessage(
            f"Добавлен образ {image.id} в контекст {image.context_type.value}",
            "LogisticsPlugin"
        )
    
    def get_images_by_context(self, context_type: ContextType) -> List[Image]:
        # Получить все образы заданного контекста
        image_ids = self.context_images.get(context_type, [])
        return [self.images[img_id] for img_id in image_ids if img_id in self.images]
    
    def get_all_contexts(self) -> List[ContextType]:
        # Получить все доступные контексты
        return list(self.context_images.keys())
    
    def load_from_file(self, filepath: str):
        # Загрузка базы знаний из файла
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for img_data in data.get('images', []):
                image = self._deserialize_image(img_data)
                if image:
                    self.add_image(image)
            
            QgsMessageLog.logMessage(
                f"Загружено {len(self.images)} образов из {filepath}",
                "LogisticsPlugin", Qgis.Info
            )
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Ошибка загрузки: {str(e)}",
                "LogisticsPlugin", Qgis.Critical
            )
    
    def save_to_file(self, filepath: str):
        # Сохранение базы знаний в файл
        data = {
            'images': [self._serialize_image(img) for img in self.images.values()],
            'context_mapping': {
                ctx.value: ids 
                for ctx, ids in self.context_images.items()
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def _serialize_image(self, image: Image) -> Dict:
        # Сериализация образа в словарь
        return {
            'id': image.id,
            'context_type': image.context_type.value,
            'center': [self._serialize_object(obj) for obj in image.center],
            'neighborhood': [self._serialize_object(obj) for obj in image.neighborhood],
            'transformations': {
                t_type.value: [self._serialize_transformation(t) for t in transforms]
                for t_type, transforms in image.transformations.items()
            },
            'solution': image.solution,
            'metadata': image.metadata
        }
    
    def _deserialize_image(self, data: Dict) -> Optional[Image]:
        # Десериализация образа из словаря
        try:
            context_type = ContextType(data['context_type'])
            center = [self._deserialize_object(obj) for obj in data.get('center', [])]
            neighborhood = [self._deserialize_object(obj) for obj in data.get('neighborhood', [])]
            
            transformations = {}
            for t_type_str, transforms_data in data.get('transformations', {}).items():
                t_type = TransformationType(t_type_str)
                transformations[t_type] = [
                    self._deserialize_transformation(t) for t in transforms_data
                ]
            
            return Image(
                id=data['id'],
                context_type=context_type,
                center=center,
                neighborhood=neighborhood,
                transformations=transformations,
                solution=data.get('solution', {}),
                metadata=data.get('metadata', {})
            )
        except Exception as e:
            QgsMessageLog.logMessage(
                f"Ошибка десериализации образа: {str(e)}",
                "LogisticsPlugin", Qgis.Warning
            )
            return None
    
    def _serialize_object(self, obj: SpatialObject) -> Dict:
        # Сериализация пространственного объекта
        return {
            'geometry_wkt': obj.geometry.asWkt() if obj.geometry else None,
            'attributes': obj.attributes,
            'object_type': obj.object_type
        }
    
    def _deserialize_object(self, data: Dict) -> SpatialObject:
        # Десериализация пространственного объекта
        geometry = None
        if data.get('geometry_wkt'):
            geometry = QgsGeometry.fromWkt(data['geometry_wkt'])
        
        return SpatialObject(
            geometry=geometry,
            attributes=data.get('attributes', {}),
            object_type=data.get('object_type', 'точка')
        )
    
    def _serialize_transformation(self, t: Transformation) -> Dict:
        # Сериализация преобразования
        return {
            'type': t.type.value,
            'geometry_wkt': t.geometry.asWkt() if t.geometry else None,
            'affected_objects': t.affected_objects,
            'constraints': t.constraints,
            'weight': t.weight
        }
    
    def _deserialize_transformation(self, data: Dict) -> Transformation:
        # Десериализация преобразования
        geometry = None
        if data.get('geometry_wkt'):
            geometry = QgsGeometry.fromWkt(data['geometry_wkt'])
        
        return Transformation(
            type=TransformationType(data['type']),
            geometry=geometry,
            affected_objects=data.get('affected_objects', []),
            constraints=data.get('constraints', {}),
            weight=data.get('weight', 1.0)
        )


# МЕТРИКИ СХОДСТВА СИТУАЦИЙ
class SimilarityMetric:
    # Метрика оценки сходства пространственных ситуаций. 
    # Реализует двухэтапный подход:
    # Трансформирование прецедента в область отображения проблемной ситуации
    # Оценка близости полученного отображения и проблемной ситуации
    # Учитывает:
    # - Геометрическую близость объектов
    # - Семантическую целостность (через преобразования)
    # - Контекстную зависимость
    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.distance_calculator = QgsDistanceArea()
        self.distance_calculator.setSourceCrs(
            QgsCoordinateReferenceSystem("EPSG:4326"),
            QgsProject.instance().transformContext()
        )
        self.distance_calculator.setEllipsoid("WGS84")
    
    def calculate_similarity(
        self,
        image: Image,
        situation: ProblemSituation,
        transformed: bool = True
    ) -> float: 
        # Вычисление сходства между образом и проблемной ситуацией.
        # Args:
        #    image: Образ из базы знаний
        #    situation: Проблемная ситуация
        #    transformed: Если True, сначала трансформировать образ
        # Returns:
        #    float: Значение сходства от 0 до 1 (1 - полное сходство)
        # Этап 1: Трансформирование образа в целевую область
        if transformed:
            transformed_image = self._transform_image(image, situation.target_area)
            if transformed_image is None:
                return 0.0  # Нереализуемо в данной области
        else:
            transformed_image = image
        
        # Этап 2: Оценка близости
        geometric_similarity = self._geometric_similarity(
            transformed_image, situation
        )
        semantic_similarity = self._semantic_similarity(
            transformed_image, situation
        )
        contextual_similarity = self._contextual_similarity(
            transformed_image, situation
        )
        
        # Интегральная метрика с весами
        weights = {
            'geometric': 0.4,
            'semantic': 0.4,
            'contextual': 0.2
        }
        
        total_similarity = (
            weights['geometric'] * geometric_similarity +
            weights['semantic'] * semantic_similarity +
            weights['contextual'] * contextual_similarity
        )
        
        return total_similarity
    
    def _geometric_similarity(
        self,
        image: Image,
        situation: ProblemSituation
    ) -> float:
        # Геометрическая составляющая метрики.
        # Оценивает близость расположения объектов образа к точкам старта и финиша проблемной ситуации.
        if not image.center or situation.start_point is None:
            return 0.0
        
        # Находим центроид образа
        image_centroid = self._calculate_centroid(image.center)
        
        # Расстояние от центроида образа до линии старта-финиша
        route_line = QgsGeometry.fromPolylineXY([
            situation.start_point.asPoint(),
            situation.end_point.asPoint()
        ])
        
        distance = image_centroid.distance(route_line)
        
        # Нормализация расстояния (чем ближе, тем выше сходство)
        max_distance = 10000  # 10 км в метрах
        normalized_distance = min(distance / max_distance, 1.0)
        
        return 1.0 - normalized_distance
    
    def _semantic_similarity(
        self,
        image: Image,
        situation: ProblemSituation
    ) -> float:
        # Семантическая составляющая метрики.
        # Оценивает сохранение смысла через анализ преобразований.
        # Проверяет, не нарушены ли недопустимые преобразования.
        impermissible = image.get_impermissible_transformations()
        
        if not impermissible:
            return 1.0  # Нет ограничений - полное семантическое соответствие
        
        violations = 0
        for transform in impermissible:
            if transform.geometry and situation.target_area:
                # Проверяем пересечение недопустимой зоны с целевой областью
                if transform.geometry.intersects(situation.target_area):
                    violations += 1
        
        # Чем меньше нарушений, тем выше семантическое сходство
        violation_ratio = violations / len(impermissible) if impermissible else 0
        return 1.0 - violation_ratio
    
    def _contextual_similarity(
        self,
        image: Image,
        situation: ProblemSituation
    ) -> float:
        # Контекстная составляющая метрики.
        # Оценивает соответствие контекста образа контексту ситуации.
        if situation.context_type is None:
            return 0.5  # Неопределенный контекст
        
        if image.context_type == situation.context_type:
            return 1.0
        
        # Матрица близости контекстов
        context_distances = {
            (ContextType.URBAN, ContextType.RURAL): 0.7,
            (ContextType.URBAN, ContextType.HIGHWAY): 0.5,
            (ContextType.HIGHWAY, ContextType.RURAL): 0.6,
            (ContextType.MOUNTAIN, ContextType.RURAL): 0.8,
            (ContextType.URBAN, ContextType.MOUNTAIN): 0.3,
            (ContextType.EMERGENCY, ContextType.URBAN): 0.6,
        }
        
        key = (image.context_type, situation.context_type)
        reverse_key = (situation.context_type, image.context_type)
        
        return context_distances.get(key, context_distances.get(reverse_key, 0.4))
    
    def _transform_image(
        self,
        image: Image,
        target_area: QgsGeometry
    ) -> Optional[Image]:
        # Трансформирование образа в целевую область.        
        # Проверяет реализуемость образа в заданной локации через анализ допустимых преобразований.
        # Returns:
        #    Image: Трансформированный образ или None, если нереализуемо
        # Проверяем все недопустимые преобразования
        for transform in image.get_impermissible_transformations():
            if transform.geometry and target_area:
                if transform.geometry.intersects(target_area):
                    # Образ нереализуем в данной области
                    return None
        
        # Создаем трансформированную копию образа
        transformed = Image(
            id=f"{image.id}_transformed",
            context_type=image.context_type,
            center=[self._transform_object(obj, target_area) for obj in image.center],
            neighborhood=[self._transform_object(obj, target_area) for obj in image.neighborhood],
            transformations=image.transformations,
            solution=image.solution,
            metadata={**image.metadata, 'transformed': True}
        )
        
        return transformed
    
    def _transform_object(
        self,
        obj: SpatialObject,
        target_area: QgsGeometry
    ) -> SpatialObject:
        # Трансформирование отдельного объекта в целевую область
        if obj.geometry is None:
            return obj
        
        # Проецируем объект на целевую область
        centroid = obj.geometry.centroid()
        target_centroid = target_area.centroid()
        
        # Вычисляем смещение
        dx = target_centroid.asPoint().x() - centroid.asPoint().x()
        dy = target_centroid.asPoint().y() - centroid.asPoint().y()
        
        # Создаем трансформированную геометрию
        transformed_geom = obj.geometry.translated(dx, dy)
        
        return SpatialObject(
            geometry=transformed_geom,
            attributes=obj.attributes,
            object_type=obj.object_type
        )
    
    def _calculate_centroid(self, objects: List[SpatialObject]) -> QgsGeometry:
        # Вычисление центроида набора объектов
        if not objects:
            return QgsGeometry()
        
        points = []
        for obj in objects:
            if obj.geometry:
                centroid = obj.geometry.centroid()
                points.append(centroid.asPoint())
        
        if not points:
            return QgsGeometry()
        
        # Среднее арифметическое координат
        avg_x = sum(p.x() for p in points) / len(points)
        avg_y = sum(p.y() for p in points) / len(points)
        
        return QgsGeometry.fromPointXY(QgsPointXY(avg_x, avg_y))


# МЕТОДЫ ОПРЕДЕЛЕНИЯ КОНТЕКСТА
class ContextDetector:
    # Метод определения контекста проблемной ситуации.
    # Анализирует пространственные характеристики ситуации и дополнительные параметры для определения типа контекста.
    def __init__(self):
        self.rules = self._initialize_rules()
    
    def _initialize_rules(self) -> Dict[ContextType, Dict[str, any]]:
        # Инициализация правил определения контекста
        return {
            ContextType.URBAN: {
                'density_threshold': 0.7,  # Плотность застройки
                'max_distance_km': 5,      # Максимальное расстояние
                'road_types': ['primary', 'secondary', 'residential']
            },
            ContextType.HIGHWAY: {
                'density_threshold': 0.2,
                'min_distance_km': 10,
                'road_types': ['motorway', 'trunk']
            },
            ContextType.RURAL: {
                'density_threshold': 0.3,
                'max_distance_km': 50,
                'road_types': ['unclassified', 'tertiary']
            },
            ContextType.MOUNTAIN: {
                'elevation_threshold': 500,  # Высота над уровнем моря
                'slope_threshold': 15,       # Уклон в градусах
                'road_types': ['unclassified']
            },
            ContextType.EMERGENCY: {
                'time_constraint': True,
                'priority': 'high',
                'road_types': ['motorway', 'primary', 'secondary']
            }
        }
    
    def detect_context(
        self,
        situation: ProblemSituation,
        additional_data: Dict[str, any] = None
    ) -> ContextType:
        # Определение контекста проблемной ситуации.
        # Args:
        #    situation: Проблемная ситуация
        #    additional_data: Дополнительные данные (плотность застройки, тип дорог и т.д.)
        # Returns:
        #    ContextType: Определенный тип контекста
        #
        if situation.context_type is not None:
            return situation.context_type
        
        # Если есть дополнительные данные, используем их
        if additional_data:
            return self._detect_from_data(additional_data)
        
        # Иначе определяем по геометрическим характеристикам
        return self._detect_from_geometry(situation)
    
    def _detect_from_data(self, data: Dict[str, any]) -> ContextType:
        # Определение контекста на основе дополнительных данных
        scores = {}
        
        for context_type, rules in self.rules.items():
            score = 0
            matches = 0
            
            # Проверяем соответствие правилам
            if 'density_threshold' in rules:
                density = data.get('density', 0.5)
                if context_type in [ContextType.URBAN, ContextType.EMERGENCY]:
                    if density >= rules['density_threshold']:
                        score += 1
                else:
                    if density < rules['density_threshold']:
                        score += 1
                matches += 1
            
            if 'max_distance_km' in rules:
                distance = data.get('distance_km', 0)
                if distance <= rules['max_distance_km']:
                    score += 1
                matches += 1
            
            if 'elevation_threshold' in rules:
                elevation = data.get('elevation', 0)
                if elevation >= rules['elevation_threshold']:
                    score += 1
                matches += 1
            
            scores[context_type] = score / matches if matches > 0 else 0
        
        # Возвращаем контекст с максимальным score
        return max(scores, key=scores.get)
    
    def _detect_from_geometry(self, situation: ProblemSituation) -> ContextType:
        # Определение контекста на основе геометрических характеристик
        if situation.start_point is None or situation.end_point is None:
            return ContextType.URBAN  # По умолчанию
        
        # Вычисляем расстояние между точками
        distance = self.distance_calculator.measureLine(
            situation.start_point.asPoint(),
            situation.end_point.asPoint()
        )
        distance_km = distance / 1000
        
        # Эвристики определения контекста
        if distance_km < 5:
            return ContextType.URBAN
        elif distance_km < 50:
            return ContextType.HIGHWAY
        else:
            return ContextType.RURAL
    
    @property
    def distance_calculator(self):
        return QgsDistanceArea()

# МЕТОДЫ ПОИСКА ОБРАЗОВ
class ImageSearcher:
    # Метод поиска наиболее близкого по смыслу и контексту образа. Реализует двухуровневый поиск:
    # 1. Поиск в заданном контексте
    # 2. Если не найдено - поиск ближайшего контекста
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        metric: SimilarityMetric
    ):
        self.kb = knowledge_base
        self.metric = metric
    
    def find_best_image(
        self,
        situation: ProblemSituation,
        min_similarity: float = 0.3
    ) -> Tuple[Optional[Image], float]:
        # Поиск наиболее близкого образа в базе знаний.
        #   Args:
        #    situation: Проблемная ситуация
        #    min_similarity: Минимальный порог сходства
        # Returns:
        #    Tuple[Image, float]: Найденный образ и значение сходства
        # Уровень 1: Поиск в заданном контексте
        if situation.context_type:
            best_image, best_score = self._search_in_context(
                situation, situation.context_type
            )
            
            if best_image and best_score >= min_similarity:
                return best_image, best_score
        
        # Уровень 2: Поиск ближайшего контекста
        best_image, best_score = self._search_nearest_context(situation)
        
        if best_image and best_score >= min_similarity:
            return best_image, best_score
        
        return None, 0.0
    
    def _search_in_context(
        self,
        situation: ProblemSituation,
        context_type: ContextType
    ) -> Tuple[Optional[Image], float]:
        # Поиск образа в заданном контексте
        images = self.kb.get_images_by_context(context_type)
        
        if not images:
            return None, 0.0
        
        best_image = None
        best_score = 0.0
        
        for image in images:
            score = self.metric.calculate_similarity(image, situation)
            if score > best_score:
                best_score = score
                best_image = image
        
        return best_image, best_score
    
    def _search_nearest_context(
        self,
        situation: ProblemSituation
    ) -> Tuple[Optional[Image], float]:
        # Поиск образа в ближайшем контексте.
        # Используется когда в заданном контексте нет образов.
        all_contexts = self.kb.get_all_contexts()
        
        if not all_contexts:
            return None, 0.0
        
        # Находим контекст с максимальным сходством
        best_context = None
        best_context_score = 0.0
        
        for context_type in all_contexts:
            images = self.kb.get_images_by_context(context_type)
            if not images:
                continue
            
            # Усредненное сходство по всем образам контекста
            avg_score = sum(
                self.metric.calculate_similarity(img, situation)
                for img in images
            ) / len(images)
            
            if avg_score > best_context_score:
                best_context_score = avg_score
                best_context = context_type
        
        if best_context:
            return self._search_in_context(situation, best_context)
        
        return None, 0.0
    
    def find_image_chain(
        self,
        situation: ProblemSituation,
        chain_length: int = 3
    ) -> List[Image]:
        # Формирование межконтекстного образного ряда.
        # Создает цепочку образов, связанных отношением смысловой близости через общие допустимые преобразования.
        #    Args:
        #    situation: Проблемная ситуация
        #   chain_length: Длина цепочки
        # Returns:
        #    List[Image]: Цепочка образов
        best_image, _ = self.find_best_image(situation)
        
        if not best_image:
            return []
        
        chain = [best_image]
        current_image = best_image
        
        for _ in range(chain_length - 1):
            # Ищем образ с максимальным пересечением допустимых преобразований
            next_image = self._find_related_image(current_image, situation)
            
            if next_image and next_image not in chain:
                chain.append(next_image)
                current_image = next_image
            else:
                break
        
        return chain
    
    def _find_related_image(
        self,
        current_image: Image,
        situation: ProblemSituation
    ) -> Optional[Image]:
        # Поиск образа, связанного с текущим через общие преобразования
        current_transforms = set(
            id(t) for t in current_image.get_permissible_transformations()
        )
        
        best_image = None
        best_overlap = 0
        
        for image_id, image in self.kb.images.items():
            if image.id == current_image.id:
                continue
            
            image_transforms = set(
                id(t) for t in image.get_permissible_transformations()
            )
            
            overlap = len(current_transforms & image_transforms)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_image = image
        
        return best_image

# МЕТОДЫ ГЕНЕРАЦИИ И АДАПТАЦИИ
class ImageGenerator:
    # Метод генерации новых образов на основе существующих.
    # Реализует два подхода:
    # 1. Адаптация известного решения в целевую область
    # 2. Генерация нового решения на основе всех образов контекста    
    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        metric: SimilarityMetric
    ):
        self.kb = knowledge_base
        self.metric = metric
    
    def adapt_image(
        self,
        source_image: Image,
        situation: ProblemSituation
    ) -> Optional[Image]:
        # Адаптация известного образа из базы знаний в заданную целевую область.
        # Args:
        #    source_image: Исходный образ для адаптации
        #    situation: Проблемная ситуация с целевой областью
        # Returns:
        #    Image: Адаптированный образ или None, если адаптация невозможна
        # Проверяем реализуемость в целевой области
        if not self._is_feasible(source_image, situation.target_area):
            return None
        
        # Создаем адаптированную копию
        adapted = Image(
            id=f"{source_image.id}_adapted",
            context_type=situation.context_type or source_image.context_type,
            center=self._adapt_objects(source_image.center, situation),
            neighborhood=self._adapt_objects(source_image.neighborhood, situation),
            transformations=source_image.transformations,
            solution=source_image.solution.copy(),
            metadata={
                **source_image.metadata,
                'adapted_from': source_image.id,
                'adaptation_date': '2024-01-01'
            }
        )
        
        # Обновляем решение для новой области
        adapted.solution = self._adapt_solution(
            source_image.solution, situation
        )
        
        return adapted
    
    def generate_new_image(
        self,
        context_images: List[Image],
        situation: ProblemSituation
    ) -> Optional[Image]:
        # Генерация нового образа на основе всех образов заданного контекста.
        # Использует обобщение знаний из множества прецедентов для создания нового решения.
        # Args:
        #    context_images: Образы контекста для обобщения
        #   situation: Проблемная ситуация
        # Returns:
        #    Image: Сгенерированный образ или None
        if not context_images:
            return None
        
        # Анализируем все образы контекста
        common_transforms = self._find_common_transformations(context_images)
        typical_objects = self._find_typical_objects(context_images)
        
        # Создаем новый образ на основе обобщений
        new_center = self._generate_center(typical_objects, situation)
        new_neighborhood = self._generate_neighborhood(context_images, situation)
        
        new_image = Image(
            id=f"generated_{len(self.kb.images)}",
            context_type=situation.context_type or ContextType.URBAN,
            center=new_center,
            neighborhood=new_neighborhood,
            transformations={
                TransformationType.PERMISSIBLE: common_transforms,
                TransformationType.IMPERMISSIBLE: [],
                TransformationType.POSSIBLE: []
            },
            solution=self._generate_solution(context_images, situation),
            metadata={'generated': True, 'source_count': len(context_images)}
        )
        
        return new_image
    
    def _is_feasible(
        self,
        image: Image,
        target_area: QgsGeometry
    ) -> bool:
        # Проверка реализуемости образа в целевой области
        for transform in image.get_impermissible_transformations():
            if transform.geometry and target_area:
                if transform.geometry.intersects(target_area):
                    return False
        return True
    
    def _adapt_objects(
        self,
        objects: List[SpatialObject],
        situation: ProblemSituation
    ) -> List[SpatialObject]:
        # Адаптация объектов к целевой области
        adapted = []
        
        for obj in objects:
            if obj.geometry is None:
                adapted.append(obj)
                continue
            
            # Трансформируем геометрию
            centroid = obj.geometry.centroid()
            target_centroid = situation.target_area.centroid()
            
            dx = target_centroid.asPoint().x() - centroid.asPoint().x()
            dy = target_centroid.asPoint().y() - centroid.asPoint().y()
            
            new_geom = obj.geometry.translated(dx, dy)
            
            adapted.append(SpatialObject(
                geometry=new_geom,
                attributes=obj.attributes,
                object_type=obj.object_type
            ))
        
        return adapted
    
    def _adapt_solution(
        self,
        solution: Dict,
        situation: ProblemSituation
    ) -> Dict:
        # Адаптация решения к новой ситуации
        adapted = solution.copy()
        
        # Пересчитываем расстояние
        if situation.start_point and situation.end_point:
            distance = QgsDistanceArea().measureLine(
                situation.start_point.asPoint(),
                situation.end_point.asPoint()
            )
            adapted['distance_m'] = distance
            adapted['distance_km'] = distance / 1000
        
        return adapted
    
    def _find_common_transformations(
        self,
        images: List[Image]
    ) -> List[Transformation]:
        # Поиск общих допустимых преобразований во всех образах
        all_transforms = []
        
        for image in images:
            all_transforms.extend(image.get_permissible_transformations())
        
        # Группируем по типу и находим наиболее частые
        transform_counts = defaultdict(int)
        for t in all_transforms:
            key = (t.type, str(t.constraints))
            transform_counts[key] += 1
        
        # Возвращаем преобразования, встречающиеся в большинстве образов
        threshold = len(images) * 0.5
        common = [
            t for t in all_transforms
            if transform_counts[(t.type, str(t.constraints))] >= threshold
        ]
        
        return common[:5]  # Ограничиваем количество
    
    def _find_typical_objects(
        self,
        images: List[Image]
    ) -> List[SpatialObject]:
        # Поиск типичных объектов для контекста
        all_objects = []
        for image in images:
            all_objects.extend(image.center)
        
        if not all_objects:
            return []
        
        # Кластеризуем объекты по типу и атрибутам
        object_types = defaultdict(list)
        for obj in all_objects:
            object_types[obj.object_type].append(obj)
        
        # Для каждого типа берем медианный объект
        typical = []
        for obj_type, objects in object_types.items():
            if objects:
                # Берем объект с медианными атрибутами
                median_obj = self._find_median_object(objects)
                typical.append(median_obj)
        
        return typical
    
    def _find_median_object(self, objects: List[SpatialObject]) -> SpatialObject:
        # Нахождение объекта с медианными характеристиками
        if len(objects) == 1:
            return objects[0]
        
        # Сортируем по площади геометрии и берем медианный
        sorted_objects = sorted(
            objects,
            key=lambda obj: obj.geometry.area() if obj.geometry else 0
        )
        
        median_idx = len(sorted_objects) // 2
        return sorted_objects[median_idx]
    
    def _generate_center(
        self,
        typical_objects: List[SpatialObject],
        situation: ProblemSituation
    ) -> List[SpatialObject]:
        # Генерация центральных объектов нового образа
        if not typical_objects:
            # Создаем базовые объекты для маршрута
            return [
                SpatialObject(
                    geometry=situation.start_point,
                    attributes={'type': 'start'},
                    object_type='точка'
                ),
                SpatialObject(
                    geometry=situation.end_point,
                    attributes={'type': 'end'},
                    object_type='точка'
                )
            ]
        
        # Трансформируем типичные объекты в целевую область
        return self._adapt_objects(typical_objects, situation)
    
    def _generate_neighborhood(
        self,
        context_images: List[Image],
        situation: ProblemSituation
    ) -> List[SpatialObject]:
        # Генерация объектов окрестности
        all_neighborhood = []
        for image in context_images:
            all_neighborhood.extend(image.neighborhood)
        
        if not all_neighborhood:
            return []
        
        # Берем наиболее типичные объекты окрестности
        return self._find_typical_objects(all_neighborhood)[:3]
    
    def _generate_solution(
        self,
        context_images: List[Image],
        situation: ProblemSituation
    ) -> Dict:
        # Генерация решения на основе образов контекста
        # Анализируем решения всех образов
        solutions = [img.solution for img in context_images if img.solution]
        
        if not solutions:
            return {
                'type': 'generated',
                'distance_km': 0,
                'confidence': 0.5
            }
        
        # Усредняем числовые параметры
        avg_solution = {}
        numeric_keys = ['distance_km', 'time_hours', 'cost']
        
        for key in numeric_keys:
            values = [s.get(key, 0) for s in solutions if key in s]
            if values:
                avg_solution[key] = sum(values) / len(values)
        
        # Добавляем параметры ситуации
        if situation.start_point and situation.end_point:
            distance = QgsDistanceArea().measureLine(
                situation.start_point.asPoint(),
                situation.end_point.asPoint()
            )
            avg_solution['distance_km'] = distance / 1000
        
        avg_solution['type'] = 'generated'
        avg_solution['confidence'] = 0.7
        
        return avg_solution

# МЕТОДЫ ОЦЕНКИ РИСКА
class RiskEvaluator:
    # Метод оценки риска использования решения в заданной целевой области.
    # Оценивает:
    # - Успешность трансформирования
    # - Достоверность решения
    # - Потенциальные риски
    def __init__(self, metric: SimilarityMetric):
        self.metric = metric
    
    def evaluate_transformation_success(
        self,
        original_image: Image,
        transformed_image: Image,
        situation: ProblemSituation
    ) -> Dict[str, float]:
        # Оценка успешности трансформирования образа.
        # Returns:
        #    Dict с метриками успешности:
        #    - feasibility: реализуемость (0-1)
        #    - semantic_preservation: сохранение смысла (0-1)
        #    - geometric_accuracy: геометрическая точность (0-1)
        # Проверяем реализуемость
        feasibility = 1.0 if transformed_image else 0.0
        
        if not transformed_image:
            return {
                'feasibility': 0.0,
                'semantic_preservation': 0.0,
                'geometric_accuracy': 0.0,
                'overall': 0.0
            }
        
        # Оцениваем сохранение смысла через преобразования
        original_transforms = len(original_image.get_permissible_transformations())
        transformed_transforms = len(transformed_image.get_permissible_transformations())
        
        semantic_preservation = (
            transformed_transforms / original_transforms
            if original_transforms > 0 else 0.5
        )
        
        # Оцениваем геометрическую точность
        geometric_accuracy = self.metric.calculate_similarity(
            transformed_image, situation, transformed=False
        )
        
        # Интегральная оценка
        overall = (
            0.4 * feasibility +
            0.3 * semantic_preservation +
            0.3 * geometric_accuracy
        )
        
        return {
            'feasibility': feasibility,
            'semantic_preservation': semantic_preservation,
            'geometric_accuracy': geometric_accuracy,
            'overall': overall
        }
    
    def evaluate_risk(
        self,
        image: Image,
        situation: ProblemSituation,
        similarity_score: float
    ) -> Dict[str, any]:
        # Комплексная оценка риска использования решения.
        # Returns:
        #    Dict с оценкой риска:
        #    - risk_level: уровень риска (low/medium/high)
        #    - risk_score: числовая оценка (0-1, чем выше тем рискованнее)
        #    - risk_factors: факторы риска
        #    - recommendation: рекомендация
        risk_factors = []
        risk_score = 0.0
        
        # Фактор 1: Низкое сходство
        if similarity_score < 0.5:
            risk_score += 0.3
            risk_factors.append("Низкое сходство с прецедентом")
        
        # Фактор 2: Наличие возможных преобразований (неопределенность)
        possible_transforms = image.transformations.get(
            TransformationType.POSSIBLE, []
        )
        if possible_transforms:
            risk_score += 0.2 * len(possible_transforms)
            risk_factors.append("Наличие неопределенных преобразований")
        
        # Фактор 3: Сложность целевой области
        if situation.target_area:
            area_complexity = self._assess_area_complexity(situation.target_area)
            risk_score += 0.2 * area_complexity
            if area_complexity > 0.7:
                risk_factors.append("Сложная целевая область")
        
        # Фактор 4: Контекстная неопределенность
        if situation.context_type is None:
            risk_score += 0.2
            risk_factors.append("Неопределенный контекст")
        
        # Нормализация
        risk_score = min(risk_score, 1.0)
        
        # Определение уровня риска
        if risk_score < 0.3:
            risk_level = "low"
            recommendation = "Решение может быть использовано с высокой уверенностью"
        elif risk_score < 0.6:
            risk_level = "medium"
            recommendation = "Требуется дополнительная проверка экспертом"
        else:
            risk_level = "high"
            recommendation = "Высокий риск, рекомендуется найти альтернативное решение"
        
        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'recommendation': recommendation
        }
    
    def _assess_area_complexity(self, area: QgsGeometry) -> float:
        # Оценка сложности целевой области
        if area is None:
            return 0.5
        
        # Используем отношение периметра к площади как меру сложности
        perimeter = area.length()
        area_size = area.area()
        
        if area_size == 0:
            return 1.0
        
        complexity = perimeter / math.sqrt(area_size)
        
        # Нормализация
        return min(complexity / 10.0, 1.0)
    
    def detect_anomaly(
        self,
        situation: ProblemSituation
    ) -> Tuple[bool, str]:
        # Определение аномальной ситуации.
        # Проверяет, не являются ли точки старта/финиша ошибкой эксперта.
        # Returns:
        #    Tuple[bool, str]: (является_аномалией, описание)
        if situation.start_point is None or situation.end_point is None:
            return True, "Отсутствуют точки старта или финиша"
        
        # Проверяем расстояние
        distance = QgsDistanceArea().measureLine(
            situation.start_point.asPoint(),
            situation.end_point.asPoint()
        )
        
        # Аномально малое расстояние
        if distance < 10:  # менее 10 метров
            return True, "Расстояние между точками слишком мало"
        
        # Аномально большое расстояние
        if distance > 1000000:  # более 1000 км
            return True, "Расстояние между точками слишком велико"
        
        # Проверяем дополнительные параметры
        if situation.additional_params:
            for key, value in situation.additional_params.items():
                if isinstance(value, (int, float)):
                    if value < 0:
                        return True, f"Отрицательное значение параметра: {key}"
        
        return False, "Ситуация нормальна"

# ПЛАГИН QGIS
class LogisticsPlugin:
    # Плагин QGIS для интеллектуального анализа логистических маршрутов.
    # Реализует полный цикл ситуационного анализа:
    # 1. Загрузка базы знаний образов
    # 2. Определение контекста ситуации
    # 3. Поиск ближайшего образа
    # 4. Генерация/адаптация решения
    # 5. Отображение на карте
    # 6. Оценка риска
    def __init__(self, iface):
        self.iface = iface
        self.knowledge_base = KnowledgeBase()
        self.metric = SimilarityMetric(self.knowledge_base)
        self.context_detector = ContextDetector()
        self.searcher = ImageSearcher(self.knowledge_base, self.metric)
        self.generator = ImageGenerator(self.knowledge_base, self.metric)
        self.risk_evaluator = RiskEvaluator(self.metric)
        
        self.dock_widget = None
        self.start_point = None
        self.end_point = None
        
        # Слои для отображения результатов
        self.result_layer = None
        self.image_layer = None
    
    def initGui(self):
        # Инициализация интерфейса плагина
        # Создаем действие
        self.action = QAction("Логистический анализ", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        
        # Добавляем в меню
        self.iface.addPluginToMenu("Логистический анализ", self.action)
        
        # Создаем dock widget
        self.dock_widget = QDockWidget("Логистический анализ", self.iface.mainWindow())
        self.dock_widget.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        
        # Создаем интерфейс
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Группа: База знаний
        kb_group = QGroupBox("База знаний")
        kb_layout = QVBoxLayout()
        
        self.load_kb_btn = QPushButton("Загрузить базу знаний")
        self.load_kb_btn.clicked.connect(self.load_knowledge_base)
        kb_layout.addWidget(self.load_kb_btn)
        
        self.kb_status = QLabel("База знаний не загружена")
        kb_layout.addWidget(self.kb_status)
        
        kb_group.setLayout(kb_layout)
        layout.addWidget(kb_group)
        
        # Группа: Точки маршрута
        route_group = QGroupBox("Маршрут")
        route_layout = QFormLayout()
        
        self.set_start_btn = QPushButton("Указать точку старта")
        self.set_start_btn.clicked.connect(lambda: self.set_point('start'))
        route_layout.addRow(self.set_start_btn)
        
        self.set_end_btn = QPushButton("Указать точку финиша")
        self.set_end_btn.clicked.connect(lambda: self.set_point('end'))
        route_layout.addRow(self.set_end_btn)
        
        route_group.setLayout(route_layout)
        layout.addWidget(route_group)
        
        # Группа: Контекст
        context_group = QGroupBox("Контекст")
        context_layout = QFormLayout()
        
        self.context_combo = QComboBox()
        self.context_combo.addItems([
            "Автоопределение",
            "Городской",
            "Магистральный",
            "Сельский",
            "Горный",
            "Экстренный"
        ])
        context_layout.addRow("Тип:", self.context_combo)
        
        context_group.setLayout(context_layout)
        layout.addWidget(context_group)
        
        # Группа: Анализ
        analysis_group = QGroupBox("Анализ")
        analysis_layout = QVBoxLayout()
        
        self.analyze_btn = QPushButton("Выполнить анализ")
        self.analyze_btn.clicked.connect(self.perform_analysis)
        self.analyze_btn.setEnabled(False)
        analysis_layout.addWidget(self.analyze_btn)
        
        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        analysis_layout.addWidget(self.results_text)
        
        analysis_group.setLayout(analysis_layout)
        layout.addWidget(analysis_group)
        
        layout.addStretch()
        
        self.dock_widget.setWidget(widget)
        self.iface.addDockWidget(Qt.LeftDockWidgetArea, self.dock_widget)
    
    def unload(self):
        # Выгрузка плагина
        self.iface.removePluginMenu("Логистический анализ", self.action)
        self.iface.removeDockWidget(self.dock_widget)
    
    def run(self):
        # Запуск плагина
        if self.dock_widget:
            self.dock_widget.show()
    
    def load_knowledge_base(self):
        # Загрузка базы знаний из файла
        filepath, _ = QFileDialog.getOpenFileName(
            self.iface.mainWindow(),
            "Загрузить базу знаний",
            "",
            "JSON файлы (*.json);;Все файлы (*)"
        )
        
        if filepath:
            self.knowledge_base.load_from_file(filepath)
            self.kb_status.setText(
                f"Загружено образов: {len(self.knowledge_base.images)}"
            )
            self.analyze_btn.setEnabled(True)
    
    def set_point(self, point_type: str):
        # Установка точки старта или финиша
        # В реальном плагине здесь был бы мап-тул для выбора точки на карте
        # Для демонстрации используем центр карты
        canvas = self.iface.mapCanvas()
        center = canvas.extent().center()
        
        if point_type == 'start':
            self.start_point = QgsGeometry.fromPointXY(center)
            QMessageBox.information(
                self.iface.mainWindow(),
                "Точка старта",
                f"Установлена точка старта: {center.x():.4f}, {center.y():.4f}"
            )
        else:
            # Смещаем точку финиша
            end_point = QgsPointXY(center.x() + 0.01, center.y() + 0.01)
            self.end_point = QgsGeometry.fromPointXY(end_point)
            QMessageBox.information(
                self.iface.mainWindow(),
                "Точка финиша",
                f"Установлена точка финиша: {end_point.x():.4f}, {end_point.y():.4f}"
            )
    
    def perform_analysis(self):
        # Выполнение полного цикла анализа
        if not self.start_point or not self.end_point:
            QMessageBox.warning(
                self.iface.mainWindow(),
                "Ошибка",
                "Укажите точки старта и финиша"
            )
            return
        
        # Определяем контекст
        context_type = None
        if self.context_combo.currentIndex() > 0:
            context_map = {
                1: ContextType.URBAN,
                2: ContextType.HIGHWAY,
                3: ContextType.RURAL,
                4: ContextType.MOUNTAIN,
                5: ContextType.EMERGENCY
            }
            context_type = context_map.get(self.context_combo.currentIndex())
        
        situation = ProblemSituation(
            start_point=self.start_point,
            end_point=self.end_point,
            context_type=context_type,
            additional_params={},
            target_area=self._create_target_area()
        )
        
        # Проверяем на аномалию
        is_anomaly, anomaly_desc = self.risk_evaluator.detect_anomaly(situation)
        
        results = []
        results.append("=" * 50)
        results.append("РЕЗУЛЬТАТЫ АНАЛИЗА")
        results.append("=" * 50)
        
        if is_anomaly:
            results.append(f"\n⚠ ВНИМАНИЕ: Аномальная ситуация")
            results.append(f"Причина: {anomaly_desc}")
            results.append("Рекомендуется проверить корректность ввода данных")
        
        # Определяем контекст
        detected_context = self.context_detector.detect_context(situation)
        results.append(f"\nОпределенный контекст: {detected_context.value}")
        
        # Ищем ближайший образ
        best_image, similarity = self.searcher.find_best_image(situation)
        
        if best_image:
            results.append(f"\nНайден ближайший образ: {best_image.id}")
            results.append(f"Сходство: {similarity:.3f}")
            results.append(f"Контекст образа: {best_image.context_type.value}")
            
            # Адаптируем образ
            adapted_image = self.generator.adapt_image(best_image, situation)
            
            if adapted_image:
                results.append(f"\n✓ Образ успешно адаптирован")
                
                # Оцениваем успешность трансформирования
                success_metrics = self.risk_evaluator.evaluate_transformation_success(
                    best_image, adapted_image, situation
                )
                results.append(f"Реализуемость: {success_metrics['feasibility']:.3f}")
                results.append(f"Сохранение смысла: {success_metrics['semantic_preservation']:.3f}")
                results.append(f"Геометрическая точность: {success_metrics['geometric_accuracy']:.3f}")
                results.append(f"Общая оценка: {success_metrics['overall']:.3f}")
                
                # Оцениваем риск
                risk = self.risk_evaluator.evaluate_risk(
                    adapted_image, situation, similarity
                )
                results.append(f"\nУровень риска: {risk['risk_level']}")
                results.append(f"Оценка риска: {risk['risk_score']:.3f}")
                results.append(f"Рекомендация: {risk['recommendation']}")
                
                if risk['risk_factors']:
                    results.append("Факторы риска:")
                    for factor in risk['risk_factors']:
                        results.append(f"  - {factor}")
                
                # Отображаем результат на карте
                self.display_solution(adapted_image, situation)
                
                results.append(f"\nРешение отображено на карте")
                if adapted_image.solution.get('distance_km'):
                    results.append(
                        f"Расстояние: {adapted_image.solution['distance_km']:.2f} км"
                    )
            else:
                results.append(f"\n✗ Адаптация невозможна в данной области")
                
                # Пробуем сгенерировать новый образ
                context_images = self.knowledge_base.get_images_by_context(
                    detected_context
                )
                new_image = self.generator.generate_new_image(
                    context_images, situation
                )
                
                if new_image:
                    results.append(f"\n✓ Сгенерирован новый образ")
                    self.display_solution(new_image, situation)
                else:
                    results.append(f"\n✗ Генерация нового образа невозможна")
        else:
            results.append(f"\n✗ Подходящий образ не найден")
            results.append("Рекомендуется расширить базу знаний")
        
        self.results_text.setText("\n".join(results))
    
    def _create_target_area(self) -> QgsGeometry:
        # Создание целевой области вокруг маршрута
        if not self.start_point or not self.end_point:
            return None
        
        # Создаем буфер вокруг линии маршрута
        line = QgsGeometry.fromPolylineXY([
            self.start_point.asPoint(),
            self.end_point.asPoint()
        ])
        
        # Буфер 500 метров
        buffer = line.buffer(0.005, 5)  # Приблизительно 500м в градусах
        
        return buffer
    
    def display_solution(self, image: Image, situation: ProblemSituation):
        # Отображение решения на карте
        # Создаем слой для отображения
        layer = QgsVectorLayer(
            "LineString?crs=epsg:4326",
            "Решение",
            "memory"
        )
        
        provider = layer.dataProvider()
        
        # Добавляем поле для расстояния
        provider.addAttributes([
            QgsField("distance_km", QVariant.Double),
            QgsField("confidence", QVariant.Double)
        ])
        layer.updateFields()
        
        # Создаем feature с линией маршрута
        feature = QgsFeature()
        feature.setGeometry(QgsGeometry.fromPolylineXY([
            situation.start_point.asPoint(),
            situation.end_point.asPoint()
        ]))
        
        distance = QgsDistanceArea().measureLine(
            situation.start_point.asPoint(),
            situation.end_point.asPoint()
        )
        
        feature.setAttributes([
            distance / 1000,
            image.solution.get('confidence', 0.5)
        ])
        
        provider.addFeature(feature)
        layer.updateExtents()
        
        # Добавляем слой на карту
        QgsProject.instance().addMapLayer(layer)
        
        # Настраиваем отображение
        symbol = QgsLineSymbol.createSimple({
            'color': 'red',
            'width': '2'
        })
        layer.renderer().setSymbol(symbol)
        
        self.iface.mapCanvas().refresh()
        
        QgsMessageLog.logMessage(
            f"Решение отображено на карте. Расстояние: {distance/1000:.2f} км",
            "LogisticsPlugin", Qgis.Info
        )

# Импорт для QVariant
from PyQt5.QtCore import QVariant

def classFactory(iface):
    # Фабрика плагинов QGIS
    return LogisticsPlugin(iface)