
<p align="center">
     <H2 align="center">Команда Neutral Freaks</H2> 
    <H2 align="center">Автоматизированная путевая машинная станция</H2> 
</p>

## Описание решения
Команда Neutral Hub представляет программный модуль определения дефектных ноутбуков по фотографиям на основе сервисной информации. Решение позволяет:
- автоматически находить и классифицировать дефекты на оборудовании по изображениям;
- формировать отчёт о наличии дефектов продукции с указанием его серийного номера, выводить его в Web-интерфейс и сохранять в удобный формат;
- снизить нагрузку на специалиста по контролю, исключив случаи невнимательности или неопытности, позволяя переключиться на выполнение других задач;
- облегчить задачу осмотра устройств в удалённых локациях и производить его в кратчайшие сроки.

Программный модуль в автоматическом формате находит и классифицирует дефекты на оборудовании по изображениям. Web-интерфейс позволяет удобно производить контроль распознавания и визуализирует отчёт. Модуль машинного распознавания обладает широким набором видов дефектов и постоянно обучается, получая новые данные.

## Особенности
- Эргономика и автономность. Постоянное самообучение и автономная адаптация нейронной сети под новые виды дефектов с повышением качества классификации. Модульная архитектура проекта позволяет гибко настраивать требуемые инструменты, а применение Docker создаёт гармоничное сочетание удобства развёртывания при высокой производительности и надёжности
-  Наглядность и функциональность. Интуитивный Web-интерфейс предоставляет возможность удобного ручного контроля операций распознавания с аннотациями на изображениях, добавлять пользовательские классы дефектов и формировать отчёт по результатам детекции в удобный формат. Система также предоставляет аналитический дашборд для специалистов, отслеживающий статистику по частоте дефектов
- Интеграция с ERP. Система имеет возможность интеграции с системами учета для автоматического сохранения результатов проверки и создания статистических отчетов на базе всех обработанных устройств


## Используемые технологии

Интерфейс: Streamlit

Серверная часть: Streamlit

Нейронная сеть: YOLO11 (torch, opencv)

База данных: PostgreSQL


## Запуск проекта

Для запуска проекта локально требуется:




## Запуск проекта на виртуальной машине:

Для запуска проекта на виртуальной машине выполните следующие шаги:







### Пример работы нашего приложения вы можете увидеть на данном видео.






