import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

DB_CONNECTION = None
DB_CURSOR = None

DATABASE_KWARGS = {
    "dbname": "postgres",
    "user": "postgres",
    "password":os.getenv("password"),
    "host": "localhost",
    "port": "5432",  # Обычно 5432 - стандартный порт для PostgreSQL
}


def add_event(camera_name, track_name, object_name, event_type, event_time, coefficient):
    try:
        # Устанавливаем соединение с базой данных
        conn = psycopg2.connect(**DATABASE_KWARGS)
        # Создаем курсор для выполнения SQL-запроса
        cursor = conn.cursor()

        # SQL-запрос на вставку данных
        insert_query = """
        INSERT INTO events (camera_name, track_name, object_name, event_type, event_time, coefficient)
        VALUES (%s, %s, %s, %s, %s, %s)
        """

        # Выполняем запрос с переданными данными
        cursor.execute(insert_query, (camera_name, track_name,
                       object_name, event_type, event_time, coefficient))

        # Подтверждаем изменения в базе данных
        conn.commit()
        print("Запись успешно добавлена в таблицу events.")

    except Exception as e:
        print("Ошибка при добавлении записи:", e)
        raise e
    
    finally:
        # Закрываем курсор и соединение
        cursor.close()
        conn.close()


def clear_events_table():
    try:
        # Устанавливаем соединение с базой данных
        conn = psycopg2.connect(**DATABASE_KWARGS)
        # Создаем курсор для выполнения SQL-запроса
        cursor = conn.cursor()

        # Выполняем SQL-команду для очистки таблицы
        cursor.execute("TRUNCATE TABLE events RESTART IDENTITY;")

        # Подтверждаем изменения
        conn.commit()
        print("Таблица events успешно очищена.")

    except Exception as e:
        print("Ошибка при очистке таблицы:", e)

    finally:
        # Закрываем курсор и соединение
        cursor.close()
        conn.close()

def get_data_query_result(query : str):

    try:
        # Устанавливаем соединение с базой данных
        conn = psycopg2.connect(**DATABASE_KWARGS)
        # Создаем курсор для выполнения SQL-запроса
        cursor = conn.cursor()
        
        cursor.execute(query)
        result = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(result, columns=columns)

        return df

    except Exception as e:
        print("Ошибка при добавлении записи:", e)
        raise e

    finally:
        # Закрываем курсор и соединение
        cursor.close()
        conn.close()        

def recreate_events_table():
    """
    Функция удаляет таблицу events, если она существует, и создаёт её заново в базе данных PostgreSQL.

    Параметры:
    - db_name: Имя базы данных
    - user: Имя пользователя
    - password: Пароль пользователя
    - host: Хост базы данных
    - port: Порт базы данных
    """
    try:
        # Подключаемся к базе данных
        connection = psycopg2.connect(**DATABASE_KWARGS)
        cursor = connection.cursor()

        # Удаляем таблицу events, если она существует
        drop_table_query = "DROP TABLE IF EXISTS events;"
        cursor.execute(drop_table_query)

        # Создаём таблицу events заново
        create_table_query = """
        CREATE TABLE events (
            event_id SERIAL PRIMARY KEY,
            camera_name VARCHAR(255),
            track_name VARCHAR(255),
            object_name VARCHAR(255),
            event_type VARCHAR(255),
            event_time TIMESTAMP,
            coefficient FLOAT
        );
        """
        cursor.execute(create_table_query)
        connection.commit()
        print("Таблица events успешно удалена и создана заново.")

    except (Exception, psycopg2.DatabaseError) as error:
        print(f"Ошибка при обновлении таблицы: {error}")

    finally:
        # Закрываем соединение
        if connection:
            cursor.close()
            connection.close()