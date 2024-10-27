import psycopg2
import pandas as pd

DB_CONNECTION = None
DB_CURSOR = None

DATABASE_KWARGS = {
    "dbname": "postgres",
    "user": "postgres",
    "password": "",
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
