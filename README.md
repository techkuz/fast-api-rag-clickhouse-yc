## Архитектура RAG (Retrieval Augmented Generation) приложения

### Абстрактная архитектура RAG 
![Rag architecture abstract](https://github.com/techkuz/fast-api-rag-clickhouse-yc/blob/main/images/RAG-base.png "Rag architecture abstract")

### Архитектура RAG on-premise
![Rag architecture on-premise](https://github.com/techkuz/fast-api-rag-clickhouse-yc/blob/main/images/RAG-on-premise.png "Rag architecture on-premise")

Веб сервер реализован на fastapi с использованием Clickhouse в качестве хранения векторов. В качестве LLM используется YaGPT через фреймворк langchain, а исходный датасет находится в s3 хранилище в формате CSV.

### Архитектура RAG в Yandex Cloud
![Rag architecture Yandex Cloud](https://github.com/techkuz/fast-api-rag-clickhouse-yc/blob/main/images/rag-yc.png  "Rag architecture Yandex Cloud")

Веб сервер на fastapi, размещенный на compute cloud. В качестве LLM используется YaGPT с интеграцией через фреймворк langchain. Векторы хранятся в managed Clickhouse, а исходный датасет - в object storage в формате CSV. Для показа функциональности используется datasphere

## Пример использования системы
Разработчики хотят использовать LLM для обновления существующей кодовой базы. Однако обычные запросы не позволяют получить актуальные данные, например,о последних обновлениях библиотек, таких как Pytorch 2.3.0. 

## Цель данного проекта
Показать возможность построения RAG на YaGPT. 

#### Описание папок:
* images - картинки для репозитория
* rag_usage_onpremise - пример использования приложения через API запросы на fastapi веб-сервер (содержит ноутбук и requrements, а также примеры возможных запросов по API)
* rag-usage_yc - пример построения приложения с использованием облачных ресурсов и явное использование тех составных частей кода, которые скрыты в onpremise демо
* terraform - код для создания облачных ресурсов
* web-server - код веб-сервера RAG приложения на fastapi
* generate_data.sql - скрипт для генерации исходного датасета pytorch 2.3.0 changelog (если необходимо, но необязательно) через clickhouse как ETL в object storage

#### Комментарии
* Переработать логику передачи docsearch в случае переиспользования кода (сейчас он сохраняется в non-volatile storage)

#### Запуск terraform в YC
1. Получить ключ для сервисного аккаунта в формате json https://cloud.yandex.com/en-ru/docs/iam/concepts/users/service-accounts
2. Подготовить датасет в CSV (либо спарсить вручную, либо использовать generate_data.sql)
3. Создать community в datasphere 
4. Добавить к community DS сервисный аккаунт из п1
5. Заполнить все значения в variables.tf
6. Запустить terraform init и terraform apply
7. Загрузить датасет в поднятый object-storage
8. Выполнять запросы из ноутбука

#### Запуск on-premise
1. Загрузить датасет в поднятое object-storage хранилище
2. Запустить fastapi сервер в режиме отладки (!) ```fastapi dev main.py```
3. Развернуть clickhouse, например, с помощью https://hub.docker.com/r/clickhouse/clickhouse-server/
4. Заполнить креды в on-premise ноутбуке
5. Выполнять запросы из ноутбука

#### Полезные ссылки:
* https://github.com/yandex-datasphere/yandexgpt-qa-scenarios/
* https://github.com/dzhechko/yagpt-rag-bot
* https://github.com/pytorch/pytorch/releases/tag/v2.3.0
* https://python.langchain.com/v0.1/docs/integrations/vectorstores/clickhouse/
* https://python.langchain.com/v0.1/docs/integrations/text_embedding/yandex/
* https://python.langchain.com/v0.1/docs/integrations/llms/yandex/

Версия Python для показа: 3.11.5  