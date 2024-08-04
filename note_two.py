import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity

person = {
  "id": 2,
  "first_name": "Алексеевич",
  "diploms_files": [],
  "surname": "Шешеня",
  "second_name": "Алексеевич",
  "email": "ssssord",
  "grade_up": 0,
  "phone_number": "string",
  "about_me": "сбор, систематизация и проработка требований заказчиков преобразование требований в функциональные и технические спецификации для дальнейшей передачи их в разработку подготовка проектной документации (функциональные требования, технические задания и решения, спецификации, инструкции и т.д.), постановка задач на разработку изучение и анализ источников данных и взаимосвязей, возможностей их извлечения, обогащения и очистки Презентация доработок заказчикам и заинтересованным подразделениям на приемо-сдаточных испытаниях Подготовка прототипов витрин и отчетных форм создание постановки и маппингов S2T. Чистка унитазов, услуги сантехника",
  "professions": [
    {
      "id": 1,
      "profession_name": "системный аналитик DWH"
    },
    {
      "id": 7,
      "profession_name": "Сантехник"
    }
  ]
}

object_list = {
  "objects_constructions": [
    {
      "id": 0,
      "work_name": "Системный аналитик",
      "price": 5001510,
      "work_description": "Уровень: СА уровня Middle+/Senior от 3х лет Возраст: до 50 Локация: РФ Вилка до 350 000 net Этапы собеседований во вложении: Интервью СА этапы Общее описание: Сейчас у нас открыты позиции сразу в нескольких направлениях - Lifestyle-сервисы, банковский бэк (в том числе процессинг, АБС, эквайринг, PRM), SME, Открытие новых продуктов, Инвестиции и многое другое. Задачи СА: Подготовка спецификаций (сбор требований, анализ, документирование постановки) Проектирование интеграционного взаимодействия Помощь команде на стадии разработки и тестирования Взаимодействие с аналитиками и разработчиками смежных подразделений часто, но не всегда - работа с БД Основные требования к системным аналитикам: Актуальный релевантный опыт от 3 лет Проектирует интеграции (API и брокеры сообщений) Составляет ТЗ и занимается сбором требований как плюс (но на самом деле это тоже очень важно и без этого вряд ли позовут на техничку, особенно хороший уровень SQL!!!) Работает с SQL Был опыт проектирования баз данных Имеет опыт проектирования пользовательских интерфейсов Технологии/инструменты Микросервисная архитектура; REST/gRPC; Kafka/RabbitMQ; SQL; UML/BPMN Детали могут варьироваться от проекта Знание какого-либо языка программирования и умение разбираться в коде будет плюсом Кто точно не подходит? аналитики данных/ BI/ ETL/ DWH – всё что в сторону Big Data, упор на работу с БД, построение моделей менеджеры-эксперты-руководители-начальники - больше про управление и контроль сроков и задач бизнес-аналитики/технологи/консультанты - больше про работу с заказчиком, мало или ничего про интеграции, может быть опыт с БД кандидаты с большой разницей во времени (максимум +4ч от Мск, но важно уточнять готов ли кандидат двигать начало раб.дня) Дополнение по портрету кандидата: Кандидаты из аутсорса, аутстаффа и интеграторов должны обладать 2 годами опыта работы в продуктовых компаниях. Идеально: 2 года в продуктовой компании последним местом работы. Остальное будет рассматриваться кейсово. Компании-доноры: банки, финтех, страховые (наши конкуренты) крупные системные интеграторы (EPAM, Luxoft, Accenture, Bell, Jett, AT Consulting и т.д.) высокотехнологичные компании ( , Avito,   и т.д.) с осторожностью - нефтегаз, ретейл ( ), зависит от задач НЕ СМОТРИМ - НИИ, ВУЗы (некоммерческие проекты или pet-проекты), «оборонка», «госуха» (работа по ГОСТам, приемка и тд - мало общего с задачами наших СА) Где ещё почитать про СА:   внимание, тайминги секций указаны устаревшие (1час), обе секции сейчас 1,5 часа идут   это лендинг, где есть информация о стеке, технологиях, требованиях к соискателю и возможностях роста. Нужно сохранить в отдельный файл, ссылку кандидатам не давать.",
      "available_vacancies": 2,
      "professions": [
        {
          "id": 1,
          "profession_name": "Аналитик"
        },
        {
          "id": 2,
          "profession_name": "Системный аналитик"
        }
      ]
    },
    {
      "id": 1,
      "work_name": "Product manager",
      "price": 0,
      "work_description": "Продактов в компании сейчас порядка 250, все вертикали, как и сам   активно растут и занимают лидирующие позиции по рынку (Например, Недвижимость в январе 2023г. обогнала конкурентов и стала лидером на рынке). Как можно допродать: - Большое количество лучших специалистов, в т.ч. с международной экспертизей - В июле   занял 2 место в рейтинге самых привлекательных работодателей среди продактов ( - Огромная структура, на базе которой постоянно запускаются новые продукты (есть возможность как развивать уже готовое, так и создавать с “0”) - Крутая практика менторства, очень активно растут продакты, полугодовые перфоманс ревью Откуда смотрим: - глобально, компании с крутыми и популярными многопользовательскими продуктами, лидеры рынка (особенно ecom), можно с бэкграундом в стартапах, но чтоб не только они - Другие маркетплейсы (Озон,  мегамаркет, Ламода, Алиэкспресс и проч.) / классифайды (Циан, HH, Aviasales, Работа.Ру, Суперджоб и проч.) - Из Банков - хорошо заходят Тинёк и  , можно с бэкграундом оттуда (остальные банки лучше не нужно - только если человек ну супер-движовый и есть крутые результаты, а в банк его как-то занесло - например, в СБЕР и Точка) - Компании с крутой продуктовой культурой (Skyeng, Epam, ex. VK/Mail.ru, ex.  ) - Крупный телеком case by case, но важно, чтоб кандидаты были скилловые * под специфичные позиции можно смотреть, например, компании, где крутой биллинг или же идти по Аналогии с вертикалями   (нужен человек в Авто, смотрим по рынку компании с Авто и т.д.) Точно нет : Всё, что с негибкой структурой и отсутствием нацеленности на результат (госуха, банки и проч). Что важно: - для Lead/Head и выше важен +- хороший английский, т.к. есть топы-иностранцы, которые говорят плохо по-русски и есть документация на английском (формата 6 page) Этапы интервью (стандартный флоу в   для Middle/Senior, где мы сначала идём без конкретной вакансии, ввиду специфичности опыта или Head/Cluster Lead, можно сразу идти под конкретную вакансию): HR-блок 1) Скрининг с Sense (узнаём о команде, подчинении, с какими метриками работал, делал ли сам аналитику, проводил ли А/Б тесты, результаты по продуктам, в целом) Направляем в чат резюме + сопровод или файл скрининга ( ) !!! Деньги указываем только в сопроводе, из резюме и скрининга убираем 2) HR-скоринг  на час, в т.ч. проверяет харды !!!Технический блок - раньше было 4 секции, о которых продакты наслышаны (харды, кейс, аналитика и UX), сейчас процесс оптимизировали: 1) скоринг по оптимизации продукта на 30 мин с продактам грейда Senior+ 2) 1,5ч кейс с Продактом Sen+ и Аналитиком Sen+ в формате ролевой игры) 3) HR-скоринг на Culture Fit - проверка на то, как взаимодействует с командой (даже если продакт круто прошёл техничку, ему могут отказать по калча фиту - поэтому нужны люди структурные, с классными софтами, проактивные и готовые драйвить, с “горящими глазами”) - 1-1,5 ч. И наконец - знакомство с командами (обычно 2-3 команды, чтобы было из чего выбрать) Вилки примерные: * звёздам готовы согласовывать больше - Middle - 350/руки max - Senior - до 500 гросс/ либо совокуп - Lead - 560-630 гросс max + LTI - Бонусы у продактов полугодовые как и перформанс ревью (есть возможность растить з/п, если заперформить) Формат работы + оформление: Офисы: Москва/казань/питер/самара Готовы гибрид/полную удалёнку (в том числе из любой страны, если человек приехал и оформился в офис  , может уезжать обратно) есть армянское юр лицо туда сейчас офорляют от грейда Senior+ и если критичная ситуация (например, потеря резидентства) Если локация спорная, уточняем, есть ли критичные аспекты по трудоустройству С СНГ ребят устраивают, если есть разрешение на работу в РФ  описание",
      "available_vacancies": 1,
      "professions": [
        {
          "id": 3,
          "profession_name": "Product manager"
        }
      ]
    },
    {
      "id": 2,
      "work_name": "DevOps инженер",
      "price": 1560154,
      "work_description": "писание профиля (требования) Требуемый опыт работы: 2-4 года опыта в DevOps, 1–2 года в банковской сфере Полная занятость, полный день. Мы ищем DevOps инженера с опытом администрирования enterprise инфраструктуры в технологическую команду развития Business critical систем банка. В задачи кандидата входит развитие DevOps окружения проекта с использование современных технологий и практик. Приветствуется наличие опыта самостоятельного проектирования и развертывания CI / CD пайпов для Java-приложений. Личные качества: хорошие коммуникативные навыки, внимательность к деталям, аккуратность, работа на результат. Технические компетенции: опыт реализации решений на JVM стеке в существующем IT-ландшафте; TeamCity / Jenkins, Helm / Ansible, Kubernetes / OpenShift + ServiceMesh / Istio, Docker + Docker Compose + Dockerfile, Nexus, Maven, BitBucket, Jira, Bash / Python / Kotlin / Groovy, Git, Terraform, Wildfly / Tomcat, SQL + PostgreSQL, S3, Prometheus, Grafana, ELK (logstash, fluentb, Kibana), REST API, OSI. Основные задачи, которые предстоит решать сотруднику) - развитие DevOps окружения проекта, включая: - администрирование тестовых сред (работа с сертификатами, mTLS, конфигурирование серверов в кластерах Kubernetes, S3); - адаптация и развитие CI / CD pipeline-а на TeamCity и Jenkins с учетом реалий разработки; - проектирование инфраструктуры проекта с упором на надежность, отказоустойчивость, с учетом требований ИБ; - аллокация облачных ресурсов; - разработка Helm chart-ов, Ansible playbook-ов, настройка sidecar-ов; - настройка ELK;",
      "available_vacancies": 2,
      "professions": [
        {
          "id": 0,
          "profession_name": "Инженер-строитель"
        },
        {
          "id": 1,
          "profession_name": "Инжинер, как инжир"
        }
      ]
    },
    {
      "id": 5,
      "work_name": "Укладка плитки",
      "price": 4500,
      "work_description": "Требуется плиточники на объект",
      "available_vacancies": 15,
      "professions": [
        {
          "id": 5,
          "profession_name": "Плиточник"
        }
      ]
    },
    {
      "id": 6,
      "work_name": "Требуется бригада рабочих",
      "price": 20000,
      "work_description": "Требуется плиточники, сварщики, маляры и прорабы на объекты",
      "available_vacancies": 50,
      "professions": [
        {
          "id": 5,
          "profession_name": "Плиточник"
        },
        {
          "id": 4,
          "profession_name": "Прораб"
        },
        {
          "id": 6,
          "profession_name": "Маляр"
        },
        {
          "id": 1,
          "profession_name": "Каменщик"
        }
      ]
    }
  ]
}


def vectorize_and_rank(input_sentence, sentences, titl):
    # Векторизуем все предложения (включая входное)
    all_sentences = [input_sentence] + sentences
    count_vectorizer = CountVectorizer()
    sentence_vectors = count_vectorizer.fit_transform(all_sentences)

    # Вычисляем косинусное сходство между входным предложением и всеми остальными
    similarities = cosine_similarity(sentence_vectors[0], sentence_vectors[1:]).flatten()

    # Создаем упорядоченный ранжированный список
    ranked_sentences = sorted(list(zip(tuple(titl), similarities)), key=lambda x: x[1], reverse=True)

    return [x[0] for x in ranked_sentences]


def recommendation(sentences, titl):
    # Вычисляем косинусное сходство между входным предложением и всеми остальными
    similarities = cosine_similarity([sentences[0]], sentences[1:]).flatten()
    # Создаем упорядоченный ранжированный список
    ranked_sentences = sorted(list(zip(titl, similarities)), key=lambda x: x[1], reverse=True)
    return [x[0] for x in ranked_sentences]


#----------------------------------ОБНОВЛЕНИЕ------------------------------------------------------


def parsing_person(path):
    # в data можно поместить словарь с одним пользователем или считать из json файла, который подадут на вход
    # data = pd.read_json(path)
    data = path
    result = ' '
    for i in range(len(data['professions'])):
        result += str(data['professions'][i]['profession_name']) + " "
    result += str(data['about_me'])
    return result


def parsing_objects(path):
    # в data можно поместить словарь с вакансиями или считать из json файла, который подадут на вход
    # data = pd.read_json(path)
    data = path
    result = pd.DataFrame(columns=['0'])
    title = pd.DataFrame(columns=['0'])
    id_lst = pd.DataFrame(columns=['0'])
    for i in range(len(data['objects_constructions'])):
        id_lst.loc[len(id_lst.index)] = data['objects_constructions'][i]['id']
        title.loc[len(title.index)] = data['objects_constructions'][i]['work_name']
        s = ''
        for j in range(len(data['objects_constructions'][i]['professions'])):
            s += data['objects_constructions'][i]['professions'][j]['profession_name'] + " "
        result.loc[len(result.index)] = str(data['objects_constructions'][i]['work_name']).lower() + " " + s + str(data['objects_constructions'][i]['work_description']).lower()
    # возвращает датафрейм, хранящий id, название профессии и её описание
    dt = pd.DataFrame({'id': id_lst['0'].tolist(), 'title': title['0'].tolist(), 'vacancy': result['0'].tolist()})
    return dt


def back_to_back(lst, path):
    # повторно считываю весь словарь профессий, его можно напрямую в data сюда передать
    # data = pd.read_json(path)
    data = path
    res = {'objects_constructions': []}
    for el in lst:
        for c in range(len(data['objects_constructions'])):
            if data['objects_constructions'][c]['id'] == el:
                res['objects_constructions'].append(data['objects_constructions'][c])
                break
    # возвращает отсортированный словарь профессий, по списку подходящих пользователю вакансий
    return res


# Вот здесь в input_sentence загоняешь сторку формата "название_профессии краткое_описание"
input_sentence = parsing_person(person)
print(input_sentence)
# эту не меняешь, здесь все вакансии, среди которых будет поиск (ну либо меняешь, но в том же формате)
base = parsing_objects(object_list)
# base содержит и title и other_sentences
other_sentences = base['vacancy'].tolist()
title = dict(zip(base['id'].tolist(), base['title'].tolist()))
ranked_sentences = vectorize_and_rank(input_sentence, other_sentences, title)

print(f'Result: ', ranked_sentences, '\n')
for idx, sen in enumerate(ranked_sentences):
    print(f"{idx+1}. {title[sen]}")
    # можно break убрать, но будет выводить все 200+ строк
    if idx == 15:
        break

print(back_to_back(ranked_sentences, object_list))

print('\n////////////////////RuBERT////////////////////////////\n')

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('cointegrated/rubert-tiny2')

input_sentence = parsing_person(person)
base = parsing_objects(object_list)
other_sentences = base['vacancy'].tolist()
titl_ = dict(zip(base['id'].tolist(), base['title'].tolist()))
sentence = [input_sentence] + other_sentences
sentence_vectors = model.encode(sentence)

ranked_sentences_1 = recommendation(sentence_vectors, titl_)
print(f'Result: ', ranked_sentences_1, '\n')
for idx, sentence in enumerate(ranked_sentences_1):
    print(f"{idx+1}. {titl_[sentence]}")
    if idx == 25:
        break

print(back_to_back(ranked_sentences, object_list))

print('\n////////////////////LaBSE////////////////////////////\n')

from sentence_transformers import SentenceTransformer
model1 = SentenceTransformer('sentence-transformers/LaBSE')

input_sentence = parsing_person(person)
base = parsing_objects(object_list)
other_sentences = base['vacancy'].tolist()
titl_ = dict(zip(base['id'].tolist(), base['title'].tolist()))
sentence = [input_sentence] + other_sentences
sentence_vectors = model1.encode(sentence)

ranked_sentences_1 = recommendation(sentence_vectors, titl_)
print(f'Result: ', ranked_sentences_1, '\n')
for idx, sentence in enumerate(ranked_sentences_1):
    print(f"{idx+1}. {titl_[sentence]}")
    if idx == 25:
        break

print(back_to_back(ranked_sentences, object_list))