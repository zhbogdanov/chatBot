import collections
import itertools
import logging
import re
from typing import List, Tuple, Set
from collections import OrderedDict
from datetime import datetime, timedelta

import numpy as np
import pymorphy2
from aiogram import Bot, Dispatcher, executor, types
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from aiogram.dispatcher.filters.state import State, StatesGroup
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import time


TOKEN = "6105040822:AAFp2dMIhgTa7sUeiKTYPuGtH3Rl7PEzUic"

logging.basicConfig(level=logging.INFO)

bot = Bot(token=TOKEN)
dp = Dispatcher(bot=bot, storage=MemoryStorage())


syn_sets = None
small_db = None
similarity = None
date_time_pattern = None


class SmallDB:
    """Небольшая NoSQL базу данных"""
    __small_nosql_db = {}

    def __init__(self, user_id: int):
        self.__user_id = user_id
        self.__small_nosql_db[user_id] = {}
        self.__last_event = None

    def write_event(self, event_name: str, dt: datetime) -> None:
        """Запись события в данные пользователя в базе данных"""
        self.__small_nosql_db[self.__user_id][event_name] = dt

    def get_events(self) -> dict:
        """Получение всех событий пользователя из базы данных"""
        return self.__small_nosql_db[self.__user_id]

    def write_last_event(self, event_name: tuple) -> None:
        """Запись последнего названия события в переменную экземпляра"""
        self.__last_event = event_name

    def get_last_event(self) -> Tuple[str, str]:
        """Получение последнего названия события из переменной экземпляра"""
        return self.__last_event

    def delete_event(self, event_name: str) -> bool:
        """Удаление события из данных пользователя в базе данных"""
        if event_name in self.__small_nosql_db[self.__user_id]:
            del self.__small_nosql_db[self.__user_id][event_name]
            return True

        return False


class SynonymsSets:
    add_event_synonyms = {
        "Добавь событие",
        "Создать событие",
        "Запланировать событие",
        "Добавить в календарь",
        "Отметить дату в календаре",
        "Добавь дедлайн",
        "Установить крайний срок",
        "Задать срок",
        "Определить дату окончания",
        "Назначить дату завершения",
        "Указать последний день для выполнения задачи",
        "Запланировать встречу",
        "Создать встречу",
        "Добавить встречу в календарь",
        "Назначить встречу",
        "Добавь встречу",
        "Поставь уведомление",
        "Поставить уведомление",
        "Добавь уведомление",
        "Добавить уведомление",
    }

    delete_meeting_synonyms = {
        "Отменить встречу",
        "Удалить встречу",
        "Убрать встречу",
        "Отменить запланированную встречу",
        "Удали событие",
        "Удалить событие",
        "Убрать событие",
        "Уничтожить событие",
        "Отменить событие",
        "Удали дедлайн",
        "Удалить дедлайн",
        "Убрать дедлайн",
        "Уничтожить дедлайн",
        "Отменить дедлайн",
        "Отмени дедлайн",
        "Удали уведомление",
        "Удалить уведомление",
    }

    errors_synonyms = {
        "Неверное событие",
        "Неправильное событие",
        "Ошибочное событие",
        "Некорректное событие",
        "Ошибка в названии события",
        "Ошибочное название события",
        "Некорректное название события",
    }

    display_events_synonyms = {
        "Покажи все события",
        "Покажи мне все мероприятия",
        "Выведи список всех событий",
        "Отобрази все события",
        "Выведи список событий",
        "Покажи мне список мероприятий",
        "Отобрази все мероприятия",
        "Выведи мне список событий",
        "Покажи календарь",
        "Открой календарь",
        "Покажи мне календарь",
        "Выведи календарь",
    }

    def __init__(self):
        """Список имен множеств"""
        self.__all_synonyms = set(
            filter(
                lambda attr: attr.endswith('_synonyms') and not callable(getattr(self, attr)),
                dir(self)
            )
        )
        self.lemmatize_sets()
        """Список всех элементов множеств"""
        self.__all_synonyms_attr_tuple = tuple(getattr(self, synonym) for synonym in self.__all_synonyms)

    def get_all_synonyms(self) -> List[str]:
        """Возвращает список всех синонимов в виде списка строк."""
        return list(itertools.chain.from_iterable(self.__all_synonyms_attr_tuple))

    def get_attr_attr_name_by_element(self, element: str) -> Tuple[Set[str], str]:
        """Возвращает имя атрибута, содержащего элемент, и его значение в виде кортежа."""
        for synonym in self.__all_synonyms:
            if element in getattr(self, synonym):
                return getattr(self, synonym), synonym

    def get_synonyms_attr_tuple(self) -> Tuple[Set[str]]:
        """Возвращает кортеж атрибутов, содержащих синонимы."""
        return self.__all_synonyms_attr_tuple

    def lemmatize_sets(self) -> None:
        """Лемматизирует все множества синонимов."""
        for attr in self.__all_synonyms:
            setattr(self, attr, Lemma.lemmatize_set(getattr(self, attr)))

    def write_syn_to_add_set(self, syn: str):
        """Добавляет синоним в множество add_event_synonyms."""
        self.add_event_synonyms.add(syn)

    def delete_syn_from_add_set(self, syn: str) -> bool:
        """Удаляет синоним из множества add_event_synonyms."""
        if syn in self.add_event_synonyms:
            self.add_event_synonyms.remove(syn)
            return True

        return False


class Similarity:
    tf_idf_vectorizer = TfidfVectorizer()

    def __init__(self, metric: str):
        metrics = Similarity.get_metrics()
        self.metric_func = metrics[metric]

    def find_synonym_set(self, input_text: str, synonym_sets: SynonymsSets) -> Tuple[set, Set[str]]:
        """Находит множество синонимов, наиболее похожее на входной текст."""
        return self.metric_func(input_text, synonym_sets)

    @staticmethod
    def get_metrics() -> dict:
        """Возвращает словарь с доступными метриками для измерения схожести."""
        return {
            'tanimoto': Similarity.__find_synonym_set_tanimoto,
            'tf-idf': Similarity.__find_synonym_set_tfidf,
            'bm25': Similarity.__find_synonym_set_bm25,
        }

    @staticmethod
    def __tanimoto_func(s1: List[str], s2: List[str]) -> float:
        """Вычисляет коэффициент Танимото для двух списков строк."""
        set1, set2 = set(s1), set(s2)
        intersection, union = set1 & set2, set1 | set2
        return len(intersection) / len(union)

    @staticmethod
    def __find_matched_tokens(
            input_text: str,
            most_similar_phrase: str,
            synonym_sets: SynonymsSets
    ) -> Tuple[str, Set[str]]:
        """Находит совпадающие токены между входным текстом и наиболее похожим набором синонимов."""
        # Получение наиболее похожего набора синонимов и его имя
        most_similar_set, most_similar_set_name = synonym_sets.get_attr_attr_name_by_element(most_similar_phrase)

        matched_tokens = set()
        input_tokens = input_text.split()

        # Поиск совпадающих токенов
        for key in most_similar_set:
            for token in input_tokens:
                if token in key.split():
                    matched_tokens.add(token)

        # Возврат имени наиболее похожего набора синонимов и совпадающих токенов
        return most_similar_set_name, matched_tokens

    @staticmethod
    def __find_synonym_set_tanimoto(input_text: str, synonym_sets: SynonymsSets) -> Tuple[str, Set[str]]:
        """Находит наиболее похожий набор синонимов на основе коэффициента Танимото."""
        all_synonyms = synonym_sets.get_all_synonyms()
        similarity_scores = []

        # Расчет коэффициента Танимото для каждого набора синонимов
        for synonyms in all_synonyms:
            similarity_score = Similarity.__tanimoto_func(input_text.split(), synonyms.split())
            similarity_scores.append(similarity_score)

        # Нахождение наиболее похожего набора синонимов
        most_similar_phrase = all_synonyms[similarity_scores.index(max(similarity_scores))]

        # Возврат совпадающих токенов между входным текстом и наиболее похожим набором синонимов
        return Similarity.__find_matched_tokens(input_text, most_similar_phrase, synonym_sets)

    @staticmethod
    def __find_synonym_set_tfidf(input_text: str, synonym_sets: SynonymsSets) -> Tuple[str, Set[str]]:
        """Находит наиболее похожий набор синонимов на основе TF-IDF векторов."""
        all_synonyms = synonym_sets.get_all_synonyms()
        all_texts = all_synonyms + [input_text]
        text_vectors = Similarity.tf_idf_vectorizer.fit_transform(all_texts)

        # Вычисление сходства между входным текстом и наборами синонимов на основе TF-IDF векторов
        similarity_scores = cosine_similarity(text_vectors)[-1][:-1]

        # Нахождение наиболее похожего набора синонимов
        most_similar_phrase = all_synonyms[similarity_scores.argmax()]

        # Возврат совпадающих токенов между входным текстом и наиболее похожим набором синонимов
        return Similarity.__find_matched_tokens(input_text, most_similar_phrase, synonym_sets)

    @staticmethod
    def __find_synonym_set_bm25(input_text: str, synonym_sets: SynonymsSets) -> Tuple[str, Set[str]]:
        """Находит наиболее похожий набор синонимов на основе BM25."""
        all_synonyms = synonym_sets.get_all_synonyms()
        tokenized_synonyms = [synonyms.split() for synonyms in all_synonyms]

        # Создание объекта BM25Okapi с токенизированными наборами синонимов
        bm25 = BM25Okapi(tokenized_synonyms)

        # Токенизация входного текста
        tokenized_input = input_text.split()

        # Вычисление сходства между входным текстом и наборами синонимов с помощью BM25
        similarity_scores = bm25.get_scores(tokenized_input)

        # Нахождение наиболее похожего набора синонимов
        most_similar_phrase = all_synonyms[np.argmax(similarity_scores)]

        # Возврат совпадающих токенов между входным текстом и наиболее похожим набором синонимов
        return Similarity.__find_matched_tokens(input_text, most_similar_phrase, synonym_sets)


class DateTimePattern:
    def __init__(self):
        self.__patterns = set(
            filter(
                lambda attr: attr.endswith('_pattern'),
                dir(self)
            )
        )

    @staticmethod
    def __date_time_pattern(text: str):
        """Распознает дату и время в тексте, согласно заданному шаблону."""
        now = datetime.now()

        pattern = r'(на сегодня|на завтра|сегодня|завтра)\s*(на|в)?\s*(\d{1,2}(:\d{1,2})?)\s*(утра|дня|вечера)?'

        match = re.search(pattern, text.lower())

        if match:
            day = match.group(1)
            if 'сегодня' in day:
                date = now.date()
            else:
                date = (now + timedelta(days=1)).date()

            time_str = match.group(3)
            if ':' in time_str:
                time_format = '%H:%M'
            else:
                time_format = '%H'
            time = datetime.strptime(time_str, time_format).time()

            if day_part := match.group(5):
                if ('вечера' in day_part or 'дня' in day_part) and time.hour < 12:
                    time = time.replace(hour=time.hour + 12)

            datetime_obj = datetime.combine(date, time)

            return datetime_obj, match.group()

        return None, None

    @staticmethod
    def __weekend_time_pattern(text: str):
        """Распознает дату и время в тексте, согласно заданному шаблону."""
        now = datetime.now()

        pattern = r'(на|в этот|в эту|в следующий|в следующую|на эту|на следующую)\s*' \
                  r'([а-яё]+)\s*(на|в)?\s*(\d{1,2}(:\d{1,2})?)\s*(утра|дня|вечера)?'

        match = re.search(pattern, text)

        if match:
            delta_range = 1 if 'следующ' in match.group(1) else 0
            weekday = match.group(2)
            weekday_num = ['понедельник', 'вторник', 'среду', 'четверг', 'пятницу', 'субботу', 'воскресенье'].index(
                weekday.lower())
            date = (now + timedelta(days=(weekday_num - now.weekday() + 7) % 7) + timedelta(
                days=delta_range * 7)).date()

            time_str = match.group(4)
            if ':' in time_str:
                time_format = '%H:%M'
            else:
                time_format = '%H'
            time = datetime.strptime(time_str, time_format).time()

            if day_part := match.group(6):
                if ('вечера' in day_part or 'дня' in day_part) and time.hour < 12:
                    time = time.replace(hour=time.hour + 12)

            datetime_obj = datetime.combine(date, time)
            return datetime_obj, match.group()

        return None, None

    @staticmethod
    def __month_time_pattern(text: str):
        """Распознает дату и время в тексте, согласно заданному шаблону."""
        now = datetime.now()

        pattern = r'на\s*(\d{1,2})\s*([а-яё]+)\s*(на|в)?\s*(\d{1,2}(:\d{1,2})?)?\s*(утра|дня|вечера)?'

        match = re.search(pattern, text)

        if match:
            day = int(match.group(1))
            month = match.group(2)
            month_num = ['января', 'февраля', 'марта', 'апреля', 'мая', 'июня', 'июля', 'августа', 'сентября',
                         'октября', 'ноября', 'декабря'].index(month.lower()) + 1
            year = now.year

            if month_num < now.month or (month_num == now.month and day < now.day):
                year += 1
            date = datetime(year, month_num, day).date()

            if match.group(4):
                if ':' in match.group(4):
                    time_format = '%H:%M'
                else:
                    time_format = '%H'
                time = datetime.strptime(match.group(4), time_format).time()

                if day_part := match.group(6):
                    if ('вечера' in day_part or 'дня' in day_part) and time.hour < 12:
                        time = time.replace(hour=time.hour + 12)
            else:
                time = datetime.strptime('00:00', '%H:%M').time()

            datetime_obj = datetime.combine(date, time)
            return datetime_obj, match.group()

        return None, None

    def get_date_time(self, text: str):
        """Извлекает дату и время из текста, используя доступные шаблоны."""
        for pattern in sorted(self.__patterns):
            dt, extracted_pattern = getattr(self, pattern)(text)
            if dt is not None and extracted_pattern is not None:
                return dt, extracted_pattern
        return None, None


class Lemma:
    morph = pymorphy2.MorphAnalyzer()

    @staticmethod
    def lemmatize_word(word: str) -> str:
        """Лемматизирует отдельное слово."""
        parsed_word = Lemma.morph.parse(word)[0]
        return parsed_word.normal_form

    @staticmethod
    def lemmatize_sentence(sentence: str) -> str:
        """Лемматизирует предложение."""
        return ' '.join([Lemma.lemmatize_word(word) for word in sentence.split()])

    @staticmethod
    def lemmatize_set(inp_tuple: tuple) -> set:
        """Лемматизирует набор слов."""
        return set(Lemma.lemmatize_sentence(word) for word in inp_tuple)

    @staticmethod
    def create_lemmas_dict(text: str) -> OrderedDict:
        """Создает словарь, где ключами являются слова, а значениями - их леммы."""
        words = text.split()
        lemmas_dict = collections.OrderedDict()
        for word in words:
            lemmas_dict[word] = Lemma.lemmatize_word(word)

        return lemmas_dict


def extract_wrong_right_events(text: str):
    """Извлекает пару неправильного и правильного событий из текста."""
    pattern = r'"([\w\s]+)"'
    match = re.findall(pattern, text)
    if len(match) != 2:
        return None, None

    return match


async def refresh_db():
    """Обновляет базу данных, удаляя события, которые уже прошли."""
    user_events = small_db.get_events()
    events_to_delete = []
    for event_name, dt in user_events.items():
        if isinstance(dt, datetime) and dt < datetime.now():
            events_to_delete.append(event_name)

    for event_name in events_to_delete:
        small_db.delete_event(event_name)



class BotStates(StatesGroup):
    start_state = State()
    answering_questions_state = State()


@dp.message_handler(commands=["start"])
async def start_handler(message: types.Message):
    global syn_sets, small_db, similarity, date_time_pattern

    user_id = message.from_user.id
    user_name = message.from_user.first_name
    user_full_name = message.from_user.full_name
    logging.info(f'{user_id} {user_full_name} {time.asctime()}')

    # Инициализация пользоваткльских и функциональных классов
    syn_sets = SynonymsSets()
    small_db = SmallDB(user_id=user_id)
    similarity = Similarity(metric='tf-idf')
    date_time_pattern = DateTimePattern()

    await message.reply(f"Привет, {user_name}!")
    await bot.send_message(message.from_user.id, 'Перед использованием прочтите документацию')
    await BotStates.answering_questions_state.set()


@dp.message_handler(state=BotStates.answering_questions_state)
async def answering_questions(message: types.Message):
    try:
        global syn_sets, small_db, similarity, date_time_pattern

        text = message.text.lower()
        user_id = message.from_user.id

        # Выделение даты и времени из текста
        dt, extracted_pattern = date_time_pattern.get_date_time(text=text)

        # Удаление из текста распознанного шаблона даты и времени
        text = text.replace(extracted_pattern or '', '')

        # Создание словаря лемм
        lemmas_dict = Lemma.create_lemmas_dict(text)

        # Поиск наиболее подходящей команды на основе синонимов команд
        synonym_set_name, inters = similarity.find_synonym_set(
            ' '.join(lemmas_dict.values()),
            syn_sets
        )

        # Определение позиции команды в тексте
        min_pos, max_pos = len(text) + 1, 0
        for item in inters:
            item_pos = list(lemmas_dict.values()).index(item)
            min_pos = min(min_pos, item_pos)
            max_pos = max(max_pos, item_pos)

        # Выделение команды из текста и удаление ее из исходного текста
        cmd = ' '.join(text.split()[min_pos:max_pos + 1])
        text_without_dt = text
        text = text.replace(cmd, '').strip()

        # Обработка ошибок
        if synonym_set_name.startswith('errors'):
            # Извлекаем из текста ошибочное и правильное события
            wrong_event, right_event = extract_wrong_right_events(text)

            # Получаем последнее событие из "короткой памяти"
            short_memory = small_db.get_last_event()

            # Если не удалось извлечь какое-то из трех значений, отправляем сообщение об ошибке пользователю
            if not all((wrong_event, right_event, short_memory)):
                await bot.send_message(
                    user_id,
                    'Не смог найти шаблон "<неверное имя события>" ... "<верное имя события>"'
                )
                return

            # Создаем команду, заменяя ошибочное событие на правильное
            old_cmd_text, old_text = ' '.join(short_memory[:2]), short_memory[2]
            right_cmd = old_text.replace(right_event, '').strip()
            # Лемматизируем старую команду
            old_cmd = Lemma.lemmatize_sentence(short_memory[0])

            # Удаляем старую команду из набора синонимов и добавляем новую
            syn_sets.delete_syn_from_add_set(old_cmd)
            syn_sets.write_syn_to_add_set(Lemma.lemmatize_sentence(right_cmd))

            # Получаем все события пользователя
            user_events = small_db.get_events()

            # Если последнее событие присутствует в списке событий пользователя, заменяем его
            if short_memory[1] in user_events:
                event_dt = user_events[short_memory[1]]
                small_db.delete_event(short_memory[1])
                small_db.write_event(right_event, event_dt)
                await bot.send_message(user_id, f'Событие "{short_memory[1]}" успешно заменено на "{right_event}"')

        # Добавление событий
        elif synonym_set_name.startswith('add'):
            # Если время не было распознано, отправляем сообщение об ошибке пользователю
            if dt is None:
                await bot.send_message(user_id, 'Ошибка добавления события, время не распознано')
                return

            # Добавляем событие в базу данных
            small_db.write_event(text, dt)

            # Добавляем команду в набор синонимов после ее лемматизации
            syn_sets.write_syn_to_add_set(Lemma.lemmatize_sentence(cmd))

            # Записываем последнее событие в "короткую память"
            small_db.write_last_event((cmd, text, text_without_dt))

            # Отправляем сообщение пользователю о том, что событие было успешно добавлено
            await bot.send_message(user_id, f'Событие "{text}" добавлено на {dt}')

        # Просмотр событий
        elif synonym_set_name.startswith('display'):
            user_events = small_db.get_events()
            if user_events:
                # Если есть события в базе данных пользователя
                msg = ''
                for event, dt in user_events.items():
                    # Формирование сообщения со списком событий и их временем
                    msg = msg + f'*Событие*: {event}\n*Время*: {dt}\n\n'
                await bot.send_message(user_id, msg, parse_mode='markdown')
            else:
                # Если нет событий в базе данных пользователя
                await bot.send_message(user_id, 'События не найдены')

        # Удаление событий
        elif synonym_set_name.startswith('delete'):
            if small_db.delete_event(text):
                # Если удаление события из базы данных прошло успешно
                await bot.send_message(user_id, f'Событие "{text}" успешно удалено')
            else:
                # Если событие не найдено в базе данных
                await bot.send_message(user_id, f'Событие "{text}" не найдено')

        # Обновление базы данных
        await refresh_db()
    except Exception as err:
        await bot.send_message(
            user_id,
            f'Возникла ошибка. Обратитесь к разработчики или переформулируйте запрос. "{err}".'
        )


if __name__ == "__main__":
    executor.start_polling(dp)
