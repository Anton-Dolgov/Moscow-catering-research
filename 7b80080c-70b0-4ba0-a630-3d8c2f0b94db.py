#!/usr/bin/env python
# coding: utf-8

# Описание проекта
# 
# Инвесторы из фонда «Shut Up and Take My Money» решили попробовать себя в новой области и открыть заведение общественного питания в Москве. Заказчики ещё не знают, что это будет за место: кафе, ресторан, пиццерия, паб или бар, — и какими будут расположение, меню и цены.
# 
# Для начала они просят вас — аналитика — подготовить исследование рынка Москвы, найти интересные особенности и презентовать полученные результаты, которые в будущем помогут в выборе подходящего инвесторам места.
# Постарайтесь сделать презентацию информативной и лаконичной. Её структура и оформление сильно влияют на восприятие информации читателями вашего исследования. Выбирать инструменты (matplotlib, seaborn и другие) и типы визуализаций вы можете самостоятельно.
# 
# Вам доступен датасет с заведениями общественного питания Москвы, составленный на основе данных сервисов Яндекс Карты и Яндекс Бизнес на лето 2022 года. Информация, размещённая в сервисе Яндекс Бизнес, могла быть добавлена пользователями или найдена в общедоступных источниках. Она носит исключительно справочный характер.

# Открою файл с данными и изучу общую информацию

# In[1]:


# импорт библиотек

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import scipy.stats as stats
import warnings
import re
import plotly.express as px
import plotly.graph_objects as go
from folium import Map, Marker, Figure, Choropleth
from folium.plugins import MarkerCluster


# In[2]:


# снимаем ограничение на количество столбцов
pd.set_option('display.max_columns', None)

# снимаем ограничение на ширину столбцов
pd.set_option('display.max_colwidth', None)


# In[3]:


# подключаем модуль для работы с JSON-форматом
import json

# читаем файл и сохраняем в переменной
with open('/datasets/admin_level_geomap.geojson', encoding='utf-8') as f:
    geo_json = json.load(f)

print(json.dumps(geo_json, indent=2, ensure_ascii=False, sort_keys=True))


# In[4]:


df = pd.read_csv('/datasets/moscow_places.csv')


# In[5]:


display(df)
df.info()


# In[6]:


# для удобства заменим обозначения сетевых заведений
df['chain'] = df['chain'].map({0:'несетевое',1:'сетевое'})


# Вывод
# 
# всего в данных хранится информация о 8406 заведениях;
# типы данных у всех столбцов верные.

# Шаг 2. Выполните предобработку данных
# Изучу, есть ли дубликаты и пропуски в данных

# In[7]:


for col in ['name', 'category', 'address', 'hours', 'price', 'avg_bill']:
    df[col] = df[col].str.lower()


# In[8]:


print(f'Дубликатов: {df.duplicated().sum()}')


# In[9]:


df[df.duplicated(subset = ["name","address"])]


# 
# print ('Дубликтов по названию и адресу:',df.duplicated(subset=['name', 'address']).sum())

# In[10]:


df.loc[df.duplicated(subset=['name', 'address'], keep = False)]


# Действительно это одни и те же заведения питания. Удалим дубликаты оставив те, где информации больше либо зачения больше солответсвуют действительности.
# 
# Кафе на Ангаровских прудах которые показались как дубликаты - разные. 
# При проверке по координатам выяснилось, что название кафе с индексом 189 назвается "get&fly" второе же предприятие питания, действительно назвается "кафе"
# 
# Ресторан more poke работает с 9 до 21 ежедневно Источник: https://o-eda-dostavka.ru/goroda/moscow/rest/more_poke_wfuoa/
# 
# Бар Раковарня работает Понедельник – Четверг, Воскресенье – с 12:00 до 00:00 Пятница – Суббота – с 12:00 до 01:00 Источник: https://alekseevskaya.kleshnihvosti.ru/contacts/
# 
# Пекарня Хлеб да выпечка работает с 9 до 22. Источник: https://yandex.ru/maps/org/khleb_da_vypechka/57676860798/?indoorLevel=-1&ll=37.411004%2C55.738591&z=17
# Таких дубликатов немного. Уберем лишнее по индексам

# In[11]:


df.loc[189, 'name'] = df.loc[189, 'name'].replace('кафе','get&fly')


# In[12]:


df.query('name == "get&fly"')


# In[13]:


df = df.drop(index= [1511, 2211 ,3109])
df.reset_index(drop=True)


# In[14]:


df.isnull().sum()


# In[15]:


df.plot.scatter(x='lng', y='lat', figsize=(10, 10))
plt.show()


# Дубликатов нет.
# Пропуски есть, и в некоторых столбцах их очень много, но на данный момент заменить их не получится.
# Координаты образуют собой карту Москвы, без аномалий.

# Шаг 2.1.Создам столбец street с названиями улиц из столбца с адресом.

# In[16]:


def get_street(address):
    address = address.strip().lower()
    address = re.sub(r'\s*,\s*', ',', address)
    address = re.split(',|\.', address)
    return address[1]


# In[17]:


df['street'] = df['address'].apply(get_street)


# Шаг 2.2. Создам столбец is_24/7 с обозначением, что заведение работает ежедневно и круглосуточно (24/7):

# In[18]:


def is_24_7(time):
    if time != time:
        return np.nan
    elif time == 'ежедневно, круглосуточно':
        return True
    else:
        return False


# In[19]:


df['is_24_7'] = df['hours'].apply(is_24_7)
df.head()


# Вывод
# 
# Дубликатов нет.
# Восстановить пропуски на данный момент нет возможности.
# Были добавлены столбцы с названиями улиц и с маркером работы 24/7.

# Шаг 3. Анализ данных

# Шаг 3.1 Какие категории заведений представлены в данных? Исследую количество объектов общественного питания по категориям: рестораны, кофейни, пиццерии, бары и так далее. Построю визуализации.

# In[20]:


category = df.groupby('category').agg({'name' : 'count'}).sort_values('name', ascending = False).reset_index()
category


# In[21]:


fig = px.bar(category,
             x='category',
             y='name',
             text='name',
             labels={'name':'Количество заведений', 'category':'Формат заведения'},
             title='Распределение заведений по категориям')
fig.show()


# In[22]:


ax = px.pie(category,
             values = 'name',
             names ='category',
             title='Распределение заведений по категориям')
ax.update_layout(legend_title='Формат заведения')
ax.show()


# Вывод
# 
# 28.3% - кафе
# 24.3% - ресторан
# 16.8% - кофейня
# 7%-9% - бар/паб, пиццерия, быстрое питание
# 3%-4% - столовая, булочная
# Больше всего заведений в категориях кафе, ресторан и кофейня.

# Шаг 3.2 Исследую количество посадочных мест в местах по категориям: рестораны, кофейни, пиццерии, бары и так далее. Построю визуализации. 

# In[23]:


seats_analysis = df.groupby('category')['seats'].describe().sort_values('50%', ascending=False).reset_index()
seats_analysis


# In[24]:


df.loc[df['seats'] == 1288.0, ['name', 'category', 'seats']]


# In[25]:


fig = px.box(df.query('seats < 500.0'),
             x="category",
             y="seats",
             points="all",
             color="category",
             labels={'seats':'Количество посадочных мест', 'category':'Формат заведения'}
            )
fig.update_layout(title = 'Количество посадочных мест по категориям',
                  yaxis_range=[0,390])
fig.update_xaxes(categoryorder='array', categoryarray= seats_analysis['category'])

fig.show()


# In[26]:


plt.figure(figsize=(15,8))
ax = sns.barplot(x='category', y='50%', data=seats_analysis)
ax.set_xlabel('Формат заведения')
ax.set_ylabel('Количество посадочных мест')
ax.set_title('Количество посадочных мест по категориям')
plt.xticks(rotation=45)
plt.show()


# Вывод
# 
# В подавляющем большинстве заведений количество посадочных мест не превышает 100.
# В ресторанах и барах медианное количество посадочных мест около 80.

# Шаг 3.3. Рассмотрю и изобразижу соотношение сетевых и несетевых заведений в датасете. 

# In[27]:


fig = px.pie(df.groupby('chain').agg(count = ('chain', 'count')).reset_index(),
             values = 'count',
             names = 'chain',
             title='Соотношение сетевых и несетевых заведений на рынке Москвы'
            )
fig.update_layout(legend_title='Тип заведений')
fig.show()


# Вывод
# 
# В Москве несетевых заведений больше (61.9%) чем сетевых (38.1%)

# Шаг 3.4. Какие категории заведений чаще являются сетевыми? 

# In[28]:


chain_merge = category.merge((df
                              .query('chain == "сетевое"')
                              .groupby('category', as_index=False)
                              .agg(count=('name', 'count'))
                             ), on='category')
chain_merge['percent'] = round((chain_merge['count'] / chain_merge['name'])*100, 0).astype('int')
chain_merge.sort_values('percent', ascending=False)


# In[29]:


fig = px.bar(chain_merge.sort_values('percent', ascending=False),
             x='category',
             y='percent',
             text='percent',
             labels={'percent':'Процент сетевых заведений',
                     'category':'Формат заведения'
                    },
             title='Процент сетевых заведений от общего количества заведений'
            )
fig.update_layout(yaxis = {"categoryorder":"total ascending"})
fig.show()


# Вывод
# 
# Чаще всего сетевыми заведениями являются:
# 
# Булочная
# Пиццерия
# Кофейня

# Шаг 4.5. Сгруппирую данные по названиям заведений и найдите топ-15 популярных сетей в Москве. Под популярностью понимается количество заведений этой сети в регионе.

# In[30]:


top_15 = (df
          .query('chain == "сетевое"')
          .groupby('name', as_index=False)
          .agg(count=('chain', 'count'))
          .sort_values('count', ascending=False)
          .head(15)
         )
top_15


# In[31]:


df_chain_cat = (df[df['name'].isin(top_15['name'])]
                .groupby('category', as_index=False)
                .agg(count=('category', 'count'))
                .sort_values('count', ascending=False)
               )


# In[32]:


fig = px.bar(df_chain_cat,
             x='category',
             y='count',
             text='count',
             labels={'count':'Количество заведений', 'category':'Формат заведения'},
             title='Топ 15 заведение Москвы, сгруппированных по их формату'
            )
fig.show()


# Вывод
# 
# Самая крупная сеть в Москве - "Шоколадница".
# В категории "кофейня" самое большое количетсво сетей.

# Шаг 4.6. Какие административные районы Москвы присутствуют в датасете? 

# In[33]:


administration = df.groupby(['district', 'category'], as_index=False).agg(count = ('name', 'count')).sort_values(['district', 'count'], ascending=False)
administration.head(20)


# In[34]:


fig = px.bar(administration,
             x='count',
             y='district',
             color='category',
             orientation='h',
             text='count',
             labels={'count':'Количество заведений', 'category':'Формат заведения', 'district':'Округ Москвы'},
             title='Распределение формата заведений по округам'
            )
fig.update_layout(height=600, yaxis = {"categoryorder":"total ascending"})
fig.show()


# Вывод
# 
# Самое большое количество заведений в Центральном административном округе, рестораны доминируют.
# Самое маленькое количество заведений в Северо-Западном административном округе.

# Шаг 4.7. Визуализируйю распределение средних рейтингов по категориям заведений. 

# In[35]:


rating_mean_category = (df
                        .groupby(['district', 'category'], as_index=False)
                        .agg(mean_rating = ('rating', 'mean'))
                        .sort_values('mean_rating', ascending=False)
                       )
rating_mean_category


# In[36]:


fig = px.histogram(rating_mean_category,
                   x='district',
                   y='mean_rating',
                   color = 'category',
                   barmode='group',
                   labels={'district':'Округ Москвы', 'mean_rating':'Средний рейтинг заведения', 'category':'Формат заведения'},
                   title='Распределение средних рейтингов по категориям заведений'
                  )
fig.update_layout(
    yaxis_range=[3.9, 4.5],
    height=500)
fig.show()


# Вывод
# 
# Самые высокие рейтинги заведений в Центральном административном окргуе. Большинство заведений в ЦАО вероятнее всего сильно выше по уровню сервиса, в сравнении с заведениями в соседних регионах.

# Шаг 4.8. Построю фоновую картограмму (хороплет) со средним рейтингом заведений каждого района. 

# In[37]:


rating_mean_district = df.groupby('district', as_index=False).agg(mean_rating = ('rating', 'mean'))
rating_mean_district


# In[38]:


moscow_lat, moscow_lng = 55.751244, 37.618423
m = Map(location=[moscow_lat, moscow_lng], zoom_start=10)
Choropleth(
    geo_data=geo_json,
    data=rating_mean_district,
    columns=['district', 'mean_rating'],
    key_on='feature.name',
    fill_color='YlGn',
    fill_opacity=0.8,
    legend_name='Средний рейтинг заведений по районам'
).add_to(m)
m


# In[39]:


# создаём карту Москвы
j = Map(location=[moscow_lat, moscow_lng], zoom_start=10)
# создаём пустой кластер, добавляем его на карту
marker_cluster = MarkerCluster().add_to(j)
# пишем функцию, которая принимает строку датафрейма,
# создаёт маркер в текущей точке и добавляет его в кластер marker_cluster
def create_clusters(row):
    Marker(
        [row['lat'], row['lng']],
        popup=f"{row['name']} {row['rating']}",
    ).add_to(marker_cluster)

# применяем функцию create_clusters() к каждой строке датафрейма
df.apply(create_clusters, axis=1)

# выводим карту
j


# Шаг 4.9.Найду топ-15 улиц по количеству заведений. 

# In[40]:


street_count = (df.groupby('street')
                .agg(count=('name', 'count'))
                .sort_values('count', ascending=False)
               )
street_count.head(15)


# In[41]:


street_cat = df[df['street'].isin(street_count.head(15).index.tolist())].groupby(['street', 'category'], as_index=False).agg(
    count=('name', 'count')).sort_values('count', ascending=False)


# In[42]:


ax = px.bar(street_cat, 
            x='count', 
            y='street', 
            color='category', 
            orientation='h', 
            text='count', 
            labels={'street':'Название улицы', 'count':'Количество заведений', 'category':'Формат заведения'},   
            title='ТОП-15 улиц с самым большим количеством заведений')
ax.update_layout(height=600, yaxis = {"categoryorder":"total ascending"})
ax.show()


# Вывод
# 
# Самое большое количество заведений на улице Проспект мира.
# Самые популярные категории заведений: кафе и ресторан.

# Шаг 4.10. Найду улицы, на которых находится только один объект общепита. Что можно сказать об этих заведениях?

# In[43]:


street_count_solo = (df[df['street'].isin(street_count.query('count == 1').index.tolist())]
                     .groupby('category', as_index=False)
                     .agg(count=('name', 'count'))
                     .sort_values('count', ascending=False))
street_count_solo


# In[44]:


fig = px.bar(street_count_solo, 
             x='category', 
             y='count',  
             text='count',
             labels={'count':'Количество улиц', 'category':'Формат заведения', 'category':'Формат заведения'},
             title='Количество улиц с одним заведением')
fig.show()


# Вывод
# По графику видно:
# 
# На не популярных улицах больше всего кафе, далее идут рестораны и кофейни.
# Можно предположить что связано это с тем, что такие заведения как столовая, бар, пиццерия и быстрое питание, открывают там где большой поток клиентов. А кафе, ресторан и кофейни, могут открывать в местах где нет конкурентов и без огромной проходимости.

# Шаг 4.11. Значения средних чеков заведений хранятся в столбце middle_avg_bill. Эти числа показывают примерную стоимость заказа в рублях, которая чаще всего выражена диапазоном. Посчитаю медиану этого столбца для каждого района. Используйю это значение в качестве ценового индикатора района. Построю фоновую картограмму (хороплет) с полученными значениями для каждого района. Проанализирую цены в центральном административном округе и других.

# In[45]:


bill_median_district = (df
                        .groupby('district', as_index=False)
                        .agg(median_bill = ('middle_avg_bill', 'median'))
                        .sort_values('median_bill', ascending=False)
                       )
bill_median_district


# In[46]:


moscow_lat, moscow_lng = 55.751244, 37.618423
n = Map(location=[moscow_lat, moscow_lng], zoom_start=10)
Choropleth(
    geo_data=geo_json,
    data=bill_median_district,
    columns=['district', 'median_bill'],
    key_on='feature.name',
    fill_color='YlGn',
    fill_opacity=0.8,
    legend_name='Медиана средних чеков по районам'
).add_to(n)
n


# Вывод
# 
# ожидаемо самый дорогой средний чек в ЦАО и ЗАО (самые дорогие районы Москвы)
# самые дешевый средний чек в СВАО и ЮВАО
# остальные районы не сильно отличаются друг от друга

# Общий вывод
# больше всего заведений в категориях кафе, ресторан и кофейня;
# в подавляющем большинстве заведений количество посадочных мест не превышает 100;
# 40% всех заведений являются сетевыми ( у булочных 60% и кофейня, пиццерия 50%);
# чаще всего сетевыми заведениями являются: кафе, рестораны и кофейни;
# больше всего заведений в ЦАО, так же в нем в отличие от других районов, доминируют рестораны;
# самый высокий рейтинг по всем районам у баров, самый низкий рейтинг почти по всем районам у быстрого питания и кафе;
# больше всего заведений на пр. Мира, ул. Профсоюзной. пр. Вернадского и Ленинский пр. (обусловлено это тем что эти улицы очень длинные);
# на не популярных улицах (1 заведение на всю улицу) чаще всего расположены кафе, рестораны и кофейни;
# самый дорогой средний чек в ЦАО и ЗАО (самые дорогие районы Москвы), а самый дешевый средний чек в СВАО и ЮВАО.

# Шаг 4. Детализируем исследование: открытие кофейни

# Шаг 4.1. Сколько всего кофеен в датасете? В каких районах их больше всего, каковы особенности их расположения?

# In[47]:


df_cafe = df.query('category == "кофейня"')
print(f'Всего в Москве {df_cafe["name"].nunique()} кофеен')


# In[48]:


district_info_cafe = (df_cafe
                      .groupby('district', as_index=False)
                      .agg(count=('name', 'count'), 
                           cup_avg=('middle_coffee_cup', 'mean'), 
                           mean_rating = ('rating', 'mean'),
                           median_seats = ('seats', 'median')
                          )
                      .sort_values('count', ascending=False))
district_info_cafe


# In[49]:


moscow_lat, moscow_lng = 55.751244, 37.618423
v = Map(location=[moscow_lat, moscow_lng], zoom_start=10)
Choropleth(
    geo_data=geo_json,
    data=district_info_cafe,
    columns=['district', 'count'],
    key_on='feature.name',
    fill_color='BuPu',
    fill_opacity=0.8,
    legend_name='Количество кофеен в каждом районе'
).add_to(v)
v


# Исключим ЦАО из нашего отображения, так в данном регионе самое большое количество кофеен. Это позволит нам лучше детализировать регионы вокруг ЦАО.

# In[50]:


moscow_lat, moscow_lng = 55.751244, 37.618423
v = Map(location=[moscow_lat, moscow_lng], zoom_start=10)
Choropleth(
    geo_data=geo_json,
    data=district_info_cafe.tail(-1),
    columns=['district', 'count'],
    key_on='feature.name',
    fill_color='BuPu',
    fill_opacity=0.8,
    legend_name='Количество кофеен в каждом районе без ЦАО'
).add_to(v)
v


# In[51]:


cafe = px.bar(district_info_cafe,
              x='district',
              y='count',
              text='count',
              labels={'count':'Количество кофеен', 'district':'Район'},
              title='Количество кофеен по районам')
cafe.show()


# In[52]:


moscow_lat, moscow_lng = 55.751244, 37.618423
b = Map(location=[moscow_lat, moscow_lng], zoom_start=10)
Choropleth(
    geo_data=geo_json,
    data=district_info_cafe,
    columns=['district', 'cup_avg'],
    key_on='feature.name',
    fill_color='YlOrBr',
    fill_opacity=0.8,
    legend_name='Средняя стоимость чашки капучино по районам'
).add_to(b)

df_cafe_top3 = df_cafe.query('name != "Шоколадница"').sort_values('middle_coffee_cup', ascending = False).head(3)

# создаём пустой кластер, добавляем его на карту
marker_cluster = MarkerCluster().add_to(b)
# пишем функцию, которая принимает строку датафрейма,
# создаёт маркер в текущей точке и добавляет его в кластер marker_cluster
def create_clusters(row):
    Marker(
        [row['lat'], row['lng']],
        popup=f"{row['name']} {row['middle_coffee_cup']}",
    ).add_to(marker_cluster)

# применяем функцию create_clusters() к каждой строке датафрейма
df_cafe_top3.apply(create_clusters, axis=1)

# выводим карту
b


# Точками на хороплете отображены топ-3 кофейни с самой дорогой чашкой капучино в Москве. Две из них ожидаемо находятся в ЦАО.

# Шаг 4.2. Определю есть ли круглосуточные кофейни

# In[53]:


df_cafe_24_7 = df_cafe.query('is_24_7 == True')


# In[54]:


cafe = px.bar((df_cafe_24_7
                      .groupby('district', as_index=False)
                      .agg(count=('name', 'count'))
                      .sort_values('count', ascending=False)),
              x='district',
              y='count',
              text='count',
              labels={'count':'Количество круглосуточных кофеен', 'district':'Район'},
              title = 'Количество круглосуточных кофеен по районам'
             )
cafe.show()


# Ожидаемо, в ЦАО круглосуточных кофеен больше всего. В целом, распределение бьется с общим количеством заведений по регионам.

# Шаг 4.3. Какие у кофеен рейтинги? Как они распределяются по районам?

# In[55]:


moscow_lat, moscow_lng = 55.751244, 37.618423
g = Map(location=[moscow_lat, moscow_lng], zoom_start=10)
Choropleth(
    geo_data=geo_json,
    data=district_info_cafe,
    columns=['district', 'mean_rating'],
    key_on='feature.name',
    fill_color='Blues',
    fill_opacity=0.8,
    legend_name='Распределение средних рейтингов кофеен по районам',
    ).add_to(g)
g


# На хороплете видно, что самый высокий рейтинг в регионах ЦАО, СЗАО.

# Шаг 4.4. На какую стоимость чашки капучино стоит ориентироваться при открытии и почему?

# In[56]:


order = df_cafe.groupby('district')['middle_coffee_cup'].median().sort_values(ascending=False).index
ax = sns.boxplot(data=df_cafe, y='district', x='middle_coffee_cup', order=order)
ax.set_xlabel("Цена чашки капучино")
ax.set_ylabel("Район")
ax.set_title("Средняя стоимость чашки капучино по районам")
plt.show()


# Дешевый сегмент кафе находится в регионах: ЮАО, ЮВАО, ВАО.
# Средний сегмент кафе находится в регионах: САО, СВАО, СЗАО.
# Дорогой сегмент кафе находится в регионах: ЦАО, ЗАО, ЮЗАО.

# In[57]:


coffee_house_avg = df_cafe.query('middle_coffee_cup.notnull()')


# In[58]:


coffee_house_avg.groupby('district').agg({'middle_coffee_cup': 'describe'})


# Вывод
# ЦАО доминирует по всем фронтам, конкуренция огромная, требования к заведениям очень высокие. Если мы только пробуем открывать кафе, то этот регион не самый предпочтительный для теста.
# 
# Одним из перспективных районов для открытия кофейни считаю СЗАО.
# 
# Самое низкое количество кофеен в районе, конкуренция будет ниже;
# Самые высокие оценки кофеен и в целом по каждому заведению, значит жители более лояльны;
# Очень мало, а именно всего две круглосуточные кофейни;
# Цена чашки капучино как и везде, около 165 рублей;
# Жители уважают кофейни с большим количеством посадочных мест (а в данном районе в среднем 87 мест).

# Ссылка на презентацию: https://disk.yandex.ru/d/wSX1IVjyfovKhw
