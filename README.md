## Основная задача

Мультилейбл классификация космических снимков бассейна Амазонки


### Датасет

Трейн датасет включает 40479 космических снимков и 17 лейблов.

[Ссылка](https://www.kaggle.com/datasets/nikitarom/planets-dataset) на исходный датасет

Скаченные данные лежат в [папке data](./data)

Первичный анализ и подготовка данных в папке [тетрадке](notebooks/EDA.ipynb)


### Обучение

Запуск тренировки:

```
PYTHONPATH=. python src/train.py configs/config_eff_b0_base.yaml
```

### Логи финальной модели в ClearML

Перформанс модели можно посмотреть тут:

[ClearML efficientnet b0](https://app.clear.ml/projects/422eb34b25884733baf0e5ea20ae9b93/experiments/fd74fab66fb346b2972ea4142f205012/output/execution)


### Актуальная версия чекпойнта модели:

dvc pull models/checkpoint.dvc


Я пока к сожалению не разобралась с dvc staging :(, и к тому же запускала эксперименты на vast.ai. Добавила dvc для трекинга модел чекпойнт


### Инеренс

Посмотреть результаты работы обученной сети можно посмотреть в [тетрадке](notebooks/inference.ipynb)


А также запустить скрипт для инференса

```
.PYTHONPATH  ./src/infer.py
```

### Страдания модельера

Также очень прошу можно ознакомиться со всей болью начинающей модельера, возникшими вопросами и комментариями

[HISTORY&COMMENTS.md](HISTORY&COMMENTS.md)