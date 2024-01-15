## Основная задача

Создание и обучение модели мультилейбл классификации космических снимков бассейна Амазонки
Данный репозиторий подготовлен в рамках 1-й проектной работы курса 'CV-Rocket' от команды Deepschool, часть по созданию и обучению модели мультилейбл классификации.


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

[ClearML eff b0](https://app.clear.ml/projects/422eb34b25884733baf0e5ea20ae9b93/experiments/f892ff037efd4ff3999341aa0a267baf/output/execution)


### Актуальная версия чекпойнта модели:

dvc pull models/checkpoint.dvc

### Актуальная версия сохраненной torscript модели:

dvc pull models/jit_model/final_model.pt.dvc


### Инеренс

Посмотреть результаты работы обученной сети можно посмотреть в [тетрадке](notebooks/inference.ipynb)


А также запустить скрипт для инференса
```
PYTHONPATH=. python src/convert_checkpoint.py --checkpoint ./models/checkpoint/epoch_epoch=07-val_f1=0.651.ckpt
```

И запустить скрипт для инференса
```
PYTHONPATH=.  python ./src/infer.py --model_path ./models/jit_model/final_model.pt --image_path ./data/Images/train_26419.jpg
```
