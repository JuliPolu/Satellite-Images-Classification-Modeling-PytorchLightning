# Комментарии и история экспериментов


Хочется написать дисклеймер что опыта в моделировании практически нет, поэтому пробовала идти на ощупь, используя готовые темплейты

Взяла за снову готовый код Lightening, привела данные к небходимому виду (детали в [ноутбуке](notebooks/EDA.ipynb) )

в качестве метрики взяла f1score как было рекомендовано на консультации по проекту хотя f2score показывала выше результат и она использовлась на кагле

удалось добиться скромного значения f1score 0.668 думаю вследствие сильного дисбаланса классов и использования уследнения macro чтобы все классы участвовали, если использовать weighted - то метрики подскакивают сразу почти до 0.9


### Не смогла разобраться

не поняла как работает DVC, не включла его во премя тренировки моделей, поэтому пока просто добавила туда чекпойнт последней модели

### Что зашло
 
- Базовые параметры модели из лекций в целом зашли хорошо

- threshold 0.2

- efficientnet-b0 [ClearML](https://app.clear.ml/projects/422eb34b25884733baf0e5ea20ae9b93/experiments/fd74fab66fb346b2972ea4142f205012/output/execution) 
и resnet50 [ClearML](https://app.clear.ml/projects/422eb34b25884733baf0e5ea20ae9b93/experiments/ecee3257695e43b89fa1abd05092bf8d/output/execution) 
показали схожые самые высокие результаты по f1score macro но так как b0 существенно легче то я выбрала ее как основную

- удаление из аугментаций манипуляций с контрастами и цветом
  

### Что не зашло

- Focal loss (не нашла готовую реализаци и добавила класс в [файл](src/focal_loss.py) : сильно подскакивал recall и падал precision причем чем больше гамма тем значительнее изменения [ClearML](https://app.clear.ml/projects/422eb34b25884733baf0e5ea20ae9b93/experiments/d1313a54812e4eb199f0c6a372fa21cc/output/execution) 

- Adam c различными lr показал результаты хуже

- albu.CLAHE ухудшил результаты

-  размер батч сайза не повлиял никак

- Resnet18 с доубучением всех слоев: метркии растут слабее [ClearML](https://app.clear.ml/projects/422eb34b25884733baf0e5ea20ae9b93/experiments/c3bc30b88b8949fbae7182ed8f6da093/output/execution)

- Resnext50, efficientnet-b3, efficientnet-b5 с доубучением всех слоев в целом показали выше результаты чем Resnet18 но ниже чем Resnet50 и efficientnet-b3

- Resnet18, Resnet50, efficientnet-b0, efficientnet-b3, efficientnet-b5  с заморозкой весов, и обучением только классификатора, и разморозкой последних 5 - 10 слоев (видимо потому что спутниковые снимки весьма специфическая категория изображений) - f1score просел до 0.3 - 0.4 [ClearML](https://app.clear.ml/projects/422eb34b25884733baf0e5ea20ae9b93/experiments/c3bc30b88b8949fbae7182ed8f6da093/output/execution)  


### Открытые вопросы

  
- Инферила через загрузку модел чекпойнт. Вопрос это норм или лучше отдельно модель выгружать через onnx тот же?

- какой f1score macro реально добиться и какими способами? :) чтобы я смoгла дообучить сетку на досуге

- почему не сработал focal loss и правильно ли я его реализовала?

- каким образом бороться с дисбалансом классов 

- Каким образом и нужно ли отдельно подбирать threshold для каждого класса, попробовала написать код для подбора порогового значения (сам код лежит [тут](src/lightning_module_with_thld.py) ) - но в итоге метрики особо не улучшились [ClearML](https://app.clear.ml/projects/422eb34b25884733baf0e5ea20ae9b93/experiments/e902d4174136444e98b7840399848623/output/execution)

### После ревью

- Удалила лишние файлы

- добавила в dvc папку с удачным чекпойнтом и файл финальной модели

- сделала отдельные скрипты для конвертации [тут](src/convert_checkpoint.py) и для инференса [тут](src/infer.py)  Так нормально или все же лучше в одном скрипте это делать?

- Попробовала прогнать с обновленной функцией focal loss с взвешиванием по обратной частоте и квадратной обратной частоте и различными гаммами, но улучшить метрики так и не удалось. расчет обратныx частот в [тетрадке](notebooks/EDA.ipynb)
  
  [ClearML_eff-b0_focal_gamma_2.0_vanila]  (https://app.clear.ml/projects/422eb34b25884733baf0e5ea20ae9b93/experiments/995c04ec66284444a0ab6ba335c30ac0/output/execution)
  [ClearML_eff-b0_focal_gamma_2_sqrtif] (https://app.clear.ml/projects/422eb34b25884733baf0e5ea20ae9b93/experiments/ea639b850a9e48c59ea7dde9c8a2b216/output/execution)
  [ClearML_eff-b0_focal_gamma_0.5_vanila] (https://app.clear.ml/projects/422eb34b25884733baf0e5ea20ae9b93/experiments/c3291cc86f1449aa8cb6b8a1321ba69d/output/execution)
  [ClearML_eff-b0_focal_gamma_0.5_weighted_sqrtif] (https://app.clear.ml/projects/422eb34b25884733baf0e5ea20ae9b93/experiments/5ecfe8621f964c6ba2e518cc5a7f0801/output/execution)
  [ClearML_eff-b0_focal_gamma_0.5_weighted_if] (https://app.clear.ml/projects/422eb34b25884733baf0e5ea20ae9b93/experiments/cf6df2ac5aac49a995c0e4c623a685a5/output/execution)

- также попробовала прогнать weighted BCEWithLogitsLoss но также целевую метрику это не улучшило
  [ClearML_eff-b0_w_sqrtif_bce] (https://app.clear.ml/projects/422eb34b25884733baf0e5ea20ae9b93/experiments/74300e7b28304626801279c264e925e1/output/execution)
  [ClearML_eff_b0_w_icf_bce] (https://app.clear.ml/projects/422eb34b25884733baf0e5ea20ae9b93/experiments/79493fdbc68643e0ac5643bb77f575c0/output/execution)
  
  
