### 1

Для начала подготовил данные в [`prepare_data.py`](./prepare_data.py): привел в формат, который принимает libsvm (`<label> <index1>:<value1> ...`) и разбил на тренировочную и тестовую в нужном соотношении (3450:1151).

Далее промасштабировал при помощи `svm-scale` с параметрами `-l 0 -u 1`. Тренировка после этого показала странные результаты: на всех размерностях, кроме `d = 1` SVM классифицировал все объекты нулями. С дефолтными параметрами (`-l -1 -u 1`) все классифицировалось порядочно и с хорошей accuracy (>90%). Тогда попробовал промасштабировать с `-l 0 -u 10`. Теперь svm нормально классифицировал, но график количества опорных векторов к размерности выглядел так: чем больше размерность, тем больше векторов было. Это странно, потому что при увеличении степени, выборку должно быть легче разделить, а если выборку легче разделить, то и сомнительных векторов (тех, которые лежат близко к разделящей кривой) должно быть меньше. Поэтому решил продолжить с дефолтными параметрами. 

Наилучшие параметры: `d = 4`, `C = 32`. Итоговое accuracy: `91.48566463944397`. Количество опорных векторов для этого решения: `904`

### 2

Для начала подготовил данные в [`prepare_data_gisette.py`](./prepare_data_gisette.py). Всего в датасете было 3 типа файлов: train, test, validation. Притом test не был размечен. Решил воспользоваться им как данными для масштабирования: совместил с train и промасштабировал ([`scale_gisette.sh`](./scale_gisette.sh)) от 0 до 1, сохранив параметры масштабирования. Дальше с этими параметрами промасштабировал train и validation (validation у меня вместо test, условно говоря)

Далее кроссвалидировал и перебирал параметры `s`, `e`, `C`. Наилучшие параметры: `s = `, `e = `, `C = `. Итоговое accuracy: ``