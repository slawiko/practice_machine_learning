### kNN Irises

Соотношение тренировочной (если можно так сказать) к тестовой: 4 к 1. Все данные пошафлены от запуску к запуску. Делаю k-fold с количеством fold = 5, перебирая k от 1 до 10, и перебериая три расстояние: Манхэттена, Евклидово и Чебышева.

Результаты семи запусков k-fold:

```bash
Best accuracy 0.9916666666666668 is achieved with k = 3, and euklidian distance
Best accuracy 0.9583333333333334 is achieved with k = 3, and euklidian distance
Best accuracy 0.9583333333333333 is achieved with k = 6, and chebyshev distance
Best accuracy 0.9916666666666668 is achieved with k = 1, and chebyshev distance
Best accuracy 0.975 is achieved with k = 7, and chebyshev distance
Best accuracy 0.95 is achieved with k = 3, and chebyshev distance
Best accuracy 0.9666666666666668 is achieved with k = 6, and euklidian distance
```

Результаты настолько разнятся, что сделать какой-то вывод довольно сложно. Разве что k в диапазоне от 3 до 8, и в основном расстояние Чебышева. 

Из-за настолько не четких данных, решил даже не пробовать подобранные параметры на тестовых данных, а вовсе не делить на тестовые и нетестовые. Вместо этого просто сделать 100 k-fold-ов на всех данных и выбрать наиболее часто встречающиеся результаты и их среднее accuracy.

Результат четырех запусков:

```bash
Best average accuracy 0.9816666666666668 is achieved with 8 chebyshev
Second best average accuracy 0.98095238095238102 is achieved with 9 euklidian

Best average accuracy 0.9866666666666667 is achieved with 8 euklidian
Second best average accuracy 0.9847619047619046 is achieved with 9 chebyshev

Best average accuracy 0.9866666666666667 is achieved with 10 euklidian
Second best average accuracy 0.9820512820512821 is achieved with 8 chebyshev

Best average accuracy 0.9866666666666667 is achieved with 7 euklidian
Second best average accuracy 0.9850000000000001 is achieved with 9 chebyshev
```

Можно сделать вывод, что лучшие параметры k от 8 до 9, и выбирать стоит расстояние чебышева либо евклидово.
