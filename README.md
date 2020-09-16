# NtechLab_CV_Task

**Описание задачи** : Необходимо обучить нейросеть, способную по входному изображению лица определять пол человека на изображении.

**Датасет** выглядел следующим образом:
![data](Task2/imgs/look_at_batch.png)

Данные были разделены на две папки - **тренировочную** и **тестовую**. Тестовая часть составляла 10 % от исходных данных

В данной задаче была обучена нейросеть **ResNet18**. Обучение проводилось при помощи библиотеки **Pytorch Lightning**.

В результате были получены следующие графики лосс функций и метрик:

Тренировочная лосс функция
![train_loss_0](Task2/imgs/train_loss_0.jpg)

Тренировочная метрика
![train_acc_0](Task2/imgs/train_acc_0.jpg)

Тестовая лосс функция
![val_loss_0](Task2/imgs/val_loss_0.jpg)

Тестовая метрика
![val_acc_0](Task2/imgs/val_acc_0.jpg)

Данные графики можно посмотреть при помощи tensorboard командой %tensorboard --logdir lightning_logs (скопируйте папку lightning_logs)
