## Инструкция к запуску

(команды применимы к терминалу или его аналогам)

0. Убедитесь, что у вас установлен интерпретатор python (данный код протестирвоан на версии 3.10)
1. Убедитесь, что у вас установлен пакетный менеджер pip (протестирован на версии 22.3.1)
2. Склониройте проект комендой ```git clone```
3. Перейдите в корневую директорию склонированного проекта (в терминале ```cd UrFU-MLOps-Labs```)
4. Установите необходимые зависимости ```python -m pip install -r requirements.txt```
5. Добавьте прав к исполняемому скрипту ```chmod u+x ./job.sh```
6. Если в хотите удалить все создаваемые в процессе работы файлы, то закоментируйте последнюю строчку исполняемого файла *job.sh*.
7. Запустите исполняемый скрипт ```./job.sh```
8. "Наслаждайтесь" результатом абсолютной ненастроенной модели линейной регрессии.