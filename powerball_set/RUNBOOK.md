# Powerball model — запуск обучения и предсказаний

Этот runbook описывает, как обучать и использовать модель из папки `powerball_set/`.

## 0) Где запускать команды

Все команды выполняй **из корня репозитория** (там, где лежит `powerball.csv`):

- `powerball.csv`
- `powerball_set/train_powerball_set.py`
- `powerball_set/predict_powerball_set.py`

## 1) Формат данных

Файл `powerball.csv` должен иметь заголовок:

```
date,n1,n2,n3,n4,n5,pb
```

Диапазоны:

- `n1..n5`: 1..69 (уникальные)
- `pb`: исторически бывает > 26 (старые эпохи), но современная механика — **1..26**

Скрипт сам сортирует по `date` (oldest → newest), если даты парсятся.

## 2) Установка зависимостей (один раз)

Внутри твоего окружения (например `.venv`) установи зависимости:

```
pip install -r requirements.txt
```

> Для GPU ставь PyTorch под свою CUDA/WSL2 сборку (как на официальной странице PyTorch “Get Started”).

## 3) Рекомендованный режим (transfer): все данные + PB только после даты перехода

Идея:
- на всей истории (включая старые эпохи с `pb > 26`) делаем **pretrain только по main (5-of-69)**
- затем делаем **finetune** только на эпохе, где `pb` уже **всегда 1..26**, и учим полную задачу (main + PB)

### 3.1) Pretrain (main-only) на всей истории

Команда:

```bash
python3 powerball_set/train_powerball_set.py \
  --data_path powerball.csv \
  --allow_legacy_pb \
  --no_pb_input \
  --pb_loss_weight 0 \
  --epochs 80 \
  --context_len 32 \
  --batch_size 256 \
  --d_model 256 --nhead 8 --num_layers 6 --dropout 0.1 \
  --ckpt checkpoints/powerball_pretrain_main.pt
```

Результат:
- чекпоинт: `checkpoints/powerball_pretrain_main.pt`

### 3.2) Finetune на современной эпохе (PB 1..26) с определённой даты

Дата по умолчанию (как обсуждали): **`10/07/15`**.

Команда:

```bash
python3 powerball_set/train_powerball_set.py \
  --data_path powerball.csv \
  --min_date 10/07/15 \
  --init_ckpt checkpoints/powerball_pretrain_main.pt \
  --epochs 150 \
  --context_len 32 \
  --batch_size 256 \
  --d_model 256 --nhead 8 --num_layers 6 --dropout 0.1 \
  --ckpt checkpoints/powerball_set.pt
```

Результат:
- финальный чекпоинт: `checkpoints/powerball_set.pt`

## 4) Простой режим: только современная эпоха (PB<=26), без pretrain

Если хочешь строго “без старых данных”:

```bash
python3 powerball_set/train_powerball_set.py \
  --data_path powerball.csv \
  --min_date 10/07/15 \
  --epochs 150 \
  --context_len 32 \
  --batch_size 256 \
  --d_model 256 --nhead 8 --num_layers 6 --dropout 0.1 \
  --ckpt checkpoints/powerball_set_postchange_only.pt
```

## 5) Предсказание (inference)

> Рекомендуемый способ запуска предикта — **через модуль** `-m`.

### 5.1) Предсказать следующий тираж после последнего в CSV

Для модели, обученной на post-change эпохе:

```bash
python3 -m powerball_set.predict_powerball_set \
  --data_path powerball.csv \
  --min_date 10/07/15 \
  --ckpt checkpoints/powerball_set.pt
```

Вывод:
- `Predicted MAIN (top-5)` (1..69)
- `Predicted PB (top-1)` (1..26)
- `PB candidates` (top-N кандидатов PB)

### 5.2) Быстрая проверка на последнем известном (sanity-check)

```bash
python3 -m powerball_set.predict_powerball_set \
  --data_path powerball.csv \
  --min_date 10/07/15 \
  --ckpt checkpoints/powerball_set.pt \
  --eval_last_known
```

## 6) Эксперимент: есть ли перенос между эпохами PRE↔POST?

Этот эксперимент оценивает transfer **только по main-числам**, чтобы смена диапазона PB не мешала:

```bash
python3 -m powerball_set.transfer_experiment \
  --data_path powerball.csv \
  --cutover_date 10/07/15 \
  --context_len 16 \
  --epochs 30 \
  --batch_size 64
```


