# Powerball set model (Transformer)

Эта папка — адаптация `train_lottomax_set.py` под **US Powerball**:

- 5 основных чисел как **неупорядоченное множество** (multi-label по 69)
- 1 Powerball как **отдельная классификация** (single-label по 26)

> В Powerball бонусный шар берётся из отдельного набора 1–26, поэтому его **не нужно** маскировать по main-числам.

## Формат данных

CSV с заголовком:

```
date,n1,n2,n3,n4,n5,pb
```

Числа:

- `n1..n5`: 1..69 (уникальные)
- `pb`: 1..26

Порядок строк:

- можно любой — скрипт попытается распарсить `date` и отсортировать по времени (oldest → newest)

## Обучение

### Рекомендуемый вариант (transfer): pretrain на всей истории → finetune на современной эпохе

Почему так:
- до ~2015 PB был из другого диапазона (в твоём CSV встречается `pb > 26`)
- PB “ломает” единую постановку, но **main-числа** (1..69) можно использовать для pretraining

#### Шаг 1 — pretrain только по main (PB отключаем и во входе, и в loss)

```bash
python powerball_set/train_powerball_set.py \
  --data_path powerball.csv \
  --allow_legacy_pb \
  --no_pb_input \
  --pb_loss_weight 0 \
  --epochs 200 \
  --context_len 16 \
  --batch_size 16 \
  --d_model 128 --nhead 4 --num_layers 4 --dropout 0.1 \
  --ckpt checkpoints/powerball_pretrain_main.pt
```

#### Шаг 2 — finetune на post-change эпохе (PB 1..26) + предсказываем main и PB

```bash
python powerball_set/train_powerball_set.py \
  --data_path powerball.csv \
  --min_date 10/07/15 \
  --init_ckpt checkpoints/powerball_pretrain_main.pt \
  --epochs 200 \
  --context_len 16 \
  --batch_size 16 \
  --d_model 128 --nhead 4 --num_layers 4 --dropout 0.1 \
  --ckpt checkpoints/powerball_set.pt
```

### Простой вариант (без transfer): только современная эпоха

Из корня репозитория:

```bash
python powerball_set/train_powerball_set.py --data_path powerball.csv --min_date 10/07/15 --epochs 200 --context_len 16 --batch_size 16
```

Чекпоинт по умолчанию:

- `checkpoints/powerball_set.pt`

## Важно про диапазон PB (исторический переход)

В исторических данных Powerball есть период, где `pb` мог быть **больше 26**.
Современная механика (после перехода) использует `pb` в диапазоне **1..26**.

Этот код **всегда** учит/предсказывает PB как классификацию на 26 значений.
Поэтому:

- если твой `powerball.csv` содержит старые строки с `pb > 26`, используй `--min_date` (например `10/07/15`), чтобы обучаться только на “современной” эпохе.

## Предсказание

### Предсказать следующий тираж после последнего известного

```bash
python powerball_set/predict_powerball_set.py --data_path powerball.csv --min_date 10/07/15 --ckpt checkpoints/powerball_set.pt
```

### Проверить на последнем известном тираже (быстрая sanity-check)

```bash
python powerball_set/predict_powerball_set.py --data_path powerball.csv --min_date 10/07/15 --ckpt checkpoints/powerball_set.pt --eval_last_known
```


