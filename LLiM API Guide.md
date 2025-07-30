# LLiM API Guide

GPU cluster hosting three downstream-task-optimized 1 B-parameter LLiM models.  
All endpoints are reachable after security authentication.

---

## 1. Service Overview

| Task No. | Task Name                | Endpoint                         |
|----------|--------------------------|----------------------------------|
| 1        | Capacity Estimation      | http://39.170.99.155:1111        |
| 3        | Remain Range Prediction  | http://39.170.99.155:3333        |
| 4        | Anomaly Detection        | http://39.170.99.155:3344        |

---

## 2. Security Protocol

For server security:

1. Send your organization’s **fixed public IP address** to **lzjoey@gmail.com**.
2. We’ll configure firewall rules to grant access.

---

## 3. API Specifications

### 3.1 Request Format

```python
{
  "inputs":  data[i][None, :, :].tolist(),
  "labels":  labels[i].tolist()
}
```


### 3.2 Access Example
You can directly use the following PY code to modify the local storage file address and task_num:.

```python
response = requests.post(server_b_url, json=sample)
prediction = response.json()

import time
import pickle
import requests
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def api_test(task_num: int):
    """
    task_num == 3:
        data: len(1162) sample shape: [450, 71]
        labels: len(1162)
    """

    # ------------- file & server selection -------------
    if task_num == 3:
        file_root = ''  # TODO: local file path
        server_b_url = 'http://39.170.99.155:3333/predict'
    elif task_num == 4:
        file_root = ''  # TODO: local file path
        server_b_url = 'http://39.170.99.155:3344/predict'
    elif task_num == 1:
        file_root = ''  # TODO: local file path
        server_b_url = 'http://39.170.99.155:1111/predict'
    else:
        print('Exceptional, there is no serial number for this task!')
        return

    with open(file_root, 'rb') as f:
        dataset = pickle.load(f)

    data   = dataset['data']
    labels = dataset['labels']

    # ------------- inference -------------
    preds, trues = [], []
    start = time.time()

    for i in range(len(data)):
        sample = {
            "inputs": data[i][None, :, :].tolist(),
            "labels": labels[i].tolist()
        }
        trues.append(labels[i])
        res = requests.post(server_b_url, json=sample).json()
        preds.append(res['preds'])
        print(res)

    end = time.time()
    print(f'Elapsed: {end - start:.2f}s')

    # ------------- evaluation -------------
    preds = torch.from_numpy(np.array(preds))
    trues = torch.from_numpy(np.array(trues))

    if task_num in (1, 3):               # regression tasks
        mae = torch.nn.functional.l1_loss(preds, trues)
        mse = torch.nn.functional.mse_loss(preds, trues)
        print('MAE:', mae.item(), 'MSE:', mse.item())
    else:                                # classification task
        y_true = trues.numpy()
        y_pred = preds.argmax(dim=1).numpy()
        print(f'Accuracy : {accuracy_score(y_true, y_pred):.4f}')
        print(f'Precision: {precision_score(y_true, y_pred, average="macro"):.4f}')
        print(f'Recall   : {recall_score(y_true, y_pred, average="macro"):.4f}')
        print(f'F1 Score : {f1_score(y_true, y_pred, average="macro"):.4f}')

if __name__ == '__main__':
    api_test(4)
```
