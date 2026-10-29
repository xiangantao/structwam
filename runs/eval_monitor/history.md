## 2026-10-27 12:10:37

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 177/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=177)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 177 | 67.2% | 74.6% | 67.2% | 21 / 8 | 0.024 |
| libero_spatial | 177 | 67.2% | 74.6% | 67.2% | 21 / 8 | 0.024 |
| Background Textures | 26 | 65.4% | 84.6% | 73.1% | 8 / 3 | 0.227 |
| Camera Viewpoints | 38 | 23.7% | 31.6% | 15.8% | 4 / 1 | 0.375 |
| Language Instructions | 35 | 97.1% | 94.3% | 94.3% | 1 / 2 | 1.000 |
| Objects Layout | 14 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Robot Initial States | 27 | 85.2% | 96.3% | 88.9% | 3 / 0 | 0.250 |
| Sensor Noise | 37 | 59.5% | 67.6% | 62.2% | 5 / 2 | 0.453 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=306)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 306 | 60.1% | 67.3% | 55.9% | 38 / 16 | 0.004 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_spatial | 177 | 67.2% | 74.6% | 67.2% | 21 / 8 | 0.024 |
| Background Textures | 67 | 49.3% | 67.2% | 49.3% | 18 / 6 | 0.023 |
| Camera Viewpoints | 86 | 36.0% | 44.2% | 24.4% | 10 / 3 | 0.092 |
| Language Instructions | 35 | 97.1% | 94.3% | 94.3% | 1 / 2 | 1.000 |
| Objects Layout | 14 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Robot Initial States | 27 | 85.2% | 96.3% | 88.9% | 3 / 0 | 0.250 |
| Sensor Noise | 77 | 63.6% | 64.9% | 59.7% | 6 / 5 | 1.000 |

### StructWAM 40k vs 24k (full eval tasks)  (n=177)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 177 | 67.2% | 74.6% | 18 / 5 | 0.011 |
| libero_spatial | 177 | 67.2% | 74.6% | 18 / 5 | 0.011 |
| Background Textures | 26 | 73.1% | 84.6% | 5 / 2 | 0.453 |
| Camera Viewpoints | 38 | 15.8% | 31.6% | 8 / 2 | 0.109 |
| Language Instructions | 35 | 94.3% | 94.3% | 0 / 0 | 1.000 |
| Objects Layout | 14 | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Robot Initial States | 27 | 88.9% | 96.3% | 2 / 0 | 0.500 |
| Sensor Noise | 37 | 62.2% | 67.6% | 3 / 1 | 0.625 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 5082/40000 | 8.41 s/step | ETA 81:36:44 | loss 0.1351 |
| b0 | 60/40000 | 8.51 s/step | ETA 94:21:50 | loss 1.9472 |
| b2 | 489/40000 | 9.07 s/step | ETA 99:33:55 | loss 0.3663 |

---
## 2026-10-27 12:20:42

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 204/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=204)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 204 | 71.1% | 77.5% | 71.6% | 22 / 9 | 0.029 |
| libero_spatial | 204 | 71.1% | 77.5% | 71.6% | 22 / 9 | 0.029 |
| Background Textures | 26 | 65.4% | 84.6% | 73.1% | 8 / 3 | 0.227 |
| Camera Viewpoints | 38 | 23.7% | 31.6% | 15.8% | 4 / 1 | 0.375 |
| Language Instructions | 35 | 97.1% | 94.3% | 94.3% | 1 / 2 | 1.000 |
| Objects Layout | 41 | 97.6% | 97.6% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 27 | 85.2% | 96.3% | 88.9% | 3 / 0 | 0.250 |
| Sensor Noise | 37 | 59.5% | 67.6% | 62.2% | 5 / 2 | 0.453 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=333)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 333 | 63.1% | 69.7% | 59.5% | 39 / 17 | 0.005 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_spatial | 204 | 71.1% | 77.5% | 71.6% | 22 / 9 | 0.029 |
| Background Textures | 67 | 49.3% | 67.2% | 49.3% | 18 / 6 | 0.023 |
| Camera Viewpoints | 86 | 36.0% | 44.2% | 24.4% | 10 / 3 | 0.092 |
| Language Instructions | 35 | 97.1% | 94.3% | 94.3% | 1 / 2 | 1.000 |
| Objects Layout | 41 | 97.6% | 97.6% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 27 | 85.2% | 96.3% | 88.9% | 3 / 0 | 0.250 |
| Sensor Noise | 77 | 63.6% | 64.9% | 59.7% | 6 / 5 | 1.000 |

### StructWAM 40k vs 24k (full eval tasks)  (n=204)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 204 | 71.6% | 77.5% | 18 / 6 | 0.023 |
| libero_spatial | 204 | 71.6% | 77.5% | 18 / 6 | 0.023 |
| Background Textures | 26 | 73.1% | 84.6% | 5 / 2 | 0.453 |
| Camera Viewpoints | 38 | 15.8% | 31.6% | 8 / 2 | 0.109 |
| Language Instructions | 35 | 94.3% | 94.3% | 0 / 0 | 1.000 |
| Objects Layout | 41 | 100.0% | 97.6% | 0 / 1 | 1.000 |
| Robot Initial States | 27 | 88.9% | 96.3% | 2 / 0 | 0.500 |
| Sensor Noise | 37 | 62.2% | 67.6% | 3 / 1 | 0.625 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 5154/40000 | 8.55 s/step | ETA 82:47:54 | loss 0.1588 |
| b0 | 130/40000 | 8.61 s/step | ETA 95:21:25 | loss 0.8397 |
| b2 | 556/40000 | 8.81 s/step | ETA 96:31:04 | loss 0.3239 |

---
## 2026-10-27 12:30:46

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 228/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=228)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 228 | 74.1% | 79.8% | 74.6% | 22 / 9 | 0.029 |
| libero_spatial | 228 | 74.1% | 79.8% | 74.6% | 22 / 9 | 0.029 |
| Background Textures | 26 | 65.4% | 84.6% | 73.1% | 8 / 3 | 0.227 |
| Camera Viewpoints | 38 | 23.7% | 31.6% | 15.8% | 4 / 1 | 0.375 |
| Language Instructions | 35 | 97.1% | 94.3% | 94.3% | 1 / 2 | 1.000 |
| Light Conditions | 18 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 97.9% | 97.9% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 27 | 85.2% | 96.3% | 88.9% | 3 / 0 | 0.250 |
| Sensor Noise | 37 | 59.5% | 67.6% | 62.2% | 5 / 2 | 0.453 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=357)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 357 | 65.5% | 71.7% | 62.2% | 39 / 17 | 0.005 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_spatial | 228 | 74.1% | 79.8% | 74.6% | 22 / 9 | 0.029 |
| Background Textures | 67 | 49.3% | 67.2% | 49.3% | 18 / 6 | 0.023 |
| Camera Viewpoints | 86 | 36.0% | 44.2% | 24.4% | 10 / 3 | 0.092 |
| Language Instructions | 35 | 97.1% | 94.3% | 94.3% | 1 / 2 | 1.000 |
| Light Conditions | 18 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 97.9% | 97.9% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 27 | 85.2% | 96.3% | 88.9% | 3 / 0 | 0.250 |
| Sensor Noise | 77 | 63.6% | 64.9% | 59.7% | 6 / 5 | 1.000 |

### StructWAM 40k vs 24k (full eval tasks)  (n=228)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 228 | 74.6% | 79.8% | 18 / 6 | 0.023 |
| libero_spatial | 228 | 74.6% | 79.8% | 18 / 6 | 0.023 |
| Background Textures | 26 | 73.1% | 84.6% | 5 / 2 | 0.453 |
| Camera Viewpoints | 38 | 15.8% | 31.6% | 8 / 2 | 0.109 |
| Language Instructions | 35 | 94.3% | 94.3% | 0 / 0 | 1.000 |
| Light Conditions | 18 | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 100.0% | 97.9% | 0 / 1 | 1.000 |
| Robot Initial States | 27 | 88.9% | 96.3% | 2 / 0 | 0.500 |
| Sensor Noise | 37 | 62.2% | 67.6% | 3 / 1 | 0.625 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 5226/40000 | 8.34 s/step | ETA 80:33:57 | loss 0.1483 |
| b0 | 198/40000 | 8.69 s/step | ETA 96:04:02 | loss 0.6141 |
| b2 | 623/40000 | 8.95 s/step | ETA 97:50:54 | loss 0.3474 |

---
## 2026-10-27 12:40:51

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 252/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=252)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 252 | 75.4% | 81.7% | 76.6% | 25 / 9 | 0.009 |
| libero_object | 11 | 72.7% | 100.0% | 90.9% | 3 / 0 | 0.250 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 37 | 67.6% | 89.2% | 78.4% | 11 / 3 | 0.057 |
| Camera Viewpoints | 38 | 23.7% | 31.6% | 15.8% | 4 / 1 | 0.375 |
| Language Instructions | 35 | 97.1% | 94.3% | 94.3% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 97.9% | 97.9% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 27 | 85.2% | 96.3% | 88.9% | 3 / 0 | 0.250 |
| Sensor Noise | 37 | 59.5% | 67.6% | 62.2% | 5 / 2 | 0.453 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=381)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 381 | 66.9% | 73.5% | 64.3% | 42 / 17 | 0.002 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_object | 11 | 72.7% | 100.0% | 90.9% | 3 / 0 | 0.250 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 78 | 52.6% | 71.8% | 55.1% | 21 / 6 | 0.006 |
| Camera Viewpoints | 86 | 36.0% | 44.2% | 24.4% | 10 / 3 | 0.092 |
| Language Instructions | 35 | 97.1% | 94.3% | 94.3% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 97.9% | 97.9% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 27 | 85.2% | 96.3% | 88.9% | 3 / 0 | 0.250 |
| Sensor Noise | 77 | 63.6% | 64.9% | 59.7% | 6 / 5 | 1.000 |

### StructWAM 40k vs 24k (full eval tasks)  (n=252)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 252 | 76.6% | 81.7% | 19 / 6 | 0.015 |
| libero_object | 11 | 90.9% | 100.0% | 1 / 0 | 1.000 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 37 | 78.4% | 89.2% | 6 / 2 | 0.289 |
| Camera Viewpoints | 38 | 15.8% | 31.6% | 8 / 2 | 0.109 |
| Language Instructions | 35 | 94.3% | 94.3% | 0 / 0 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 100.0% | 97.9% | 0 / 1 | 1.000 |
| Robot Initial States | 27 | 88.9% | 96.3% | 2 / 0 | 0.500 |
| Sensor Noise | 37 | 62.2% | 67.6% | 3 / 1 | 0.625 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 5298/40000 | 8.34 s/step | ETA 80:23:05 | loss 0.1075 |
| b0 | 266/40000 | 9.37 s/step | ETA 103:22:34 | loss 0.5732 |
| b2 | 690/40000 | 9.06 s/step | ETA 98:57:28 | loss 0.2839 |

---
## 2026-10-27 12:50:55

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 273/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=273)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 273 | 75.5% | 81.0% | 76.6% | 25 / 10 | 0.017 |
| libero_object | 32 | 75.0% | 81.2% | 81.2% | 3 / 1 | 0.625 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 52 | 67.3% | 80.8% | 75.0% | 11 / 4 | 0.118 |
| Camera Viewpoints | 38 | 23.7% | 31.6% | 15.8% | 4 / 1 | 0.375 |
| Language Instructions | 35 | 97.1% | 94.3% | 94.3% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 97.9% | 97.9% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 33 | 87.9% | 97.0% | 90.9% | 3 / 0 | 0.250 |
| Sensor Noise | 37 | 59.5% | 67.6% | 62.2% | 5 / 2 | 0.453 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=402)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 402 | 67.4% | 73.4% | 64.9% | 42 / 18 | 0.003 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_object | 32 | 75.0% | 81.2% | 81.2% | 3 / 1 | 0.625 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 93 | 54.8% | 69.9% | 57.0% | 21 / 7 | 0.013 |
| Camera Viewpoints | 86 | 36.0% | 44.2% | 24.4% | 10 / 3 | 0.092 |
| Language Instructions | 35 | 97.1% | 94.3% | 94.3% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 97.9% | 97.9% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 33 | 87.9% | 97.0% | 90.9% | 3 / 0 | 0.250 |
| Sensor Noise | 77 | 63.6% | 64.9% | 59.7% | 6 / 5 | 1.000 |

### StructWAM 40k vs 24k (full eval tasks)  (n=273)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 273 | 76.6% | 81.0% | 19 / 7 | 0.029 |
| libero_object | 32 | 81.2% | 81.2% | 1 / 1 | 1.000 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 52 | 75.0% | 80.8% | 6 / 3 | 0.508 |
| Camera Viewpoints | 38 | 15.8% | 31.6% | 8 / 2 | 0.109 |
| Language Instructions | 35 | 94.3% | 94.3% | 0 / 0 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 100.0% | 97.9% | 0 / 1 | 1.000 |
| Robot Initial States | 33 | 90.9% | 97.0% | 2 / 0 | 0.500 |
| Sensor Noise | 37 | 62.2% | 67.6% | 3 / 1 | 0.625 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 5370/40000 | 8.31 s/step | ETA 79:56:46 | loss 0.1548 |
| b0 | 335/40000 | 8.86 s/step | ETA 97:37:55 | loss 0.5988 |
| b2 | 755/40000 | 8.99 s/step | ETA 98:00:43 | loss 0.2706 |

---
## 2026-10-27 13:01:00

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 301/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=301)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 301 | 75.7% | 82.4% | 77.7% | 30 / 10 | 0.002 |
| libero_object | 60 | 76.7% | 88.3% | 85.0% | 8 / 1 | 0.039 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 53 | 66.0% | 79.2% | 73.6% | 11 / 4 | 0.118 |
| Camera Viewpoints | 38 | 23.7% | 31.6% | 15.8% | 4 / 1 | 0.375 |
| Language Instructions | 35 | 97.1% | 94.3% | 94.3% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 97.9% | 97.9% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 60 | 85.0% | 98.3% | 91.7% | 8 / 0 | 0.008 |
| Sensor Noise | 37 | 59.5% | 67.6% | 62.2% | 5 / 2 | 0.453 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=430)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 430 | 68.1% | 74.9% | 66.5% | 47 / 18 | 0.000 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_object | 60 | 76.7% | 88.3% | 85.0% | 8 / 1 | 0.039 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 94 | 54.3% | 69.1% | 56.4% | 21 / 7 | 0.013 |
| Camera Viewpoints | 86 | 36.0% | 44.2% | 24.4% | 10 / 3 | 0.092 |
| Language Instructions | 35 | 97.1% | 94.3% | 94.3% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 97.9% | 97.9% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 60 | 85.0% | 98.3% | 91.7% | 8 / 0 | 0.008 |
| Sensor Noise | 77 | 63.6% | 64.9% | 59.7% | 6 / 5 | 1.000 |

### StructWAM 40k vs 24k (full eval tasks)  (n=301)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 301 | 77.7% | 82.4% | 21 / 7 | 0.013 |
| libero_object | 60 | 85.0% | 88.3% | 3 / 1 | 0.625 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 53 | 73.6% | 79.2% | 6 / 3 | 0.508 |
| Camera Viewpoints | 38 | 15.8% | 31.6% | 8 / 2 | 0.109 |
| Language Instructions | 35 | 94.3% | 94.3% | 0 / 0 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 100.0% | 97.9% | 0 / 1 | 1.000 |
| Robot Initial States | 60 | 91.7% | 98.3% | 4 / 0 | 0.125 |
| Sensor Noise | 37 | 62.2% | 67.6% | 3 / 1 | 0.625 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 5441/40000 | 8.18 s/step | ETA 78:34:23 | loss 0.1565 |
| b0 | 402/40000 | 8.87 s/step | ETA 97:34:42 | loss 0.4335 |
| b2 | 822/40000 | 9.30 s/step | ETA 101:11:56 | loss 0.2656 |

---
## 2026-10-27 13:11:05

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 323/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=323)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 323 | 76.8% | 82.7% | 78.0% | 31 / 12 | 0.005 |
| libero_object | 82 | 80.5% | 87.8% | 84.1% | 9 / 3 | 0.146 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 53 | 66.0% | 79.2% | 73.6% | 11 / 4 | 0.118 |
| Camera Viewpoints | 49 | 38.8% | 46.9% | 26.5% | 5 / 1 | 0.219 |
| Language Instructions | 35 | 97.1% | 94.3% | 94.3% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 97.9% | 97.9% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 37 | 59.5% | 67.6% | 62.2% | 5 / 2 | 0.453 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=452)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 452 | 69.2% | 75.4% | 67.3% | 48 / 20 | 0.001 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_object | 82 | 80.5% | 87.8% | 84.1% | 9 / 3 | 0.146 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 94 | 54.3% | 69.1% | 56.4% | 21 / 7 | 0.013 |
| Camera Viewpoints | 97 | 42.3% | 50.5% | 28.9% | 11 / 3 | 0.057 |
| Language Instructions | 35 | 97.1% | 94.3% | 94.3% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 97.9% | 97.9% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 77 | 63.6% | 64.9% | 59.7% | 6 / 5 | 1.000 |

### StructWAM 40k vs 24k (full eval tasks)  (n=323)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 323 | 78.0% | 82.7% | 25 / 10 | 0.017 |
| libero_object | 82 | 84.1% | 87.8% | 7 / 4 | 0.549 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 53 | 73.6% | 79.2% | 6 / 3 | 0.508 |
| Camera Viewpoints | 49 | 26.5% | 46.9% | 12 / 2 | 0.013 |
| Language Instructions | 35 | 94.3% | 94.3% | 0 / 0 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 100.0% | 97.9% | 0 / 1 | 1.000 |
| Robot Initial States | 71 | 93.0% | 94.4% | 4 / 3 | 1.000 |
| Sensor Noise | 37 | 62.2% | 67.6% | 3 / 1 | 0.625 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 5512/40000 | 8.43 s/step | ETA 80:48:06 | loss 0.1486 |
| b0 | 468/40000 | 9.24 s/step | ETA 101:24:38 | loss 0.4651 |
| b2 | 888/40000 | 8.94 s/step | ETA 97:07:25 | loss 0.2318 |

---
## 2026-10-27 13:21:10

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 333/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=333)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 333 | 75.1% | 81.1% | 76.3% | 32 / 12 | 0.004 |
| libero_object | 92 | 73.9% | 81.5% | 77.2% | 10 / 3 | 0.092 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 53 | 66.0% | 79.2% | 73.6% | 11 / 4 | 0.118 |
| Camera Viewpoints | 59 | 35.6% | 44.1% | 25.4% | 6 / 1 | 0.125 |
| Language Instructions | 35 | 97.1% | 94.3% | 94.3% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 97.9% | 97.9% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 37 | 59.5% | 67.6% | 62.2% | 5 / 2 | 0.453 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=462)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 462 | 68.2% | 74.5% | 66.2% | 49 / 20 | 0.001 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_object | 92 | 73.9% | 81.5% | 77.2% | 10 / 3 | 0.092 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 94 | 54.3% | 69.1% | 56.4% | 21 / 7 | 0.013 |
| Camera Viewpoints | 107 | 40.2% | 48.6% | 28.0% | 12 / 3 | 0.035 |
| Language Instructions | 35 | 97.1% | 94.3% | 94.3% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 97.9% | 97.9% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 77 | 63.6% | 64.9% | 59.7% | 6 / 5 | 1.000 |

### StructWAM 40k vs 24k (full eval tasks)  (n=333)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 333 | 76.3% | 81.1% | 26 / 10 | 0.011 |
| libero_object | 92 | 77.2% | 81.5% | 8 / 4 | 0.388 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 53 | 73.6% | 79.2% | 6 / 3 | 0.508 |
| Camera Viewpoints | 59 | 25.4% | 44.1% | 13 / 2 | 0.007 |
| Language Instructions | 35 | 94.3% | 94.3% | 0 / 0 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 100.0% | 97.9% | 0 / 1 | 1.000 |
| Robot Initial States | 71 | 93.0% | 94.4% | 4 / 3 | 1.000 |
| Sensor Noise | 37 | 62.2% | 67.6% | 3 / 1 | 0.625 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 5583/40000 | 8.54 s/step | ETA 81:39:57 | loss 0.1309 |
| b0 | 536/40000 | 9.13 s/step | ETA 100:06:39 | loss 0.3786 |
| b2 | 954/40000 | 9.13 s/step | ETA 98:59:58 | loss 0.2422 |

---
## 2026-10-27 13:31:15

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 345/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=345)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 345 | 73.0% | 78.8% | 73.9% | 33 / 13 | 0.005 |
| libero_object | 104 | 67.3% | 74.0% | 69.2% | 11 / 4 | 0.118 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 53 | 66.0% | 79.2% | 73.6% | 11 / 4 | 0.118 |
| Camera Viewpoints | 71 | 32.4% | 39.4% | 22.5% | 7 / 2 | 0.180 |
| Language Instructions | 35 | 97.1% | 94.3% | 94.3% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 97.9% | 97.9% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 37 | 59.5% | 67.6% | 62.2% | 5 / 2 | 0.453 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=474)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 474 | 66.9% | 73.0% | 64.8% | 50 / 21 | 0.001 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_object | 104 | 67.3% | 74.0% | 69.2% | 11 / 4 | 0.118 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 94 | 54.3% | 69.1% | 56.4% | 21 / 7 | 0.013 |
| Camera Viewpoints | 119 | 37.8% | 45.4% | 26.1% | 13 / 4 | 0.049 |
| Language Instructions | 35 | 97.1% | 94.3% | 94.3% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 97.9% | 97.9% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 77 | 63.6% | 64.9% | 59.7% | 6 / 5 | 1.000 |

### StructWAM 40k vs 24k (full eval tasks)  (n=345)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 345 | 73.9% | 78.8% | 27 / 10 | 0.008 |
| libero_object | 104 | 69.2% | 74.0% | 9 / 4 | 0.267 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 53 | 73.6% | 79.2% | 6 / 3 | 0.508 |
| Camera Viewpoints | 71 | 22.5% | 39.4% | 14 / 2 | 0.004 |
| Language Instructions | 35 | 94.3% | 94.3% | 0 / 0 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 100.0% | 97.9% | 0 / 1 | 1.000 |
| Robot Initial States | 71 | 93.0% | 94.4% | 4 / 3 | 1.000 |
| Sensor Noise | 37 | 62.2% | 67.6% | 3 / 1 | 0.625 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 5655/40000 | 8.40 s/step | ETA 80:08:51 | loss 0.1412 |
| b0 | 603/40000 | 8.91 s/step | ETA 97:29:16 | loss 0.3504 |
| b2 | 1022/40000 | 9.07 s/step | ETA 98:11:34 | loss 0.2081 |

---
## 2026-10-27 13:41:20

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 377/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=377)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 377 | 74.8% | 79.8% | 75.3% | 33 / 14 | 0.008 |
| libero_object | 136 | 73.5% | 77.9% | 74.3% | 11 / 5 | 0.210 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 53 | 66.0% | 79.2% | 73.6% | 11 / 4 | 0.118 |
| Camera Viewpoints | 75 | 33.3% | 38.7% | 22.7% | 7 / 3 | 0.344 |
| Language Instructions | 63 | 98.4% | 96.8% | 96.8% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 97.9% | 97.9% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 37 | 59.5% | 67.6% | 62.2% | 5 / 2 | 0.453 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=506)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 506 | 68.6% | 74.1% | 66.4% | 50 / 22 | 0.001 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_object | 136 | 73.5% | 77.9% | 74.3% | 11 / 5 | 0.210 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 94 | 54.3% | 69.1% | 56.4% | 21 / 7 | 0.013 |
| Camera Viewpoints | 123 | 38.2% | 44.7% | 26.0% | 13 / 5 | 0.096 |
| Language Instructions | 63 | 98.4% | 96.8% | 96.8% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 97.9% | 97.9% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 77 | 63.6% | 64.9% | 59.7% | 6 / 5 | 1.000 |

### StructWAM 40k vs 24k (full eval tasks)  (n=377)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 377 | 75.3% | 79.8% | 28 / 11 | 0.009 |
| libero_object | 136 | 74.3% | 77.9% | 10 / 5 | 0.302 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 53 | 73.6% | 79.2% | 6 / 3 | 0.508 |
| Camera Viewpoints | 75 | 22.7% | 38.7% | 15 / 3 | 0.008 |
| Language Instructions | 63 | 96.8% | 96.8% | 0 / 0 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 100.0% | 97.9% | 0 / 1 | 1.000 |
| Robot Initial States | 71 | 93.0% | 94.4% | 4 / 3 | 1.000 |
| Sensor Noise | 37 | 62.2% | 67.6% | 3 / 1 | 0.625 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 5727/40000 | 8.40 s/step | ETA 79:58:45 | loss 0.1753 |
| b0 | 670/40000 | 9.05 s/step | ETA 98:51:34 | loss 0.3209 |
| b2 | 1088/40000 | 9.08 s/step | ETA 98:08:29 | loss 0.2281 |

---
## 2026-10-27 13:51:25

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 388/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=388)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 388 | 74.5% | 79.6% | 75.0% | 34 / 14 | 0.006 |
| libero_object | 147 | 72.8% | 77.6% | 73.5% | 12 / 5 | 0.143 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 53 | 66.0% | 79.2% | 73.6% | 11 / 4 | 0.118 |
| Camera Viewpoints | 75 | 33.3% | 38.7% | 22.7% | 7 / 3 | 0.344 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 97.9% | 97.9% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 42 | 54.8% | 64.3% | 57.1% | 6 / 2 | 0.289 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=517)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 517 | 68.5% | 74.1% | 66.3% | 51 / 22 | 0.001 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_object | 147 | 72.8% | 77.6% | 73.5% | 12 / 5 | 0.143 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 94 | 54.3% | 69.1% | 56.4% | 21 / 7 | 0.013 |
| Camera Viewpoints | 123 | 38.2% | 44.7% | 26.0% | 13 / 5 | 0.096 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 97.9% | 97.9% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 82 | 61.0% | 63.4% | 57.3% | 7 / 5 | 0.774 |

### StructWAM 40k vs 24k (full eval tasks)  (n=388)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 388 | 75.0% | 79.6% | 29 / 11 | 0.006 |
| libero_object | 147 | 73.5% | 77.6% | 11 / 5 | 0.210 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 53 | 73.6% | 79.2% | 6 / 3 | 0.508 |
| Camera Viewpoints | 75 | 22.7% | 38.7% | 15 / 3 | 0.008 |
| Language Instructions | 69 | 97.1% | 97.1% | 0 / 0 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 100.0% | 97.9% | 0 / 1 | 1.000 |
| Robot Initial States | 71 | 93.0% | 94.4% | 4 / 3 | 1.000 |
| Sensor Noise | 42 | 57.1% | 64.3% | 4 / 1 | 0.375 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 5799/40000 | 8.37 s/step | ETA 79:30:49 | loss 0.1193 |
| b0 | 736/40000 | 9.04 s/step | ETA 98:35:41 | loss 0.3299 |
| b2 | 1155/40000 | 9.15 s/step | ETA 98:44:45 | loss 0.2210 |

---
## 2026-10-27 14:01:30

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 395/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=395)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 395 | 74.4% | 79.2% | 74.4% | 34 / 15 | 0.009 |
| libero_object | 154 | 72.7% | 76.6% | 72.1% | 12 / 6 | 0.238 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 53 | 66.0% | 79.2% | 73.6% | 11 / 4 | 0.118 |
| Camera Viewpoints | 75 | 33.3% | 38.7% | 22.7% | 7 / 3 | 0.344 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 97.9% | 97.9% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 49 | 57.1% | 63.3% | 55.1% | 6 / 3 | 0.508 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=524)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 524 | 68.5% | 73.9% | 66.0% | 51 / 23 | 0.002 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_object | 154 | 72.7% | 76.6% | 72.1% | 12 / 6 | 0.238 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 94 | 54.3% | 69.1% | 56.4% | 21 / 7 | 0.013 |
| Camera Viewpoints | 123 | 38.2% | 44.7% | 26.0% | 13 / 5 | 0.096 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 97.9% | 97.9% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 89 | 61.8% | 62.9% | 56.2% | 7 / 6 | 1.000 |

### StructWAM 40k vs 24k (full eval tasks)  (n=395)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 395 | 74.4% | 79.2% | 30 / 11 | 0.004 |
| libero_object | 154 | 72.1% | 76.6% | 12 / 5 | 0.143 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 53 | 73.6% | 79.2% | 6 / 3 | 0.508 |
| Camera Viewpoints | 75 | 22.7% | 38.7% | 15 / 3 | 0.008 |
| Language Instructions | 69 | 97.1% | 97.1% | 0 / 0 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 100.0% | 97.9% | 0 / 1 | 1.000 |
| Robot Initial States | 71 | 93.0% | 94.4% | 4 / 3 | 1.000 |
| Sensor Noise | 49 | 55.1% | 63.3% | 5 / 1 | 0.219 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 5871/40000 | 8.89 s/step | ETA 84:17:19 | loss 0.1295 |
| b0 | 803/40000 | 9.02 s/step | ETA 98:09:21 | loss 0.2991 |
| b2 | 1221/40000 | 8.85 s/step | ETA 95:21:53 | loss 0.2486 |

---
## 2026-10-27 14:11:35

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 408/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=408)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 408 | 73.8% | 78.4% | 73.5% | 34 / 15 | 0.009 |
| libero_object | 167 | 71.3% | 74.9% | 70.1% | 12 / 6 | 0.238 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 53 | 66.0% | 79.2% | 73.6% | 11 / 4 | 0.118 |
| Camera Viewpoints | 75 | 33.3% | 38.7% | 22.7% | 7 / 3 | 0.344 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 97.9% | 97.9% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 62 | 56.5% | 61.3% | 53.2% | 6 / 3 | 0.508 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=537)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 537 | 68.2% | 73.4% | 65.5% | 51 / 23 | 0.002 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_object | 167 | 71.3% | 74.9% | 70.1% | 12 / 6 | 0.238 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 94 | 54.3% | 69.1% | 56.4% | 21 / 7 | 0.013 |
| Camera Viewpoints | 123 | 38.2% | 44.7% | 26.0% | 13 / 5 | 0.096 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 97.9% | 97.9% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 102 | 60.8% | 61.8% | 54.9% | 7 / 6 | 1.000 |

### StructWAM 40k vs 24k (full eval tasks)  (n=408)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 408 | 73.5% | 78.4% | 31 / 11 | 0.003 |
| libero_object | 167 | 70.1% | 74.9% | 13 / 5 | 0.096 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 53 | 73.6% | 79.2% | 6 / 3 | 0.508 |
| Camera Viewpoints | 75 | 22.7% | 38.7% | 15 / 3 | 0.008 |
| Language Instructions | 69 | 97.1% | 97.1% | 0 / 0 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 100.0% | 97.9% | 0 / 1 | 1.000 |
| Robot Initial States | 71 | 93.0% | 94.4% | 4 / 3 | 1.000 |
| Sensor Noise | 62 | 53.2% | 61.3% | 6 / 1 | 0.125 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 5941/40000 | 8.47 s/step | ETA 80:07:15 | loss 0.1385 |
| b0 | 871/40000 | 8.98 s/step | ETA 97:36:19 | loss 0.2770 |
| b2 | 1288/40000 | 9.41 s/step | ETA 101:11:18 | loss 0.2295 |

---
## 2026-10-27 14:21:40

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 415/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=415)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 415 | 73.0% | 77.8% | 72.8% | 35 / 15 | 0.007 |
| libero_object | 174 | 69.5% | 73.6% | 68.4% | 13 / 6 | 0.167 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 53 | 66.0% | 79.2% | 73.6% | 11 / 4 | 0.118 |
| Camera Viewpoints | 75 | 33.3% | 38.7% | 22.7% | 7 / 3 | 0.344 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 97.9% | 97.9% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 69 | 53.6% | 59.4% | 50.7% | 7 / 3 | 0.344 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=544)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 544 | 67.6% | 73.0% | 65.1% | 52 / 23 | 0.001 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_object | 174 | 69.5% | 73.6% | 68.4% | 13 / 6 | 0.167 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 94 | 54.3% | 69.1% | 56.4% | 21 / 7 | 0.013 |
| Camera Viewpoints | 123 | 38.2% | 44.7% | 26.0% | 13 / 5 | 0.096 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 97.9% | 97.9% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 109 | 58.7% | 60.6% | 53.2% | 8 / 6 | 0.791 |

### StructWAM 40k vs 24k (full eval tasks)  (n=415)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 415 | 72.8% | 77.8% | 32 / 11 | 0.002 |
| libero_object | 174 | 68.4% | 73.6% | 14 / 5 | 0.064 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 53 | 73.6% | 79.2% | 6 / 3 | 0.508 |
| Camera Viewpoints | 75 | 22.7% | 38.7% | 15 / 3 | 0.008 |
| Language Instructions | 69 | 97.1% | 97.1% | 0 / 0 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 47 | 100.0% | 97.9% | 0 / 1 | 1.000 |
| Robot Initial States | 71 | 93.0% | 94.4% | 4 / 3 | 1.000 |
| Sensor Noise | 69 | 50.7% | 59.4% | 7 / 1 | 0.070 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 6000/40000 | 8.56 s/step | ETA 80:49:37 | loss 0.1480 |
| b0 | 937/40000 | 8.88 s/step | ETA 96:19:40 | loss 0.2792 |
| b2 | 1354/40000 | 9.18 s/step | ETA 98:34:21 | loss 0.2195 |

---
## 2026-10-27 14:31:45

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 421/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=421)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 421 | 72.4% | 77.4% | 72.2% | 36 / 15 | 0.005 |
| libero_object | 180 | 68.3% | 72.8% | 67.2% | 14 / 6 | 0.115 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 53 | 66.0% | 79.2% | 73.6% | 11 / 4 | 0.118 |
| Camera Viewpoints | 75 | 33.3% | 38.7% | 22.7% | 7 / 3 | 0.344 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 49 | 98.0% | 98.0% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 73 | 50.7% | 57.5% | 47.9% | 8 / 3 | 0.227 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=550)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 550 | 67.3% | 72.7% | 64.7% | 53 / 23 | 0.001 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_object | 180 | 68.3% | 72.8% | 67.2% | 14 / 6 | 0.115 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 94 | 54.3% | 69.1% | 56.4% | 21 / 7 | 0.013 |
| Camera Viewpoints | 123 | 38.2% | 44.7% | 26.0% | 13 / 5 | 0.096 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 49 | 98.0% | 98.0% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 113 | 56.6% | 59.3% | 51.3% | 9 / 6 | 0.607 |

### StructWAM 40k vs 24k (full eval tasks)  (n=421)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 421 | 72.2% | 77.4% | 33 / 11 | 0.001 |
| libero_object | 180 | 67.2% | 72.8% | 15 / 5 | 0.041 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 53 | 73.6% | 79.2% | 6 / 3 | 0.508 |
| Camera Viewpoints | 75 | 22.7% | 38.7% | 15 / 3 | 0.008 |
| Language Instructions | 69 | 97.1% | 97.1% | 0 / 0 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 49 | 100.0% | 98.0% | 0 / 1 | 1.000 |
| Robot Initial States | 71 | 93.0% | 94.4% | 4 / 3 | 1.000 |
| Sensor Noise | 73 | 47.9% | 57.5% | 8 / 1 | 0.039 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 6059/40000 | 8.51 s/step | ETA 80:15:15 | loss 0.1329 |
| b0 | 1004/40000 | 8.99 s/step | ETA 97:25:34 | loss 0.2714 |
| b2 | 1422/40000 | 8.87 s/step | ETA 95:02:00 | loss 0.2245 |

---
## 2026-10-27 14:41:50

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 434/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=434)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 434 | 73.0% | 77.9% | 72.6% | 36 / 15 | 0.005 |
| libero_object | 193 | 69.9% | 74.1% | 68.4% | 14 / 6 | 0.115 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 53 | 66.0% | 79.2% | 73.6% | 11 / 4 | 0.118 |
| Camera Viewpoints | 75 | 33.3% | 38.7% | 22.7% | 7 / 3 | 0.344 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 58 | 98.3% | 98.3% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 77 | 51.9% | 58.4% | 48.1% | 8 / 3 | 0.227 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=563)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 563 | 67.9% | 73.2% | 65.2% | 53 / 23 | 0.001 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_object | 193 | 69.9% | 74.1% | 68.4% | 14 / 6 | 0.115 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 94 | 54.3% | 69.1% | 56.4% | 21 / 7 | 0.013 |
| Camera Viewpoints | 123 | 38.2% | 44.7% | 26.0% | 13 / 5 | 0.096 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 58 | 98.3% | 98.3% | 100.0% | 1 / 1 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 117 | 57.3% | 59.8% | 51.3% | 9 / 6 | 0.607 |

### StructWAM 40k vs 24k (full eval tasks)  (n=434)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 434 | 72.6% | 77.9% | 34 / 11 | 0.001 |
| libero_object | 193 | 68.4% | 74.1% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 53 | 73.6% | 79.2% | 6 / 3 | 0.508 |
| Camera Viewpoints | 75 | 22.7% | 38.7% | 15 / 3 | 0.008 |
| Language Instructions | 69 | 97.1% | 97.1% | 0 / 0 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 58 | 100.0% | 98.3% | 0 / 1 | 1.000 |
| Robot Initial States | 71 | 93.0% | 94.4% | 4 / 3 | 1.000 |
| Sensor Noise | 77 | 48.1% | 58.4% | 9 / 1 | 0.021 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 6130/40000 | 8.53 s/step | ETA 80:16:38 | loss 0.1395 |
| b0 | 1071/40000 | 8.89 s/step | ETA 96:09:58 | loss 0.2803 |
| b2 | 1489/40000 | 9.24 s/step | ETA 98:49:32 | loss 0.2568 |

---
## 2026-10-27 14:51:55

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 455/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=455)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 455 | 73.4% | 77.8% | 72.7% | 37 / 17 | 0.009 |
| libero_object | 214 | 71.0% | 74.3% | 69.2% | 15 / 8 | 0.210 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 53 | 66.0% | 79.2% | 73.6% | 11 / 4 | 0.118 |
| Camera Viewpoints | 75 | 33.3% | 38.7% | 22.7% | 7 / 3 | 0.344 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 78 | 93.6% | 93.6% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 78 | 52.6% | 57.7% | 47.4% | 8 / 4 | 0.388 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=584)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 584 | 68.3% | 73.3% | 65.6% | 54 / 25 | 0.001 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_object | 214 | 71.0% | 74.3% | 69.2% | 15 / 8 | 0.210 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 94 | 54.3% | 69.1% | 56.4% | 21 / 7 | 0.013 |
| Camera Viewpoints | 123 | 38.2% | 44.7% | 26.0% | 13 / 5 | 0.096 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 78 | 93.6% | 93.6% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 118 | 57.6% | 59.3% | 50.8% | 9 / 7 | 0.804 |

### StructWAM 40k vs 24k (full eval tasks)  (n=455)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 455 | 72.7% | 77.8% | 34 / 11 | 0.001 |
| libero_object | 214 | 69.2% | 74.3% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 53 | 73.6% | 79.2% | 6 / 3 | 0.508 |
| Camera Viewpoints | 75 | 22.7% | 38.7% | 15 / 3 | 0.008 |
| Language Instructions | 69 | 97.1% | 97.1% | 0 / 0 | 1.000 |
| Light Conditions | 31 | 100.0% | 100.0% | 0 / 0 | 1.000 |
| Objects Layout | 78 | 94.9% | 93.6% | 0 / 1 | 1.000 |
| Robot Initial States | 71 | 93.0% | 94.4% | 4 / 3 | 1.000 |
| Sensor Noise | 78 | 47.4% | 57.7% | 9 / 1 | 0.021 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 6202/40000 | 8.66 s/step | ETA 81:19:10 | loss 0.1293 |
| b0 | 1138/40000 | 9.01 s/step | ETA 97:13:45 | loss 0.2389 |
| b2 | 1558/40000 | 8.88 s/step | ETA 94:47:54 | loss 0.2289 |

---
## 2026-10-27 15:02:01

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 473/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=473)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 473 | 74.2% | 78.2% | 73.4% | 37 / 18 | 0.014 |
| libero_object | 232 | 72.8% | 75.4% | 70.7% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 53 | 66.0% | 79.2% | 73.6% | 11 / 4 | 0.118 |
| Camera Viewpoints | 75 | 33.3% | 38.7% | 22.7% | 7 / 3 | 0.344 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 47 | 100.0% | 97.9% | 97.9% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 79 | 51.9% | 57.0% | 46.8% | 8 / 4 | 0.388 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=602)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 602 | 69.1% | 73.8% | 66.3% | 54 / 26 | 0.002 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_object | 232 | 72.8% | 75.4% | 70.7% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 94 | 54.3% | 69.1% | 56.4% | 21 / 7 | 0.013 |
| Camera Viewpoints | 123 | 38.2% | 44.7% | 26.0% | 13 / 5 | 0.096 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 47 | 100.0% | 97.9% | 97.9% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 119 | 57.1% | 58.8% | 50.4% | 9 / 7 | 0.804 |

### StructWAM 40k vs 24k (full eval tasks)  (n=473)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 473 | 73.4% | 78.2% | 34 / 11 | 0.001 |
| libero_object | 232 | 70.7% | 75.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 53 | 73.6% | 79.2% | 6 / 3 | 0.508 |
| Camera Viewpoints | 75 | 22.7% | 38.7% | 15 / 3 | 0.008 |
| Language Instructions | 69 | 97.1% | 97.1% | 0 / 0 | 1.000 |
| Light Conditions | 47 | 97.9% | 97.9% | 0 / 0 | 1.000 |
| Objects Layout | 79 | 94.9% | 93.7% | 0 / 1 | 1.000 |
| Robot Initial States | 71 | 93.0% | 94.4% | 4 / 3 | 1.000 |
| Sensor Noise | 79 | 46.8% | 57.0% | 9 / 1 | 0.021 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 6274/40000 | 8.44 s/step | ETA 79:03:53 | loss 0.1769 |
| b0 | 1206/40000 | 8.87 s/step | ETA 95:35:07 | loss 0.2538 |
| b2 | 1624/40000 | 8.94 s/step | ETA 95:15:35 | loss 0.1935 |

---
## 2026-10-27 15:12:06

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 496/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=496)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 496 | 75.4% | 79.2% | 74.6% | 37 / 18 | 0.014 |
| libero_goal | 3 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 56 | 67.9% | 80.4% | 75.0% | 11 / 4 | 0.118 |
| Camera Viewpoints | 75 | 33.3% | 38.7% | 22.7% | 7 / 3 | 0.344 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 79 | 51.9% | 57.0% | 46.8% | 8 / 4 | 0.388 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=625)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 625 | 70.2% | 74.7% | 67.5% | 54 / 26 | 0.002 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_goal | 3 | 100.0% | 100.0% | 100.0% | 0 / 0 | 1.000 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 97 | 55.7% | 70.1% | 57.7% | 21 / 7 | 0.013 |
| Camera Viewpoints | 123 | 38.2% | 44.7% | 26.0% | 13 / 5 | 0.096 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 119 | 57.1% | 58.8% | 50.4% | 9 / 7 | 0.804 |

### StructWAM 40k vs 24k (full eval tasks)  (n=496)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 496 | 74.6% | 79.2% | 34 / 11 | 0.001 |
| libero_goal | 3 | 100.0% | 100.0% | 0 / 0 | 1.000 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 56 | 75.0% | 80.4% | 6 / 3 | 0.508 |
| Camera Viewpoints | 75 | 22.7% | 38.7% | 15 / 3 | 0.008 |
| Language Instructions | 69 | 97.1% | 97.1% | 0 / 0 | 1.000 |
| Light Conditions | 67 | 98.5% | 98.5% | 0 / 0 | 1.000 |
| Objects Layout | 79 | 94.9% | 93.7% | 0 / 1 | 1.000 |
| Robot Initial States | 71 | 93.0% | 94.4% | 4 / 3 | 1.000 |
| Sensor Noise | 79 | 46.8% | 57.0% | 9 / 1 | 0.021 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 6346/40000 | 8.37 s/step | ETA 78:14:08 | loss 0.1286 |
| b0 | 1272/40000 | 9.16 s/step | ETA 98:34:33 | loss 0.2118 |
| b2 | 1691/40000 | 8.92 s/step | ETA 94:54:52 | loss 0.2371 |

---
## 2026-10-27 15:22:12

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 512/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=512)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 512 | 75.8% | 79.7% | 75.2% | 39 / 19 | 0.012 |
| libero_goal | 19 | 89.5% | 94.7% | 94.7% | 2 / 1 | 1.000 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 72 | 72.2% | 83.3% | 79.2% | 13 / 5 | 0.096 |
| Camera Viewpoints | 75 | 33.3% | 38.7% | 22.7% | 7 / 3 | 0.344 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 79 | 51.9% | 57.0% | 46.8% | 8 / 4 | 0.388 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=641)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 641 | 70.7% | 75.2% | 68.2% | 56 / 27 | 0.002 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_goal | 19 | 89.5% | 94.7% | 94.7% | 2 / 1 | 1.000 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 113 | 60.2% | 73.5% | 62.8% | 23 / 8 | 0.011 |
| Camera Viewpoints | 123 | 38.2% | 44.7% | 26.0% | 13 / 5 | 0.096 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 119 | 57.1% | 58.8% | 50.4% | 9 / 7 | 0.804 |

### StructWAM 40k vs 24k (full eval tasks)  (n=512)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 512 | 75.2% | 79.7% | 35 / 12 | 0.001 |
| libero_goal | 19 | 94.7% | 94.7% | 1 / 1 | 1.000 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 72 | 79.2% | 83.3% | 7 / 4 | 0.549 |
| Camera Viewpoints | 75 | 22.7% | 38.7% | 15 / 3 | 0.008 |
| Language Instructions | 69 | 97.1% | 97.1% | 0 / 0 | 1.000 |
| Light Conditions | 67 | 98.5% | 98.5% | 0 / 0 | 1.000 |
| Objects Layout | 79 | 94.9% | 93.7% | 0 / 1 | 1.000 |
| Robot Initial States | 71 | 93.0% | 94.4% | 4 / 3 | 1.000 |
| Sensor Noise | 79 | 46.8% | 57.0% | 9 / 1 | 0.021 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 6418/40000 | 8.36 s/step | ETA 78:01:35 | loss 0.1180 |
| b0 | 1339/40000 | 8.97 s/step | ETA 96:18:41 | loss 0.2659 |
| b2 | 1759/40000 | 9.16 s/step | ETA 97:18:03 | loss 0.1912 |

---
## 2026-10-27 15:32:17

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 528/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=528)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 528 | 74.8% | 78.8% | 74.6% | 41 / 20 | 0.010 |
| libero_goal | 35 | 68.6% | 74.3% | 77.1% | 4 / 2 | 0.688 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 88 | 67.0% | 77.3% | 75.0% | 15 / 6 | 0.078 |
| Camera Viewpoints | 75 | 33.3% | 38.7% | 22.7% | 7 / 3 | 0.344 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 79 | 51.9% | 57.0% | 46.8% | 8 / 4 | 0.388 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=657)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 657 | 70.0% | 74.6% | 67.9% | 58 / 28 | 0.002 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_goal | 35 | 68.6% | 74.3% | 77.1% | 4 / 2 | 0.688 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 129 | 58.1% | 70.5% | 62.0% | 25 / 9 | 0.009 |
| Camera Viewpoints | 123 | 38.2% | 44.7% | 26.0% | 13 / 5 | 0.096 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 71 | 85.9% | 94.4% | 93.0% | 8 / 2 | 0.109 |
| Sensor Noise | 119 | 57.1% | 58.8% | 50.4% | 9 / 7 | 0.804 |

### StructWAM 40k vs 24k (full eval tasks)  (n=528)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 528 | 74.6% | 78.8% | 36 / 14 | 0.003 |
| libero_goal | 35 | 77.1% | 74.3% | 2 / 3 | 1.000 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 88 | 75.0% | 77.3% | 8 / 6 | 0.791 |
| Camera Viewpoints | 75 | 22.7% | 38.7% | 15 / 3 | 0.008 |
| Language Instructions | 69 | 97.1% | 97.1% | 0 / 0 | 1.000 |
| Light Conditions | 67 | 98.5% | 98.5% | 0 / 0 | 1.000 |
| Objects Layout | 79 | 94.9% | 93.7% | 0 / 1 | 1.000 |
| Robot Initial States | 71 | 93.0% | 94.4% | 4 / 3 | 1.000 |
| Sensor Noise | 79 | 46.8% | 57.0% | 9 / 1 | 0.021 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 6488/40000 | 8.53 s/step | ETA 79:25:22 | loss 0.1417 |
| b0 | 1407/40000 | 8.91 s/step | ETA 95:32:46 | loss 0.2232 |
| b2 | 1826/40000 | 9.10 s/step | ETA 96:27:24 | loss 0.2092 |

---
## 2026-10-27 15:42:23

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 548/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=548)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 548 | 74.3% | 79.0% | 74.5% | 47 / 21 | 0.002 |
| libero_goal | 55 | 65.5% | 78.2% | 74.5% | 10 / 3 | 0.092 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 93 | 65.6% | 75.3% | 72.0% | 16 / 7 | 0.093 |
| Camera Viewpoints | 75 | 33.3% | 38.7% | 22.7% | 7 / 3 | 0.344 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 86 | 82.6% | 95.3% | 91.9% | 13 / 2 | 0.007 |
| Sensor Noise | 79 | 51.9% | 57.0% | 46.8% | 8 / 4 | 0.388 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=677)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 677 | 69.7% | 74.9% | 67.9% | 64 / 29 | 0.000 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_goal | 55 | 65.5% | 78.2% | 74.5% | 10 / 3 | 0.092 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 123 | 38.2% | 44.7% | 26.0% | 13 / 5 | 0.096 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 86 | 82.6% | 95.3% | 91.9% | 13 / 2 | 0.007 |
| Sensor Noise | 119 | 57.1% | 58.8% | 50.4% | 9 / 7 | 0.804 |

### StructWAM 40k vs 24k (full eval tasks)  (n=548)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 548 | 74.5% | 79.0% | 39 / 14 | 0.001 |
| libero_goal | 55 | 74.5% | 78.2% | 5 / 3 | 0.727 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 93 | 72.0% | 75.3% | 9 / 6 | 0.607 |
| Camera Viewpoints | 75 | 22.7% | 38.7% | 15 / 3 | 0.008 |
| Language Instructions | 69 | 97.1% | 97.1% | 0 / 0 | 1.000 |
| Light Conditions | 67 | 98.5% | 98.5% | 0 / 0 | 1.000 |
| Objects Layout | 79 | 94.9% | 93.7% | 0 / 1 | 1.000 |
| Robot Initial States | 86 | 91.9% | 95.3% | 6 / 3 | 0.508 |
| Sensor Noise | 79 | 46.8% | 57.0% | 9 / 1 | 0.021 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 6561/40000 | 8.24 s/step | ETA 76:30:39 | loss 0.1312 |
| b0 | 1473/40000 | 9.12 s/step | ETA 97:33:41 | loss 0.2344 |
| b2 | 1894/40000 | 8.88 s/step | ETA 94:01:03 | loss 0.1831 |

---
## 2026-10-27 15:52:29

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 577/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=577)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 577 | 74.4% | 79.9% | 75.0% | 53 / 21 | 0.000 |
| libero_goal | 84 | 69.0% | 84.5% | 78.6% | 16 / 3 | 0.004 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 93 | 65.6% | 75.3% | 72.0% | 16 / 7 | 0.093 |
| Camera Viewpoints | 79 | 34.2% | 41.8% | 24.1% | 9 / 3 | 0.146 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 79 | 51.9% | 57.0% | 46.8% | 8 / 4 | 0.388 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=706)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 706 | 70.0% | 75.8% | 68.7% | 70 / 29 | 0.000 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_goal | 84 | 69.0% | 84.5% | 78.6% | 16 / 3 | 0.004 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 127 | 38.6% | 46.5% | 26.8% | 15 / 5 | 0.041 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 119 | 57.1% | 58.8% | 50.4% | 9 / 7 | 0.804 |

### StructWAM 40k vs 24k (full eval tasks)  (n=577)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 577 | 75.0% | 79.9% | 42 / 14 | 0.000 |
| libero_goal | 84 | 78.6% | 84.5% | 8 / 3 | 0.227 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 93 | 72.0% | 75.3% | 9 / 6 | 0.607 |
| Camera Viewpoints | 79 | 24.1% | 41.8% | 17 / 3 | 0.003 |
| Language Instructions | 69 | 97.1% | 97.1% | 0 / 0 | 1.000 |
| Light Conditions | 67 | 98.5% | 98.5% | 0 / 0 | 1.000 |
| Objects Layout | 79 | 94.9% | 93.7% | 0 / 1 | 1.000 |
| Robot Initial States | 111 | 91.9% | 95.5% | 7 / 3 | 0.344 |
| Sensor Noise | 79 | 46.8% | 57.0% | 9 / 1 | 0.021 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 6632/40000 | 8.31 s/step | ETA 77:00:47 | loss 0.1729 |
| b0 | 1540/40000 | 8.83 s/step | ETA 94:21:37 | loss 0.2175 |
| b2 | 1960/40000 | 8.99 s/step | ETA 94:59:30 | loss 0.1725 |

---
## 2026-10-27 16:02:34

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 585/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=585)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 585 | 74.0% | 79.7% | 74.9% | 56 / 23 | 0.000 |
| libero_goal | 92 | 67.4% | 82.6% | 77.2% | 19 / 5 | 0.007 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 93 | 65.6% | 75.3% | 72.0% | 16 / 7 | 0.093 |
| Camera Viewpoints | 87 | 35.6% | 43.7% | 27.6% | 12 / 5 | 0.143 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 79 | 51.9% | 57.0% | 46.8% | 8 / 4 | 0.388 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=714)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 714 | 69.7% | 75.6% | 68.6% | 73 / 31 | 0.000 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_goal | 92 | 67.4% | 82.6% | 77.2% | 19 / 5 | 0.007 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 135 | 39.3% | 47.4% | 28.9% | 18 / 7 | 0.043 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 119 | 57.1% | 58.8% | 50.4% | 9 / 7 | 0.804 |

### StructWAM 40k vs 24k (full eval tasks)  (n=585)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 585 | 74.9% | 79.7% | 43 / 15 | 0.000 |
| libero_goal | 92 | 77.2% | 82.6% | 9 / 4 | 0.267 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 93 | 72.0% | 75.3% | 9 / 6 | 0.607 |
| Camera Viewpoints | 87 | 27.6% | 43.7% | 18 / 4 | 0.004 |
| Language Instructions | 69 | 97.1% | 97.1% | 0 / 0 | 1.000 |
| Light Conditions | 67 | 98.5% | 98.5% | 0 / 0 | 1.000 |
| Objects Layout | 79 | 94.9% | 93.7% | 0 / 1 | 1.000 |
| Robot Initial States | 111 | 91.9% | 95.5% | 7 / 3 | 0.344 |
| Sensor Noise | 79 | 46.8% | 57.0% | 9 / 1 | 0.021 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 6704/40000 | 8.40 s/step | ETA 77:42:44 | loss 0.1193 |
| b0 | 1608/40000 | 8.88 s/step | ETA 94:44:38 | loss 0.2402 |
| b2 | 2005/40000 | 23.62 s/step | ETA 249:16:57 | loss 0.1726 |

---
## 2026-10-27 16:12:40

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 593/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=593)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 593 | 73.2% | 78.9% | 74.2% | 57 / 23 | 0.000 |
| libero_goal | 100 | 63.0% | 78.0% | 73.0% | 20 / 5 | 0.004 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 93 | 65.6% | 75.3% | 72.0% | 16 / 7 | 0.093 |
| Camera Viewpoints | 95 | 33.7% | 42.1% | 27.4% | 13 / 5 | 0.096 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 79 | 51.9% | 57.0% | 46.8% | 8 / 4 | 0.388 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=722)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 722 | 69.1% | 75.1% | 68.1% | 74 / 31 | 0.000 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_goal | 100 | 63.0% | 78.0% | 73.0% | 20 / 5 | 0.004 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 143 | 37.8% | 46.2% | 28.7% | 19 / 7 | 0.029 |
| Language Instructions | 69 | 98.6% | 97.1% | 97.1% | 1 / 2 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 119 | 57.1% | 58.8% | 50.4% | 9 / 7 | 0.804 |

### StructWAM 40k vs 24k (full eval tasks)  (n=593)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 593 | 74.2% | 78.9% | 43 / 15 | 0.000 |
| libero_goal | 100 | 73.0% | 78.0% | 9 / 4 | 0.267 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 93 | 72.0% | 75.3% | 9 / 6 | 0.607 |
| Camera Viewpoints | 95 | 27.4% | 42.1% | 18 / 4 | 0.004 |
| Language Instructions | 69 | 97.1% | 97.1% | 0 / 0 | 1.000 |
| Light Conditions | 67 | 98.5% | 98.5% | 0 / 0 | 1.000 |
| Objects Layout | 79 | 94.9% | 93.7% | 0 / 1 | 1.000 |
| Robot Initial States | 111 | 91.9% | 95.5% | 7 / 3 | 0.344 |
| Sensor Noise | 79 | 46.8% | 57.0% | 9 / 1 | 0.021 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 6777/40000 | 8.43 s/step | ETA 77:46:17 | loss 0.1483 |
| b0 | 1675/40000 | 9.06 s/step | ETA 96:29:57 | loss 0.1936 |
| b2 | 2073/40000 | 8.96 s/step | ETA 94:25:34 | loss 0.1722 |

---
## 2026-10-27 16:22:45

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 609/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=609)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 609 | 72.7% | 78.2% | 73.6% | 57 / 24 | 0.000 |
| libero_goal | 116 | 62.1% | 74.1% | 69.8% | 20 / 6 | 0.009 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 93 | 65.6% | 75.3% | 72.0% | 16 / 7 | 0.093 |
| Camera Viewpoints | 105 | 33.3% | 40.0% | 26.7% | 13 / 6 | 0.167 |
| Language Instructions | 75 | 98.7% | 97.3% | 97.3% | 1 / 2 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 79 | 51.9% | 57.0% | 46.8% | 8 / 4 | 0.388 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=738)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 738 | 68.8% | 74.5% | 67.8% | 74 / 32 | 0.000 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_goal | 116 | 62.1% | 74.1% | 69.8% | 20 / 6 | 0.009 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 153 | 37.3% | 44.4% | 28.1% | 19 / 8 | 0.052 |
| Language Instructions | 75 | 98.7% | 97.3% | 97.3% | 1 / 2 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 119 | 57.1% | 58.8% | 50.4% | 9 / 7 | 0.804 |

### StructWAM 40k vs 24k (full eval tasks)  (n=609)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 609 | 73.6% | 78.2% | 43 / 15 | 0.000 |
| libero_goal | 116 | 69.8% | 74.1% | 9 / 4 | 0.267 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 93 | 72.0% | 75.3% | 9 / 6 | 0.607 |
| Camera Viewpoints | 105 | 26.7% | 40.0% | 18 / 4 | 0.004 |
| Language Instructions | 75 | 97.3% | 97.3% | 0 / 0 | 1.000 |
| Light Conditions | 67 | 98.5% | 98.5% | 0 / 0 | 1.000 |
| Objects Layout | 79 | 94.9% | 93.7% | 0 / 1 | 1.000 |
| Robot Initial States | 111 | 91.9% | 95.5% | 7 / 3 | 0.344 |
| Sensor Noise | 79 | 46.8% | 57.0% | 9 / 1 | 0.021 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 6849/40000 | 8.52 s/step | ETA 78:26:16 | loss 0.1399 |
| b0 | 1742/40000 | 9.10 s/step | ETA 96:44:15 | loss 0.2427 |
| b2 | 2140/40000 | 8.85 s/step | ETA 93:02:00 | loss 0.1568 |

---
## 2026-10-27 16:32:51

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 634/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=634)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 634 | 73.0% | 78.4% | 73.7% | 60 / 26 | 0.000 |
| libero_goal | 141 | 65.2% | 75.9% | 70.9% | 23 / 8 | 0.011 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 93 | 65.6% | 75.3% | 72.0% | 16 / 7 | 0.093 |
| Camera Viewpoints | 109 | 33.9% | 38.5% | 25.7% | 13 / 8 | 0.383 |
| Language Instructions | 96 | 95.8% | 97.9% | 95.8% | 4 / 2 | 0.688 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 79 | 51.9% | 57.0% | 46.8% | 8 / 4 | 0.388 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=763)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 763 | 69.2% | 74.8% | 68.0% | 77 / 34 | 0.000 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_goal | 141 | 65.2% | 75.9% | 70.9% | 23 / 8 | 0.011 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 96 | 95.8% | 97.9% | 95.8% | 4 / 2 | 0.688 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 119 | 57.1% | 58.8% | 50.4% | 9 / 7 | 0.804 |

### StructWAM 40k vs 24k (full eval tasks)  (n=634)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 634 | 73.7% | 78.4% | 45 / 15 | 0.000 |
| libero_goal | 141 | 70.9% | 75.9% | 11 / 4 | 0.118 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 93 | 72.0% | 75.3% | 9 / 6 | 0.607 |
| Camera Viewpoints | 109 | 25.7% | 38.5% | 18 / 4 | 0.004 |
| Language Instructions | 96 | 95.8% | 97.9% | 2 / 0 | 0.500 |
| Light Conditions | 67 | 98.5% | 98.5% | 0 / 0 | 1.000 |
| Objects Layout | 79 | 94.9% | 93.7% | 0 / 1 | 1.000 |
| Robot Initial States | 111 | 91.9% | 95.5% | 7 / 3 | 0.344 |
| Sensor Noise | 79 | 46.8% | 57.0% | 9 / 1 | 0.021 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 6920/40000 | 8.43 s/step | ETA 77:28:40 | loss 0.1506 |
| b0 | 1809/40000 | 8.94 s/step | ETA 94:51:45 | loss 0.1944 |
| b2 | 2208/40000 | 8.98 s/step | ETA 94:17:42 | loss 0.1817 |

---
## 2026-10-27 16:42:57

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 653/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=653)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 653 | 73.5% | 78.6% | 74.3% | 60 / 27 | 0.001 |
| libero_goal | 160 | 68.1% | 76.9% | 73.8% | 23 / 9 | 0.020 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 93 | 65.6% | 75.3% | 72.0% | 16 / 7 | 0.093 |
| Camera Viewpoints | 109 | 33.9% | 38.5% | 25.7% | 13 / 8 | 0.383 |
| Language Instructions | 111 | 94.6% | 95.5% | 95.5% | 4 / 3 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 83 | 54.2% | 59.0% | 49.4% | 8 / 4 | 0.388 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=782)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 782 | 69.7% | 75.1% | 68.7% | 77 / 35 | 0.000 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_goal | 160 | 68.1% | 76.9% | 73.8% | 23 / 9 | 0.020 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 111 | 94.6% | 95.5% | 95.5% | 4 / 3 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 123 | 58.5% | 60.2% | 52.0% | 9 / 7 | 0.804 |

### StructWAM 40k vs 24k (full eval tasks)  (n=653)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 653 | 74.3% | 78.6% | 45 / 17 | 0.000 |
| libero_goal | 160 | 73.8% | 76.9% | 11 / 6 | 0.332 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 93 | 72.0% | 75.3% | 9 / 6 | 0.607 |
| Camera Viewpoints | 109 | 25.7% | 38.5% | 18 / 4 | 0.004 |
| Language Instructions | 111 | 95.5% | 95.5% | 2 / 2 | 1.000 |
| Light Conditions | 67 | 98.5% | 98.5% | 0 / 0 | 1.000 |
| Objects Layout | 79 | 94.9% | 93.7% | 0 / 1 | 1.000 |
| Robot Initial States | 111 | 91.9% | 95.5% | 7 / 3 | 0.344 |
| Sensor Noise | 83 | 49.4% | 59.0% | 9 / 1 | 0.021 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 6992/40000 | 8.90 s/step | ETA 81:37:37 | loss 0.1261 |
| b0 | 1876/40000 | 9.23 s/step | ETA 97:45:26 | loss 0.2270 |
| b2 | 2274/40000 | 8.98 s/step | ETA 94:07:59 | loss 0.1993 |

---
## 2026-10-27 16:53:03

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 660/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=660)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 660 | 73.2% | 78.2% | 73.9% | 60 / 27 | 0.001 |
| libero_goal | 167 | 67.1% | 75.4% | 72.5% | 23 / 9 | 0.020 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 93 | 65.6% | 75.3% | 72.0% | 16 / 7 | 0.093 |
| Camera Viewpoints | 109 | 33.9% | 38.5% | 25.7% | 13 / 8 | 0.383 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 89 | 53.9% | 58.4% | 49.4% | 8 / 4 | 0.388 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=789)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 789 | 69.5% | 74.8% | 68.4% | 77 / 35 | 0.000 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_goal | 167 | 67.1% | 75.4% | 72.5% | 23 / 9 | 0.020 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 129 | 58.1% | 59.7% | 51.9% | 9 / 7 | 0.804 |

### StructWAM 40k vs 24k (full eval tasks)  (n=660)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 660 | 73.9% | 78.2% | 45 / 17 | 0.000 |
| libero_goal | 167 | 72.5% | 75.4% | 11 / 6 | 0.332 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 93 | 72.0% | 75.3% | 9 / 6 | 0.607 |
| Camera Viewpoints | 109 | 25.7% | 38.5% | 18 / 4 | 0.004 |
| Language Instructions | 112 | 94.6% | 94.6% | 2 / 2 | 1.000 |
| Light Conditions | 67 | 98.5% | 98.5% | 0 / 0 | 1.000 |
| Objects Layout | 79 | 94.9% | 93.7% | 0 / 1 | 1.000 |
| Robot Initial States | 111 | 91.9% | 95.5% | 7 / 3 | 0.344 |
| Sensor Noise | 89 | 49.4% | 58.4% | 9 / 1 | 0.021 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 7063/40000 | 8.51 s/step | ETA 77:51:29 | loss 0.1315 |
| b0 | 1942/40000 | 8.93 s/step | ETA 94:26:36 | loss 0.2137 |
| b2 | 2341/40000 | 8.87 s/step | ETA 92:44:11 | loss 0.1856 |

---
## 2026-10-27 17:03:09

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 666/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=666)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 666 | 73.0% | 77.9% | 73.7% | 60 / 27 | 0.001 |
| libero_goal | 173 | 66.5% | 74.6% | 71.7% | 23 / 9 | 0.020 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 93 | 65.6% | 75.3% | 72.0% | 16 / 7 | 0.093 |
| Camera Viewpoints | 109 | 33.9% | 38.5% | 25.7% | 13 / 8 | 0.383 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 95 | 53.7% | 57.9% | 49.5% | 8 / 4 | 0.388 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=795)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 795 | 69.3% | 74.6% | 68.3% | 77 / 35 | 0.000 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_goal | 173 | 66.5% | 74.6% | 71.7% | 23 / 9 | 0.020 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 135 | 57.8% | 59.3% | 51.9% | 9 / 7 | 0.804 |

### StructWAM 40k vs 24k (full eval tasks)  (n=666)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 666 | 73.7% | 77.9% | 45 / 17 | 0.000 |
| libero_goal | 173 | 71.7% | 74.6% | 11 / 6 | 0.332 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 93 | 72.0% | 75.3% | 9 / 6 | 0.607 |
| Camera Viewpoints | 109 | 25.7% | 38.5% | 18 / 4 | 0.004 |
| Language Instructions | 112 | 94.6% | 94.6% | 2 / 2 | 1.000 |
| Light Conditions | 67 | 98.5% | 98.5% | 0 / 0 | 1.000 |
| Objects Layout | 79 | 94.9% | 93.7% | 0 / 1 | 1.000 |
| Robot Initial States | 111 | 91.9% | 95.5% | 7 / 3 | 0.344 |
| Sensor Noise | 95 | 49.5% | 57.9% | 9 / 1 | 0.021 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 7135/40000 | 8.59 s/step | ETA 78:23:00 | loss 0.1401 |
| b0 | 2000/40000 | 9.02 s/step | ETA 95:10:21 | loss 0.1975 |
| b2 | 2409/40000 | 8.94 s/step | ETA 93:22:11 | loss 0.1830 |

---
## 2026-10-27 17:13:15

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 672/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=672)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 672 | 72.6% | 77.8% | 73.5% | 62 / 27 | 0.000 |
| libero_goal | 179 | 65.4% | 74.3% | 70.9% | 25 / 9 | 0.009 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 93 | 65.6% | 75.3% | 72.0% | 16 / 7 | 0.093 |
| Camera Viewpoints | 109 | 33.9% | 38.5% | 25.7% | 13 / 8 | 0.383 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 101 | 52.5% | 58.4% | 49.5% | 10 / 4 | 0.180 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=801)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 801 | 69.0% | 74.5% | 68.2% | 79 / 35 | 0.000 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_goal | 179 | 65.4% | 74.3% | 70.9% | 25 / 9 | 0.009 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 141 | 56.7% | 59.6% | 51.8% | 11 / 7 | 0.481 |

### StructWAM 40k vs 24k (full eval tasks)  (n=672)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 672 | 73.5% | 77.8% | 46 / 17 | 0.000 |
| libero_goal | 179 | 70.9% | 74.3% | 12 / 6 | 0.238 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 93 | 72.0% | 75.3% | 9 / 6 | 0.607 |
| Camera Viewpoints | 109 | 25.7% | 38.5% | 18 / 4 | 0.004 |
| Language Instructions | 112 | 94.6% | 94.6% | 2 / 2 | 1.000 |
| Light Conditions | 67 | 98.5% | 98.5% | 0 / 0 | 1.000 |
| Objects Layout | 79 | 94.9% | 93.7% | 0 / 1 | 1.000 |
| Robot Initial States | 111 | 91.9% | 95.5% | 7 / 3 | 0.344 |
| Sensor Noise | 101 | 49.5% | 58.4% | 10 / 1 | 0.012 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 7207/40000 | 8.46 s/step | ETA 77:02:49 | loss 0.1417 |
| b0 | 2053/40000 | 9.32 s/step | ETA 98:16:04 | loss 0.2148 |
| b2 | 2475/40000 | 9.07 s/step | ETA 94:32:31 | loss 0.1552 |

---
## 2026-10-27 17:23:21

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 679/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=679)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 679 | 72.5% | 77.6% | 73.0% | 64 / 29 | 0.000 |
| libero_goal | 186 | 65.1% | 73.7% | 69.4% | 27 / 11 | 0.014 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 93 | 65.6% | 75.3% | 72.0% | 16 / 7 | 0.093 |
| Camera Viewpoints | 109 | 33.9% | 38.5% | 25.7% | 13 / 8 | 0.383 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 108 | 52.8% | 58.3% | 48.1% | 12 / 6 | 0.238 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=808)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 808 | 68.9% | 74.4% | 67.8% | 81 / 37 | 0.000 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_goal | 186 | 65.1% | 73.7% | 69.4% | 27 / 11 | 0.014 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 79 | 93.7% | 93.7% | 94.9% | 2 / 2 | 1.000 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 148 | 56.8% | 59.5% | 50.7% | 13 / 9 | 0.523 |

### StructWAM 40k vs 24k (full eval tasks)  (n=679)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 679 | 73.0% | 77.6% | 48 / 17 | 0.000 |
| libero_goal | 186 | 69.4% | 73.7% | 14 / 6 | 0.115 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 93 | 72.0% | 75.3% | 9 / 6 | 0.607 |
| Camera Viewpoints | 109 | 25.7% | 38.5% | 18 / 4 | 0.004 |
| Language Instructions | 112 | 94.6% | 94.6% | 2 / 2 | 1.000 |
| Light Conditions | 67 | 98.5% | 98.5% | 0 / 0 | 1.000 |
| Objects Layout | 79 | 94.9% | 93.7% | 0 / 1 | 1.000 |
| Robot Initial States | 111 | 91.9% | 95.5% | 7 / 3 | 0.344 |
| Sensor Noise | 108 | 48.1% | 58.3% | 12 / 1 | 0.003 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 7279/40000 | 8.60 s/step | ETA 78:12:05 | loss 0.1368 |
| b0 | 2120/40000 | 9.13 s/step | ETA 96:01:07 | loss 0.1968 |
| b2 | 2543/40000 | 8.84 s/step | ETA 91:58:01 | loss 0.2021 |

---
## 2026-10-27 17:33:27

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 688/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=688)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 688 | 72.8% | 77.9% | 73.3% | 64 / 29 | 0.000 |
| libero_goal | 195 | 66.7% | 74.9% | 70.3% | 27 / 11 | 0.014 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 93 | 65.6% | 75.3% | 72.0% | 16 / 7 | 0.093 |
| Camera Viewpoints | 109 | 33.9% | 38.5% | 25.7% | 13 / 8 | 0.383 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 80 | 93.8% | 93.8% | 95.0% | 2 / 2 | 1.000 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 116 | 56.0% | 61.2% | 50.9% | 12 / 6 | 0.238 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=817)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 817 | 69.3% | 74.7% | 68.1% | 81 / 37 | 0.000 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_goal | 195 | 66.7% | 74.9% | 70.3% | 27 / 11 | 0.014 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 80 | 93.8% | 93.8% | 95.0% | 2 / 2 | 1.000 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 156 | 59.0% | 61.5% | 52.6% | 13 / 9 | 0.523 |

### StructWAM 40k vs 24k (full eval tasks)  (n=688)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 688 | 73.3% | 77.9% | 49 / 17 | 0.000 |
| libero_goal | 195 | 70.3% | 74.9% | 15 / 6 | 0.078 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 93 | 72.0% | 75.3% | 9 / 6 | 0.607 |
| Camera Viewpoints | 109 | 25.7% | 38.5% | 18 / 4 | 0.004 |
| Language Instructions | 112 | 94.6% | 94.6% | 2 / 2 | 1.000 |
| Light Conditions | 67 | 98.5% | 98.5% | 0 / 0 | 1.000 |
| Objects Layout | 80 | 95.0% | 93.8% | 0 / 1 | 1.000 |
| Robot Initial States | 111 | 91.9% | 95.5% | 7 / 3 | 0.344 |
| Sensor Noise | 116 | 50.9% | 61.2% | 13 / 1 | 0.002 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 7350/40000 | 8.43 s/step | ETA 76:26:09 | loss 0.1166 |
| b0 | 2186/40000 | 8.88 s/step | ETA 93:18:52 | loss 0.2146 |
| b2 | 2609/40000 | 9.10 s/step | ETA 94:33:52 | loss 0.1981 |

---
## 2026-10-27 17:43:33

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 699/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=699)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 699 | 72.8% | 77.8% | 73.2% | 64 / 29 | 0.000 |
| libero_goal | 206 | 67.0% | 74.8% | 70.4% | 27 / 11 | 0.014 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 93 | 65.6% | 75.3% | 72.0% | 16 / 7 | 0.093 |
| Camera Viewpoints | 109 | 33.9% | 38.5% | 25.7% | 13 / 8 | 0.383 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 87 | 94.3% | 94.3% | 94.3% | 2 / 2 | 1.000 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 120 | 55.0% | 60.0% | 50.8% | 12 / 6 | 0.238 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=828)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 828 | 69.3% | 74.6% | 68.1% | 81 / 37 | 0.000 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_goal | 206 | 67.0% | 74.8% | 70.4% | 27 / 11 | 0.014 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 87 | 94.3% | 94.3% | 94.3% | 2 / 2 | 1.000 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 160 | 58.1% | 60.6% | 52.5% | 13 / 9 | 0.523 |

### StructWAM 40k vs 24k (full eval tasks)  (n=699)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 699 | 73.2% | 77.8% | 50 / 18 | 0.000 |
| libero_goal | 206 | 70.4% | 74.8% | 16 / 7 | 0.093 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 93 | 72.0% | 75.3% | 9 / 6 | 0.607 |
| Camera Viewpoints | 109 | 25.7% | 38.5% | 18 / 4 | 0.004 |
| Language Instructions | 112 | 94.6% | 94.6% | 2 / 2 | 1.000 |
| Light Conditions | 67 | 98.5% | 98.5% | 0 / 0 | 1.000 |
| Objects Layout | 87 | 94.3% | 94.3% | 1 / 1 | 1.000 |
| Robot Initial States | 111 | 91.9% | 95.5% | 7 / 3 | 0.344 |
| Sensor Noise | 120 | 50.8% | 60.0% | 13 / 2 | 0.007 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 7422/40000 | 8.47 s/step | ETA 76:39:57 | loss 0.1376 |
| b0 | 2254/40000 | 9.12 s/step | ETA 95:37:49 | loss 0.2217 |
| b2 | 2676/40000 | 8.94 s/step | ETA 92:42:01 | loss 0.1858 |

---
## 2026-10-27 17:53:38

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 708/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=708)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 708 | 72.5% | 77.5% | 72.9% | 65 / 29 | 0.000 |
| libero_goal | 215 | 66.0% | 74.0% | 69.3% | 28 / 11 | 0.009 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 93 | 65.6% | 75.3% | 72.0% | 16 / 7 | 0.093 |
| Camera Viewpoints | 109 | 33.9% | 38.5% | 25.7% | 13 / 8 | 0.383 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 95 | 90.5% | 91.6% | 90.5% | 3 / 2 | 1.000 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 121 | 54.5% | 59.5% | 50.4% | 12 / 6 | 0.238 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=837)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 837 | 69.1% | 74.4% | 67.9% | 82 / 37 | 0.000 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_goal | 215 | 66.0% | 74.0% | 69.3% | 28 / 11 | 0.009 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 95 | 90.5% | 91.6% | 90.5% | 3 / 2 | 1.000 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 161 | 57.8% | 60.2% | 52.2% | 13 / 9 | 0.523 |

### StructWAM 40k vs 24k (full eval tasks)  (n=708)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 708 | 72.9% | 77.5% | 51 / 18 | 0.000 |
| libero_goal | 215 | 69.3% | 74.0% | 17 / 7 | 0.064 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 93 | 72.0% | 75.3% | 9 / 6 | 0.607 |
| Camera Viewpoints | 109 | 25.7% | 38.5% | 18 / 4 | 0.004 |
| Language Instructions | 112 | 94.6% | 94.6% | 2 / 2 | 1.000 |
| Light Conditions | 67 | 98.5% | 98.5% | 0 / 0 | 1.000 |
| Objects Layout | 95 | 90.5% | 91.6% | 2 / 1 | 1.000 |
| Robot Initial States | 111 | 91.9% | 95.5% | 7 / 3 | 0.344 |
| Sensor Noise | 121 | 50.4% | 59.5% | 13 / 2 | 0.007 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 7493/40000 | 8.62 s/step | ETA 77:52:38 | loss 0.1484 |
| b0 | 2321/40000 | 8.84 s/step | ETA 92:30:34 | loss 0.2240 |
| b2 | 2742/40000 | 9.11 s/step | ETA 94:17:00 | loss 0.1638 |

---
## 2026-10-27 18:03:44

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 726/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=726)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 726 | 71.9% | 77.0% | 72.3% | 66 / 29 | 0.000 |
| libero_goal | 233 | 64.8% | 72.5% | 67.8% | 29 / 11 | 0.006 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 93 | 65.6% | 75.3% | 72.0% | 16 / 7 | 0.093 |
| Camera Viewpoints | 109 | 33.9% | 38.5% | 25.7% | 13 / 8 | 0.383 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 112 | 84.8% | 86.6% | 84.8% | 4 / 2 | 0.688 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 122 | 54.1% | 59.0% | 50.0% | 12 / 6 | 0.238 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=855)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 855 | 68.7% | 74.0% | 67.5% | 83 / 37 | 0.000 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_goal | 233 | 64.8% | 72.5% | 67.8% | 29 / 11 | 0.006 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 67 | 100.0% | 98.5% | 98.5% | 0 / 1 | 1.000 |
| Objects Layout | 112 | 84.8% | 86.6% | 84.8% | 4 / 2 | 0.688 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 162 | 57.4% | 59.9% | 51.9% | 13 / 9 | 0.523 |

### StructWAM 40k vs 24k (full eval tasks)  (n=726)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 726 | 72.3% | 77.0% | 52 / 18 | 0.000 |
| libero_goal | 233 | 67.8% | 72.5% | 18 / 7 | 0.043 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 93 | 72.0% | 75.3% | 9 / 6 | 0.607 |
| Camera Viewpoints | 109 | 25.7% | 38.5% | 18 / 4 | 0.004 |
| Language Instructions | 112 | 94.6% | 94.6% | 2 / 2 | 1.000 |
| Light Conditions | 67 | 98.5% | 98.5% | 0 / 0 | 1.000 |
| Objects Layout | 112 | 84.8% | 86.6% | 3 / 1 | 0.625 |
| Robot Initial States | 111 | 91.9% | 95.5% | 7 / 3 | 0.344 |
| Sensor Noise | 122 | 50.0% | 59.0% | 13 / 2 | 0.007 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 7565/40000 | 8.46 s/step | ETA 76:13:14 | loss 0.1092 |
| b0 | 2387/40000 | 9.13 s/step | ETA 95:24:41 | loss 0.2441 |
| b2 | 2809/40000 | 8.99 s/step | ETA 92:55:07 | loss 0.1596 |

---
## 2026-10-27 18:13:50

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 748/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=748)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 748 | 72.2% | 77.4% | 72.9% | 68 / 29 | 0.000 |
| libero_goal | 255 | 66.3% | 74.1% | 69.8% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 93 | 65.6% | 75.3% | 72.0% | 16 / 7 | 0.093 |
| Camera Viewpoints | 109 | 33.9% | 38.5% | 25.7% | 13 / 8 | 0.383 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 87 | 97.7% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 122 | 54.1% | 59.0% | 50.0% | 12 / 6 | 0.238 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=877)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 877 | 69.0% | 74.5% | 68.1% | 85 / 37 | 0.000 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_goal | 255 | 66.3% | 74.1% | 69.8% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 87 | 97.7% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 162 | 57.4% | 59.9% | 51.9% | 13 / 9 | 0.523 |

### StructWAM 40k vs 24k (full eval tasks)  (n=748)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 748 | 72.9% | 77.4% | 52 / 18 | 0.000 |
| libero_goal | 255 | 69.8% | 74.1% | 18 / 7 | 0.043 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 93 | 72.0% | 75.3% | 9 / 6 | 0.607 |
| Camera Viewpoints | 109 | 25.7% | 38.5% | 18 / 4 | 0.004 |
| Language Instructions | 112 | 94.6% | 94.6% | 2 / 2 | 1.000 |
| Light Conditions | 87 | 98.9% | 98.9% | 0 / 0 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 3 / 1 | 0.625 |
| Robot Initial States | 111 | 91.9% | 95.5% | 7 / 3 | 0.344 |
| Sensor Noise | 122 | 50.0% | 59.0% | 13 / 2 | 0.007 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 7636/40000 | 8.61 s/step | ETA 77:24:19 | loss 0.1423 |
| b0 | 2454/40000 | 9.49 s/step | ETA 99:00:16 | loss 0.2130 |
| b2 | 2875/40000 | 9.06 s/step | ETA 93:24:09 | loss 0.2120 |

---
## 2026-10-27 18:23:56

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 762/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=762)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 762 | 72.4% | 77.8% | 73.2% | 70 / 29 | 0.000 |
| libero_10 | 9 | 77.8% | 100.0% | 88.9% | 2 / 0 | 0.500 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 102 | 66.7% | 77.5% | 73.5% | 18 / 7 | 0.043 |
| Camera Viewpoints | 109 | 33.9% | 38.5% | 25.7% | 13 / 8 | 0.383 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 122 | 54.1% | 59.0% | 50.0% | 12 / 6 | 0.238 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=882)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 882 | 69.2% | 74.6% | 68.3% | 85 / 37 | 0.000 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 162 | 57.4% | 59.9% | 51.9% | 13 / 9 | 0.523 |

### StructWAM 40k vs 24k (full eval tasks)  (n=762)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 762 | 73.2% | 77.8% | 53 / 18 | 0.000 |
| libero_10 | 9 | 88.9% | 100.0% | 1 / 0 | 1.000 |
| libero_goal | 260 | 70.4% | 74.6% | 18 / 7 | 0.043 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 102 | 73.5% | 77.5% | 10 / 6 | 0.454 |
| Camera Viewpoints | 109 | 25.7% | 38.5% | 18 / 4 | 0.004 |
| Language Instructions | 112 | 94.6% | 94.6% | 2 / 2 | 1.000 |
| Light Conditions | 92 | 98.9% | 98.9% | 0 / 0 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 3 / 1 | 0.625 |
| Robot Initial States | 111 | 91.9% | 95.5% | 7 / 3 | 0.344 |
| Sensor Noise | 122 | 50.0% | 59.0% | 13 / 2 | 0.007 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 7708/40000 | 8.52 s/step | ETA 76:23:12 | loss 0.1346 |
| b0 | 2521/40000 | 9.05 s/step | ETA 94:14:52 | loss 0.1837 |
| b2 | 2943/40000 | 8.90 s/step | ETA 91:33:47 | loss 0.1750 |

---
## 2026-10-27 18:34:03

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 773/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=773)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 773 | 71.8% | 77.5% | 72.4% | 73 / 29 | 0.000 |
| libero_10 | 20 | 50.0% | 75.0% | 50.0% | 5 / 0 | 0.062 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 113 | 62.8% | 75.2% | 68.1% | 21 / 7 | 0.013 |
| Camera Viewpoints | 109 | 33.9% | 38.5% | 25.7% | 13 / 8 | 0.383 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 122 | 54.1% | 59.0% | 50.0% | 12 / 6 | 0.238 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=882)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 882 | 69.2% | 74.6% | 68.3% | 85 / 37 | 0.000 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 162 | 57.4% | 59.9% | 51.9% | 13 / 9 | 0.523 |

### StructWAM 40k vs 24k (full eval tasks)  (n=773)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 773 | 72.4% | 77.5% | 57 / 18 | 0.000 |
| libero_10 | 20 | 50.0% | 75.0% | 5 / 0 | 0.062 |
| libero_goal | 260 | 70.4% | 74.6% | 18 / 7 | 0.043 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 113 | 68.1% | 75.2% | 14 / 6 | 0.115 |
| Camera Viewpoints | 109 | 25.7% | 38.5% | 18 / 4 | 0.004 |
| Language Instructions | 112 | 94.6% | 94.6% | 2 / 2 | 1.000 |
| Light Conditions | 92 | 98.9% | 98.9% | 0 / 0 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 3 / 1 | 0.625 |
| Robot Initial States | 111 | 91.9% | 95.5% | 7 / 3 | 0.344 |
| Sensor Noise | 122 | 50.0% | 59.0% | 13 / 2 | 0.007 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 7780/40000 | 8.34 s/step | ETA 74:40:31 | loss 0.1246 |
| b0 | 2588/40000 | 8.82 s/step | ETA 91:41:56 | loss 0.2196 |
| b2 | 3009/40000 | 8.98 s/step | ETA 92:13:52 | loss 0.1680 |

---
## 2026-10-27 18:44:09

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 786/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=786)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 786 | 71.0% | 77.2% | 71.6% | 78 / 29 | 0.000 |
| libero_10 | 33 | 39.4% | 69.7% | 39.4% | 10 / 0 | 0.002 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 126 | 58.7% | 73.8% | 63.5% | 26 / 7 | 0.001 |
| Camera Viewpoints | 109 | 33.9% | 38.5% | 25.7% | 13 / 8 | 0.383 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 122 | 54.1% | 59.0% | 50.0% | 12 / 6 | 0.238 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=882)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 882 | 69.2% | 74.6% | 68.3% | 85 / 37 | 0.000 |
| libero_10 | 129 | 50.4% | 57.4% | 40.3% | 17 / 8 | 0.108 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 111 | 82.0% | 95.5% | 91.9% | 17 / 2 | 0.001 |
| Sensor Noise | 162 | 57.4% | 59.9% | 51.9% | 13 / 9 | 0.523 |

### StructWAM 40k vs 24k (full eval tasks)  (n=786)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 786 | 71.6% | 77.2% | 62 / 18 | 0.000 |
| libero_10 | 33 | 39.4% | 69.7% | 10 / 0 | 0.002 |
| libero_goal | 260 | 70.4% | 74.6% | 18 / 7 | 0.043 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 126 | 63.5% | 73.8% | 19 / 6 | 0.015 |
| Camera Viewpoints | 109 | 25.7% | 38.5% | 18 / 4 | 0.004 |
| Language Instructions | 112 | 94.6% | 94.6% | 2 / 2 | 1.000 |
| Light Conditions | 92 | 98.9% | 98.9% | 0 / 0 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 3 / 1 | 0.625 |
| Robot Initial States | 111 | 91.9% | 95.5% | 7 / 3 | 0.344 |
| Sensor Noise | 122 | 50.0% | 59.0% | 13 / 2 | 0.007 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 7850/40000 | 8.31 s/step | ETA 74:11:58 | loss 0.1159 |
| b0 | 2654/40000 | 9.34 s/step | ETA 96:56:33 | loss 0.2242 |
| b2 | 3077/40000 | 8.88 s/step | ETA 91:03:59 | loss 0.1585 |

---
## 2026-10-27 18:54:15

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 800/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=800)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 800 | 70.9% | 76.6% | 71.2% | 78 / 32 | 0.000 |
| libero_10 | 47 | 46.8% | 61.7% | 42.6% | 10 / 3 | 0.092 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 109 | 33.9% | 38.5% | 25.7% | 13 / 8 | 0.383 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 117 | 82.9% | 95.7% | 92.3% | 17 / 2 | 0.001 |
| Sensor Noise | 122 | 54.1% | 59.0% | 50.0% | 12 / 6 | 0.238 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=888)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 888 | 69.4% | 74.8% | 68.5% | 85 / 37 | 0.000 |
| libero_10 | 135 | 52.6% | 59.3% | 43.0% | 17 / 8 | 0.108 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 117 | 82.9% | 95.7% | 92.3% | 17 / 2 | 0.001 |
| Sensor Noise | 162 | 57.4% | 59.9% | 51.9% | 13 / 9 | 0.523 |

### StructWAM 40k vs 24k (full eval tasks)  (n=800)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 800 | 71.2% | 76.6% | 62 / 19 | 0.000 |
| libero_10 | 47 | 42.6% | 61.7% | 10 / 1 | 0.012 |
| libero_goal | 260 | 70.4% | 74.6% | 18 / 7 | 0.043 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 134 | 60.4% | 69.4% | 19 / 7 | 0.029 |
| Camera Viewpoints | 109 | 25.7% | 38.5% | 18 / 4 | 0.004 |
| Language Instructions | 112 | 94.6% | 94.6% | 2 / 2 | 1.000 |
| Light Conditions | 92 | 98.9% | 98.9% | 0 / 0 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 3 / 1 | 0.625 |
| Robot Initial States | 117 | 92.3% | 95.7% | 7 / 3 | 0.344 |
| Sensor Noise | 122 | 50.0% | 59.0% | 13 / 2 | 0.007 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 7923/40000 | 8.21 s/step | ETA 73:09:44 | loss 0.1224 |
| b0 | 2722/40000 | 8.91 s/step | ETA 92:18:06 | loss 0.1928 |
| b2 | 3144/40000 | 8.89 s/step | ETA 90:59:23 | loss 0.1707 |

---
## 2026-10-27 19:04:22

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 812/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=812)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 812 | 71.1% | 76.6% | 71.4% | 79 / 34 | 0.000 |
| libero_10 | 59 | 54.2% | 64.4% | 50.8% | 11 / 5 | 0.210 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 110 | 34.5% | 39.1% | 26.4% | 13 / 8 | 0.383 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 128 | 82.8% | 93.8% | 91.4% | 18 / 4 | 0.004 |
| Sensor Noise | 122 | 54.1% | 59.0% | 50.0% | 12 / 6 | 0.238 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=899)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 899 | 69.5% | 74.7% | 68.6% | 86 / 39 | 0.000 |
| libero_10 | 146 | 54.8% | 60.3% | 45.9% | 18 / 10 | 0.185 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 128 | 82.8% | 93.8% | 91.4% | 18 / 4 | 0.004 |
| Sensor Noise | 162 | 57.4% | 59.9% | 51.9% | 13 / 9 | 0.523 |

### StructWAM 40k vs 24k (full eval tasks)  (n=812)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 812 | 71.4% | 76.6% | 62 / 20 | 0.000 |
| libero_10 | 59 | 50.8% | 64.4% | 10 / 2 | 0.039 |
| libero_goal | 260 | 70.4% | 74.6% | 18 / 7 | 0.043 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 134 | 60.4% | 69.4% | 19 / 7 | 0.029 |
| Camera Viewpoints | 110 | 26.4% | 39.1% | 18 / 4 | 0.004 |
| Language Instructions | 112 | 94.6% | 94.6% | 2 / 2 | 1.000 |
| Light Conditions | 92 | 98.9% | 98.9% | 0 / 0 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 3 / 1 | 0.625 |
| Robot Initial States | 128 | 91.4% | 93.8% | 7 / 4 | 0.549 |
| Sensor Noise | 122 | 50.0% | 59.0% | 13 / 2 | 0.007 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 7994/40000 | 8.26 s/step | ETA 73:26:12 | loss 0.1241 |
| b0 | 2789/40000 | 8.91 s/step | ETA 92:05:48 | loss 0.1955 |
| b2 | 3211/40000 | 8.96 s/step | ETA 91:35:34 | loss 0.1454 |

---
## 2026-10-27 19:14:28

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 823/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=823)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 823 | 71.0% | 76.7% | 71.2% | 81 / 34 | 0.000 |
| libero_10 | 70 | 55.7% | 67.1% | 51.4% | 13 / 5 | 0.096 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 119 | 37.8% | 43.7% | 28.6% | 15 / 8 | 0.210 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 122 | 54.1% | 59.0% | 50.0% | 12 / 6 | 0.238 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=901)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 901 | 69.4% | 74.6% | 68.6% | 86 / 39 | 0.000 |
| libero_10 | 148 | 54.1% | 59.5% | 45.9% | 18 / 10 | 0.185 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 162 | 57.4% | 59.9% | 51.9% | 13 / 9 | 0.523 |

### StructWAM 40k vs 24k (full eval tasks)  (n=823)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 823 | 71.2% | 76.7% | 66 / 21 | 0.000 |
| libero_10 | 70 | 51.4% | 67.1% | 14 / 3 | 0.013 |
| libero_goal | 260 | 70.4% | 74.6% | 18 / 7 | 0.043 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 134 | 60.4% | 69.4% | 19 / 7 | 0.029 |
| Camera Viewpoints | 119 | 28.6% | 43.7% | 22 / 4 | 0.001 |
| Language Instructions | 112 | 94.6% | 94.6% | 2 / 2 | 1.000 |
| Light Conditions | 92 | 98.9% | 98.9% | 0 / 0 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 3 / 1 | 0.625 |
| Robot Initial States | 130 | 90.8% | 92.3% | 7 / 5 | 0.774 |
| Sensor Noise | 122 | 50.0% | 59.0% | 13 / 2 | 0.007 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 8040/40000 | 8.26 s/step | ETA 73:17:20 | loss 0.1285 |
| b0 | 2856/40000 | 9.19 s/step | ETA 94:49:38 | loss 0.1808 |
| b2 | 3278/40000 | 8.87 s/step | ETA 90:27:26 | loss 0.1789 |

---
## 2026-10-27 19:24:34

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 832/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=832)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 832 | 70.4% | 76.0% | 70.6% | 81 / 35 | 0.000 |
| libero_10 | 79 | 51.9% | 60.8% | 46.8% | 13 / 6 | 0.167 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 128 | 36.7% | 41.4% | 27.3% | 15 / 9 | 0.307 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 122 | 54.1% | 59.0% | 50.0% | 12 / 6 | 0.238 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=901)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 901 | 69.4% | 74.6% | 68.6% | 86 / 39 | 0.000 |
| libero_10 | 148 | 54.1% | 59.5% | 45.9% | 18 / 10 | 0.185 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 162 | 57.4% | 59.9% | 51.9% | 13 / 9 | 0.523 |

### StructWAM 40k vs 24k (full eval tasks)  (n=832)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 832 | 70.6% | 76.0% | 67 / 22 | 0.000 |
| libero_10 | 79 | 46.8% | 60.8% | 15 / 4 | 0.019 |
| libero_goal | 260 | 70.4% | 74.6% | 18 / 7 | 0.043 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 134 | 60.4% | 69.4% | 19 / 7 | 0.029 |
| Camera Viewpoints | 128 | 27.3% | 41.4% | 23 / 5 | 0.001 |
| Language Instructions | 112 | 94.6% | 94.6% | 2 / 2 | 1.000 |
| Light Conditions | 92 | 98.9% | 98.9% | 0 / 0 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 3 / 1 | 0.625 |
| Robot Initial States | 130 | 90.8% | 92.3% | 7 / 5 | 0.774 |
| Sensor Noise | 122 | 50.0% | 59.0% | 13 / 2 | 0.007 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 8112/40000 | 8.22 s/step | ETA 72:46:24 | loss 0.1271 |
| b0 | 2923/40000 | 8.92 s/step | ETA 91:54:40 | loss 0.1913 |
| b2 | 3346/40000 | 9.03 s/step | ETA 91:55:34 | loss 0.1633 |

---
## 2026-10-27 19:34:40

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 846/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=846)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 846 | 69.9% | 75.3% | 69.6% | 82 / 36 | 0.000 |
| libero_10 | 93 | 49.5% | 57.0% | 41.9% | 14 / 7 | 0.189 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 142 | 36.6% | 40.8% | 26.1% | 16 / 10 | 0.327 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 122 | 54.1% | 59.0% | 50.0% | 12 / 6 | 0.238 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=901)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 901 | 69.4% | 74.6% | 68.6% | 86 / 39 | 0.000 |
| libero_10 | 148 | 54.1% | 59.5% | 45.9% | 18 / 10 | 0.185 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 162 | 57.4% | 59.9% | 51.9% | 13 / 9 | 0.523 |

### StructWAM 40k vs 24k (full eval tasks)  (n=846)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 846 | 69.6% | 75.3% | 70 / 22 | 0.000 |
| libero_10 | 93 | 41.9% | 57.0% | 18 / 4 | 0.004 |
| libero_goal | 260 | 70.4% | 74.6% | 18 / 7 | 0.043 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 134 | 60.4% | 69.4% | 19 / 7 | 0.029 |
| Camera Viewpoints | 142 | 26.1% | 40.8% | 26 / 5 | 0.000 |
| Language Instructions | 112 | 94.6% | 94.6% | 2 / 2 | 1.000 |
| Light Conditions | 92 | 98.9% | 98.9% | 0 / 0 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 3 / 1 | 0.625 |
| Robot Initial States | 130 | 90.8% | 92.3% | 7 / 5 | 0.774 |
| Sensor Noise | 122 | 50.0% | 59.0% | 13 / 2 | 0.007 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 8184/40000 | 8.44 s/step | ETA 74:35:00 | loss 0.1158 |
| b0 | 2989/40000 | 8.96 s/step | ETA 92:05:23 | loss 0.2163 |
| b2 | 3413/40000 | 9.26 s/step | ETA 94:08:53 | loss 0.1711 |

---
## 2026-10-27 19:44:47

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 856/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=856)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 856 | 69.7% | 75.5% | 69.4% | 85 / 36 | 0.000 |
| libero_10 | 103 | 50.5% | 60.2% | 42.7% | 17 / 7 | 0.064 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 152 | 38.2% | 44.1% | 27.6% | 19 / 10 | 0.136 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 122 | 54.1% | 59.0% | 50.0% | 12 / 6 | 0.238 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=901)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 901 | 69.4% | 74.6% | 68.6% | 86 / 39 | 0.000 |
| libero_10 | 148 | 54.1% | 59.5% | 45.9% | 18 / 10 | 0.185 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 112 | 93.8% | 94.6% | 94.6% | 4 / 3 | 1.000 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 162 | 57.4% | 59.9% | 51.9% | 13 / 9 | 0.523 |

### StructWAM 40k vs 24k (full eval tasks)  (n=856)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 856 | 69.4% | 75.5% | 74 / 22 | 0.000 |
| libero_10 | 103 | 42.7% | 60.2% | 22 / 4 | 0.001 |
| libero_goal | 260 | 70.4% | 74.6% | 18 / 7 | 0.043 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 134 | 60.4% | 69.4% | 19 / 7 | 0.029 |
| Camera Viewpoints | 152 | 27.6% | 44.1% | 30 / 5 | 0.000 |
| Language Instructions | 112 | 94.6% | 94.6% | 2 / 2 | 1.000 |
| Light Conditions | 92 | 98.9% | 98.9% | 0 / 0 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 3 / 1 | 0.625 |
| Robot Initial States | 130 | 90.8% | 92.3% | 7 / 5 | 0.774 |
| Sensor Noise | 122 | 50.0% | 59.0% | 13 / 2 | 0.007 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 8257/40000 | 8.37 s/step | ETA 73:46:51 | loss 0.1633 |
| b0 | 3056/40000 | 9.32 s/step | ETA 95:40:21 | loss 0.1830 |
| b2 | 3480/40000 | 9.28 s/step | ETA 94:10:48 | loss 0.1464 |

---
## 2026-10-27 19:54:53

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 878/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=878)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 878 | 70.0% | 75.6% | 69.5% | 85 / 36 | 0.000 |
| libero_10 | 125 | 56.0% | 64.0% | 48.0% | 17 / 7 | 0.064 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 129 | 94.6% | 95.3% | 93.8% | 4 / 3 | 1.000 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 122 | 54.1% | 59.0% | 50.0% | 12 / 6 | 0.238 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=918)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 918 | 69.9% | 75.1% | 69.0% | 86 / 39 | 0.000 |
| libero_10 | 165 | 58.8% | 63.6% | 50.3% | 18 / 10 | 0.185 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 129 | 94.6% | 95.3% | 93.8% | 4 / 3 | 1.000 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 162 | 57.4% | 59.9% | 51.9% | 13 / 9 | 0.523 |

### StructWAM 40k vs 24k (full eval tasks)  (n=878)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 878 | 69.5% | 75.6% | 76 / 22 | 0.000 |
| libero_10 | 125 | 48.0% | 64.0% | 24 / 4 | 0.000 |
| libero_goal | 260 | 70.4% | 74.6% | 18 / 7 | 0.043 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 134 | 60.4% | 69.4% | 19 / 7 | 0.029 |
| Camera Viewpoints | 157 | 27.4% | 43.3% | 30 / 5 | 0.000 |
| Language Instructions | 129 | 93.8% | 95.3% | 4 / 2 | 0.688 |
| Light Conditions | 92 | 98.9% | 98.9% | 0 / 0 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 3 / 1 | 0.625 |
| Robot Initial States | 130 | 90.8% | 92.3% | 7 / 5 | 0.774 |
| Sensor Noise | 122 | 50.0% | 59.0% | 13 / 2 | 0.007 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 8328/40000 | 8.46 s/step | ETA 74:23:42 | loss 0.1381 |
| b0 | 3123/40000 | 8.93 s/step | ETA 91:30:37 | loss 0.1745 |
| b2 | 3548/40000 | 9.02 s/step | ETA 91:21:17 | loss 0.1556 |

---
## 2026-10-27 20:05:00

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 892/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=892)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 892 | 70.5% | 75.7% | 69.5% | 85 / 39 | 0.000 |
| libero_10 | 139 | 60.4% | 65.5% | 50.4% | 17 / 10 | 0.248 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 143 | 95.1% | 93.7% | 91.6% | 4 / 6 | 0.754 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 122 | 54.1% | 59.0% | 50.0% | 12 / 6 | 0.238 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=932)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 932 | 70.4% | 75.1% | 69.0% | 86 / 42 | 0.000 |
| libero_10 | 179 | 62.0% | 64.8% | 52.0% | 18 / 13 | 0.473 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 143 | 95.1% | 93.7% | 91.6% | 4 / 6 | 0.754 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 162 | 57.4% | 59.9% | 51.9% | 13 / 9 | 0.523 |

### StructWAM 40k vs 24k (full eval tasks)  (n=892)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 892 | 69.5% | 75.7% | 78 / 23 | 0.000 |
| libero_10 | 139 | 50.4% | 65.5% | 26 / 5 | 0.000 |
| libero_goal | 260 | 70.4% | 74.6% | 18 / 7 | 0.043 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 134 | 60.4% | 69.4% | 19 / 7 | 0.029 |
| Camera Viewpoints | 157 | 27.4% | 43.3% | 30 / 5 | 0.000 |
| Language Instructions | 143 | 91.6% | 93.7% | 6 / 3 | 0.508 |
| Light Conditions | 92 | 98.9% | 98.9% | 0 / 0 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 3 / 1 | 0.625 |
| Robot Initial States | 130 | 90.8% | 92.3% | 7 / 5 | 0.774 |
| Sensor Noise | 122 | 50.0% | 59.0% | 13 / 2 | 0.007 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 8399/40000 | 8.35 s/step | ETA 73:17:50 | loss 0.1251 |
| b0 | 3190/40000 | 9.09 s/step | ETA 92:53:39 | loss 0.1850 |
| b2 | 3614/40000 | 9.14 s/step | ETA 92:20:29 | loss 0.1506 |

---
## 2026-10-27 20:15:06

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 908/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=908)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 908 | 71.0% | 76.0% | 69.8% | 85 / 40 | 0.000 |
| libero_10 | 155 | 64.5% | 68.4% | 54.2% | 17 / 11 | 0.345 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 129 | 56.6% | 61.2% | 51.9% | 12 / 6 | 0.238 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=941)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 941 | 70.7% | 75.2% | 69.2% | 86 / 43 | 0.000 |
| libero_10 | 188 | 63.8% | 66.0% | 53.7% | 18 / 14 | 0.597 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 162 | 57.4% | 59.9% | 51.9% | 13 / 9 | 0.523 |

### StructWAM 40k vs 24k (full eval tasks)  (n=908)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 908 | 69.8% | 76.0% | 79 / 23 | 0.000 |
| libero_10 | 155 | 54.2% | 68.4% | 27 / 5 | 0.000 |
| libero_goal | 260 | 70.4% | 74.6% | 18 / 7 | 0.043 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 134 | 60.4% | 69.4% | 19 / 7 | 0.029 |
| Camera Viewpoints | 157 | 27.4% | 43.3% | 30 / 5 | 0.000 |
| Language Instructions | 152 | 91.4% | 93.4% | 6 / 3 | 0.508 |
| Light Conditions | 92 | 98.9% | 98.9% | 0 / 0 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 3 / 1 | 0.625 |
| Robot Initial States | 130 | 90.8% | 92.3% | 7 / 5 | 0.774 |
| Sensor Noise | 129 | 51.9% | 61.2% | 14 / 2 | 0.004 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 8470/40000 | 8.43 s/step | ETA 73:49:54 | loss 0.1613 |
| b0 | 3257/40000 | 9.07 s/step | ETA 92:37:17 | loss 0.1897 |
| b2 | 3680/40000 | 8.98 s/step | ETA 90:34:35 | loss 0.1889 |

---
## 2026-10-27 20:25:13

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 915/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=915)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 915 | 71.0% | 76.0% | 69.8% | 85 / 40 | 0.000 |
| libero_10 | 162 | 64.8% | 68.5% | 54.9% | 17 / 11 | 0.345 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 136 | 57.4% | 61.8% | 52.9% | 12 / 6 | 0.238 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=941)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 941 | 70.7% | 75.2% | 69.2% | 86 / 43 | 0.000 |
| libero_10 | 188 | 63.8% | 66.0% | 53.7% | 18 / 14 | 0.597 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 162 | 57.4% | 59.9% | 51.9% | 13 / 9 | 0.523 |

### StructWAM 40k vs 24k (full eval tasks)  (n=915)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 915 | 69.8% | 76.0% | 79 / 23 | 0.000 |
| libero_10 | 162 | 54.9% | 68.5% | 27 / 5 | 0.000 |
| libero_goal | 260 | 70.4% | 74.6% | 18 / 7 | 0.043 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 134 | 60.4% | 69.4% | 19 / 7 | 0.029 |
| Camera Viewpoints | 157 | 27.4% | 43.3% | 30 / 5 | 0.000 |
| Language Instructions | 152 | 91.4% | 93.4% | 6 / 3 | 0.508 |
| Light Conditions | 92 | 98.9% | 98.9% | 0 / 0 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 3 / 1 | 0.625 |
| Robot Initial States | 130 | 90.8% | 92.3% | 7 / 5 | 0.774 |
| Sensor Noise | 136 | 52.9% | 61.8% | 14 / 2 | 0.004 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 8541/40000 | 8.69 s/step | ETA 75:53:58 | loss 0.1171 |
| b0 | 3323/40000 | 8.93 s/step | ETA 90:56:45 | loss 0.1816 |
| b2 | 3747/40000 | 9.15 s/step | ETA 92:09:16 | loss 0.1557 |

---
## 2026-10-27 20:35:20

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 920/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=920)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 920 | 71.0% | 76.0% | 69.9% | 86 / 40 | 0.000 |
| libero_10 | 167 | 64.7% | 68.9% | 55.7% | 18 / 11 | 0.265 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 141 | 57.4% | 62.4% | 53.9% | 13 / 6 | 0.167 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=941)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 941 | 70.7% | 75.2% | 69.2% | 86 / 43 | 0.000 |
| libero_10 | 188 | 63.8% | 66.0% | 53.7% | 18 / 14 | 0.597 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 162 | 57.4% | 59.9% | 51.9% | 13 / 9 | 0.523 |

### StructWAM 40k vs 24k (full eval tasks)  (n=920)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 920 | 69.9% | 76.0% | 79 / 23 | 0.000 |
| libero_10 | 167 | 55.7% | 68.9% | 27 / 5 | 0.000 |
| libero_goal | 260 | 70.4% | 74.6% | 18 / 7 | 0.043 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 134 | 60.4% | 69.4% | 19 / 7 | 0.029 |
| Camera Viewpoints | 157 | 27.4% | 43.3% | 30 / 5 | 0.000 |
| Language Instructions | 152 | 91.4% | 93.4% | 6 / 3 | 0.508 |
| Light Conditions | 92 | 98.9% | 98.9% | 0 / 0 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 3 / 1 | 0.625 |
| Robot Initial States | 130 | 90.8% | 92.3% | 7 / 5 | 0.774 |
| Sensor Noise | 141 | 53.9% | 62.4% | 14 / 2 | 0.004 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 8611/40000 | 8.85 s/step | ETA 77:11:36 | loss 0.1175 |
| b0 | 3390/40000 | 8.96 s/step | ETA 91:05:43 | loss 0.1733 |
| b2 | 3815/40000 | 9.15 s/step | ETA 91:55:25 | loss 0.1796 |

---
## 2026-10-27 20:45:26

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 927/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=927)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 927 | 71.0% | 75.9% | 69.9% | 86 / 40 | 0.000 |
| libero_10 | 174 | 64.9% | 69.0% | 56.3% | 18 / 11 | 0.265 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 148 | 58.1% | 62.8% | 54.7% | 13 / 6 | 0.167 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=941)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 941 | 70.7% | 75.2% | 69.2% | 86 / 43 | 0.000 |
| libero_10 | 188 | 63.8% | 66.0% | 53.7% | 18 / 14 | 0.597 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 162 | 57.4% | 59.9% | 51.9% | 13 / 9 | 0.523 |

### StructWAM 40k vs 24k (full eval tasks)  (n=927)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 927 | 69.9% | 75.9% | 79 / 23 | 0.000 |
| libero_10 | 174 | 56.3% | 69.0% | 27 / 5 | 0.000 |
| libero_goal | 260 | 70.4% | 74.6% | 18 / 7 | 0.043 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 134 | 60.4% | 69.4% | 19 / 7 | 0.029 |
| Camera Viewpoints | 157 | 27.4% | 43.3% | 30 / 5 | 0.000 |
| Language Instructions | 152 | 91.4% | 93.4% | 6 / 3 | 0.508 |
| Light Conditions | 92 | 98.9% | 98.9% | 0 / 0 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 3 / 1 | 0.625 |
| Robot Initial States | 130 | 90.8% | 92.3% | 7 / 5 | 0.774 |
| Sensor Noise | 148 | 54.7% | 62.8% | 14 / 2 | 0.004 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 8682/40000 | 8.63 s/step | ETA 75:06:55 | loss 0.1446 |
| b0 | 3457/40000 | 9.27 s/step | ETA 94:03:06 | loss 0.1770 |
| b2 | 3882/40000 | 8.92 s/step | ETA 89:29:05 | loss 0.1499 |

---
## 2026-10-27 20:55:33

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 934/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=934)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 934 | 70.8% | 75.6% | 69.5% | 86 / 41 | 0.000 |
| libero_10 | 181 | 64.1% | 67.4% | 54.7% | 18 / 12 | 0.362 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 155 | 57.4% | 61.3% | 52.9% | 13 / 7 | 0.263 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=941)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 941 | 70.7% | 75.2% | 69.2% | 86 / 43 | 0.000 |
| libero_10 | 188 | 63.8% | 66.0% | 53.7% | 18 / 14 | 0.597 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 162 | 57.4% | 59.9% | 51.9% | 13 / 9 | 0.523 |

### StructWAM 40k vs 24k (full eval tasks)  (n=934)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 934 | 69.5% | 75.6% | 80 / 23 | 0.000 |
| libero_10 | 181 | 54.7% | 67.4% | 28 / 5 | 0.000 |
| libero_goal | 260 | 70.4% | 74.6% | 18 / 7 | 0.043 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 134 | 60.4% | 69.4% | 19 / 7 | 0.029 |
| Camera Viewpoints | 157 | 27.4% | 43.3% | 30 / 5 | 0.000 |
| Language Instructions | 152 | 91.4% | 93.4% | 6 / 3 | 0.508 |
| Light Conditions | 92 | 98.9% | 98.9% | 0 / 0 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 3 / 1 | 0.625 |
| Robot Initial States | 130 | 90.8% | 92.3% | 7 / 5 | 0.774 |
| Sensor Noise | 155 | 52.9% | 61.3% | 15 / 2 | 0.002 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 8753/40000 | 8.62 s/step | ETA 74:47:01 | loss 0.1330 |
| b0 | 3523/40000 | 9.06 s/step | ETA 91:48:33 | loss 0.1742 |
| b2 | 3949/40000 | 9.02 s/step | ETA 90:16:42 | loss 0.1626 |

---
## 2026-10-27 21:05:40

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 939/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=939)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 939 | 70.6% | 75.3% | 69.2% | 86 / 42 | 0.000 |
| libero_10 | 186 | 63.4% | 66.1% | 53.8% | 18 / 13 | 0.473 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 160 | 56.9% | 60.0% | 51.9% | 13 / 8 | 0.383 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=941)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 941 | 70.7% | 75.2% | 69.2% | 86 / 43 | 0.000 |
| libero_10 | 188 | 63.8% | 66.0% | 53.7% | 18 / 14 | 0.597 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 162 | 57.4% | 59.9% | 51.9% | 13 / 9 | 0.523 |

### StructWAM 40k vs 24k (full eval tasks)  (n=939)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 939 | 69.2% | 75.3% | 80 / 23 | 0.000 |
| libero_10 | 186 | 53.8% | 66.1% | 28 / 5 | 0.000 |
| libero_goal | 260 | 70.4% | 74.6% | 18 / 7 | 0.043 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 134 | 60.4% | 69.4% | 19 / 7 | 0.029 |
| Camera Viewpoints | 157 | 27.4% | 43.3% | 30 / 5 | 0.000 |
| Language Instructions | 152 | 91.4% | 93.4% | 6 / 3 | 0.508 |
| Light Conditions | 92 | 98.9% | 98.9% | 0 / 0 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 3 / 1 | 0.625 |
| Robot Initial States | 130 | 90.8% | 92.3% | 7 / 5 | 0.774 |
| Sensor Noise | 160 | 51.9% | 60.0% | 15 / 2 | 0.002 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 8824/40000 | 8.63 s/step | ETA 74:46:40 | loss 0.1368 |
| b0 | 3590/40000 | 9.00 s/step | ETA 91:02:13 | loss 0.2357 |
| b2 | 4000/40000 | 8.84 s/step | ETA 88:26:52 | loss 0.1579 |

---
## 2026-10-27 21:15:47

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 944/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=944)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 944 | 70.8% | 75.1% | 69.2% | 86 / 45 | 0.000 |
| libero_10 | 191 | 64.4% | 65.4% | 53.9% | 18 / 16 | 0.864 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 165 | 58.2% | 59.4% | 52.1% | 13 / 11 | 0.839 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=944)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 944 | 70.8% | 75.1% | 69.2% | 86 / 45 | 0.000 |
| libero_10 | 191 | 64.4% | 65.4% | 53.9% | 18 / 16 | 0.864 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 83.3% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 165 | 58.2% | 59.4% | 52.1% | 13 / 11 | 0.839 |

### StructWAM 40k vs 24k (full eval tasks)  (n=944)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 944 | 69.2% | 75.1% | 80 / 24 | 0.000 |
| libero_10 | 191 | 53.9% | 65.4% | 28 / 6 | 0.000 |
| libero_goal | 260 | 70.4% | 74.6% | 18 / 7 | 0.043 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 134 | 60.4% | 69.4% | 19 / 7 | 0.029 |
| Camera Viewpoints | 157 | 27.4% | 43.3% | 30 / 5 | 0.000 |
| Language Instructions | 152 | 91.4% | 93.4% | 6 / 3 | 0.508 |
| Light Conditions | 92 | 98.9% | 98.9% | 0 / 0 | 1.000 |
| Objects Layout | 114 | 83.3% | 85.1% | 3 / 1 | 0.625 |
| Robot Initial States | 130 | 90.8% | 92.3% | 7 / 5 | 0.774 |
| Sensor Noise | 165 | 52.1% | 59.4% | 15 / 3 | 0.008 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 8896/40000 | 8.61 s/step | ETA 74:21:07 | loss 0.1164 |
| b0 | 3658/40000 | 9.26 s/step | ETA 93:27:37 | loss 0.1740 |
| b2 | 4062/40000 | 9.03 s/step | ETA 90:07:45 | loss 0.1551 |

---
## 2026-10-27 21:25:54

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 951/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=951)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 951 | 70.8% | 75.1% | 69.2% | 86 / 45 | 0.000 |
| libero_10 | 198 | 64.6% | 65.7% | 54.5% | 18 / 16 | 0.864 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 115 | 83.5% | 85.2% | 83.5% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 171 | 58.5% | 59.6% | 52.6% | 13 / 11 | 0.839 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=951)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 951 | 70.8% | 75.1% | 69.2% | 86 / 45 | 0.000 |
| libero_10 | 198 | 64.6% | 65.7% | 54.5% | 18 / 16 | 0.864 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 115 | 83.5% | 85.2% | 83.5% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 171 | 58.5% | 59.6% | 52.6% | 13 / 11 | 0.839 |

### StructWAM 40k vs 24k (full eval tasks)  (n=951)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 951 | 69.2% | 75.1% | 80 / 24 | 0.000 |
| libero_10 | 198 | 54.5% | 65.7% | 28 / 6 | 0.000 |
| libero_goal | 260 | 70.4% | 74.6% | 18 / 7 | 0.043 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 134 | 60.4% | 69.4% | 19 / 7 | 0.029 |
| Camera Viewpoints | 157 | 27.4% | 43.3% | 30 / 5 | 0.000 |
| Language Instructions | 152 | 91.4% | 93.4% | 6 / 3 | 0.508 |
| Light Conditions | 92 | 98.9% | 98.9% | 0 / 0 | 1.000 |
| Objects Layout | 115 | 83.5% | 85.2% | 3 / 1 | 0.625 |
| Robot Initial States | 130 | 90.8% | 92.3% | 7 / 5 | 0.774 |
| Sensor Noise | 171 | 52.6% | 59.6% | 15 / 3 | 0.008 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 8967/40000 | 8.40 s/step | ETA 72:26:29 | loss 0.1371 |
| b0 | 3725/40000 | 9.06 s/step | ETA 91:14:35 | loss 0.1848 |
| b2 | 4129/40000 | 8.70 s/step | ETA 86:39:26 | loss 0.1506 |

---
## 2026-10-27 21:36:00

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 963/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=963)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 963 | 70.5% | 74.9% | 69.1% | 88 / 46 | 0.000 |
| libero_10 | 210 | 63.8% | 65.2% | 54.8% | 20 / 17 | 0.743 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 123 | 82.9% | 85.4% | 83.7% | 6 / 3 | 0.508 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 175 | 57.1% | 58.3% | 51.4% | 13 / 11 | 0.839 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=963)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 963 | 70.5% | 74.9% | 69.1% | 88 / 46 | 0.000 |
| libero_10 | 210 | 63.8% | 65.2% | 54.8% | 20 / 17 | 0.743 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 123 | 82.9% | 85.4% | 83.7% | 6 / 3 | 0.508 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 175 | 57.1% | 58.3% | 51.4% | 13 / 11 | 0.839 |

### StructWAM 40k vs 24k (full eval tasks)  (n=963)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 963 | 69.1% | 74.9% | 81 / 25 | 0.000 |
| libero_10 | 210 | 54.8% | 65.2% | 29 / 7 | 0.000 |
| libero_goal | 260 | 70.4% | 74.6% | 18 / 7 | 0.043 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 134 | 60.4% | 69.4% | 19 / 7 | 0.029 |
| Camera Viewpoints | 157 | 27.4% | 43.3% | 30 / 5 | 0.000 |
| Language Instructions | 152 | 91.4% | 93.4% | 6 / 3 | 0.508 |
| Light Conditions | 92 | 98.9% | 98.9% | 0 / 0 | 1.000 |
| Objects Layout | 123 | 83.7% | 85.4% | 4 / 2 | 0.688 |
| Robot Initial States | 130 | 90.8% | 92.3% | 7 / 5 | 0.774 |
| Sensor Noise | 175 | 51.4% | 58.3% | 15 / 3 | 0.008 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 9038/40000 | 8.59 s/step | ETA 73:53:21 | loss 0.1360 |
| b0 | 3791/40000 | 9.04 s/step | ETA 90:54:25 | loss 0.1916 |
| b2 | 4197/40000 | 8.96 s/step | ETA 89:07:02 | loss 0.1362 |

---
## 2026-10-27 21:46:07

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 983/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=983)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 983 | 70.9% | 75.4% | 69.5% | 90 / 46 | 0.000 |
| libero_10 | 230 | 66.1% | 68.3% | 57.8% | 22 / 17 | 0.522 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 142 | 83.8% | 87.3% | 84.5% | 8 / 3 | 0.227 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 176 | 57.4% | 58.5% | 51.7% | 13 / 11 | 0.839 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=983)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 983 | 70.9% | 75.4% | 69.5% | 90 / 46 | 0.000 |
| libero_10 | 230 | 66.1% | 68.3% | 57.8% | 22 / 17 | 0.522 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 92 | 97.8% | 98.9% | 98.9% | 2 / 1 | 1.000 |
| Objects Layout | 142 | 83.8% | 87.3% | 84.5% | 8 / 3 | 0.227 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 176 | 57.4% | 58.5% | 51.7% | 13 / 11 | 0.839 |

### StructWAM 40k vs 24k (full eval tasks)  (n=983)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 983 | 69.5% | 75.4% | 83 / 25 | 0.000 |
| libero_10 | 230 | 57.8% | 68.3% | 31 / 7 | 0.000 |
| libero_goal | 260 | 70.4% | 74.6% | 18 / 7 | 0.043 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 134 | 60.4% | 69.4% | 19 / 7 | 0.029 |
| Camera Viewpoints | 157 | 27.4% | 43.3% | 30 / 5 | 0.000 |
| Language Instructions | 152 | 91.4% | 93.4% | 6 / 3 | 0.508 |
| Light Conditions | 92 | 98.9% | 98.9% | 0 / 0 | 1.000 |
| Objects Layout | 142 | 84.5% | 87.3% | 6 / 2 | 0.289 |
| Robot Initial States | 130 | 90.8% | 92.3% | 7 / 5 | 0.774 |
| Sensor Noise | 176 | 51.7% | 58.5% | 15 / 3 | 0.008 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 9110/40000 | 8.63 s/step | ETA 74:01:48 | loss 0.1274 |
| b0 | 3859/40000 | 8.94 s/step | ETA 89:47:59 | loss 0.2085 |
| b2 | 4264/40000 | 9.26 s/step | ETA 91:57:34 | loss 0.1603 |

---
## 2026-10-27 21:56:14

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 991/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=991)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 991 | 70.9% | 75.3% | 69.0% | 91 / 48 | 0.000 |
| libero_10 | 238 | 66.4% | 68.1% | 56.3% | 23 / 19 | 0.644 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 98 | 96.9% | 96.9% | 92.9% | 3 / 3 | 1.000 |
| Objects Layout | 144 | 83.3% | 86.8% | 84.0% | 8 / 3 | 0.227 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 176 | 57.4% | 58.5% | 51.7% | 13 / 11 | 0.839 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=991)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 991 | 70.9% | 75.3% | 69.0% | 91 / 48 | 0.000 |
| libero_10 | 238 | 66.4% | 68.1% | 56.3% | 23 / 19 | 0.644 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 98 | 96.9% | 96.9% | 92.9% | 3 / 3 | 1.000 |
| Objects Layout | 144 | 83.3% | 86.8% | 84.0% | 8 / 3 | 0.227 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 176 | 57.4% | 58.5% | 51.7% | 13 / 11 | 0.839 |

### StructWAM 40k vs 24k (full eval tasks)  (n=991)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 991 | 69.0% | 75.3% | 87 / 25 | 0.000 |
| libero_10 | 238 | 56.3% | 68.1% | 35 / 7 | 0.000 |
| libero_goal | 260 | 70.4% | 74.6% | 18 / 7 | 0.043 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 134 | 60.4% | 69.4% | 19 / 7 | 0.029 |
| Camera Viewpoints | 157 | 27.4% | 43.3% | 30 / 5 | 0.000 |
| Language Instructions | 152 | 91.4% | 93.4% | 6 / 3 | 0.508 |
| Light Conditions | 98 | 92.9% | 96.9% | 4 / 0 | 0.125 |
| Objects Layout | 144 | 84.0% | 86.8% | 6 / 2 | 0.289 |
| Robot Initial States | 130 | 90.8% | 92.3% | 7 / 5 | 0.774 |
| Sensor Noise | 176 | 51.7% | 58.5% | 15 / 3 | 0.008 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 9182/40000 | 8.50 s/step | ETA 72:44:44 | loss 0.1171 |
| b0 | 3927/40000 | 8.85 s/step | ETA 88:38:00 | loss 0.1582 |
| b2 | 4330/40000 | 8.65 s/step | ETA 85:41:39 | loss 0.1789 |

---
## 2026-10-27 22:06:21

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 996/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=996)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 996 | 71.1% | 75.4% | 69.1% | 91 / 48 | 0.000 |
| libero_10 | 243 | 67.1% | 68.7% | 56.8% | 23 / 19 | 0.644 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 103 | 97.1% | 97.1% | 92.2% | 3 / 3 | 1.000 |
| Objects Layout | 144 | 83.3% | 86.8% | 84.0% | 8 / 3 | 0.227 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 176 | 57.4% | 58.5% | 51.7% | 13 / 11 | 0.839 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=996)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 996 | 71.1% | 75.4% | 69.1% | 91 / 48 | 0.000 |
| libero_10 | 243 | 67.1% | 68.7% | 56.8% | 23 / 19 | 0.644 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 103 | 97.1% | 97.1% | 92.2% | 3 / 3 | 1.000 |
| Objects Layout | 144 | 83.3% | 86.8% | 84.0% | 8 / 3 | 0.227 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 176 | 57.4% | 58.5% | 51.7% | 13 / 11 | 0.839 |

### StructWAM 40k vs 24k (full eval tasks)  (n=996)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 996 | 69.1% | 75.4% | 88 / 25 | 0.000 |
| libero_10 | 243 | 56.8% | 68.7% | 36 / 7 | 0.000 |
| libero_goal | 260 | 70.4% | 74.6% | 18 / 7 | 0.043 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 134 | 60.4% | 69.4% | 19 / 7 | 0.029 |
| Camera Viewpoints | 157 | 27.4% | 43.3% | 30 / 5 | 0.000 |
| Language Instructions | 152 | 91.4% | 93.4% | 6 / 3 | 0.508 |
| Light Conditions | 103 | 92.2% | 97.1% | 5 / 0 | 0.062 |
| Objects Layout | 144 | 84.0% | 86.8% | 6 / 2 | 0.289 |
| Robot Initial States | 130 | 90.8% | 92.3% | 7 / 5 | 0.774 |
| Sensor Noise | 176 | 51.7% | 58.5% | 15 / 3 | 0.008 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 9253/40000 | 8.33 s/step | ETA 71:09:11 | loss 0.1160 |
| b0 | 3993/40000 | 9.16 s/step | ETA 91:36:27 | loss 0.1870 |
| b2 | 4399/40000 | 8.71 s/step | ETA 86:07:00 | loss 0.1555 |

---
## 2026-10-27 22:16:28

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 1004/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=1004)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 1004 | 71.2% | 75.3% | 69.0% | 91 / 50 | 0.001 |
| libero_10 | 251 | 67.7% | 68.5% | 57.0% | 23 / 21 | 0.880 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 111 | 96.4% | 94.6% | 90.1% | 3 / 5 | 0.727 |
| Objects Layout | 144 | 83.3% | 86.8% | 84.0% | 8 / 3 | 0.227 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 176 | 57.4% | 58.5% | 51.7% | 13 / 11 | 0.839 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=1004)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 1004 | 71.2% | 75.3% | 69.0% | 91 / 50 | 0.001 |
| libero_10 | 251 | 67.7% | 68.5% | 57.0% | 23 / 21 | 0.880 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 111 | 96.4% | 94.6% | 90.1% | 3 / 5 | 0.727 |
| Objects Layout | 144 | 83.3% | 86.8% | 84.0% | 8 / 3 | 0.227 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 176 | 57.4% | 58.5% | 51.7% | 13 / 11 | 0.839 |

### StructWAM 40k vs 24k (full eval tasks)  (n=1004)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 1004 | 69.0% | 75.3% | 89 / 26 | 0.000 |
| libero_10 | 251 | 57.0% | 68.5% | 37 / 8 | 0.000 |
| libero_goal | 260 | 70.4% | 74.6% | 18 / 7 | 0.043 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 134 | 60.4% | 69.4% | 19 / 7 | 0.029 |
| Camera Viewpoints | 157 | 27.4% | 43.3% | 30 / 5 | 0.000 |
| Language Instructions | 152 | 91.4% | 93.4% | 6 / 3 | 0.508 |
| Light Conditions | 111 | 90.1% | 94.6% | 6 / 1 | 0.125 |
| Objects Layout | 144 | 84.0% | 86.8% | 6 / 2 | 0.289 |
| Robot Initial States | 130 | 90.8% | 92.3% | 7 / 5 | 0.774 |
| Sensor Noise | 176 | 51.7% | 58.5% | 15 / 3 | 0.008 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 9324/40000 | 8.42 s/step | ETA 71:46:55 | loss 0.1198 |
| b0 | 4040/40000 | 8.97 s/step | ETA 89:36:48 | loss 0.1658 |
| b2 | 4467/40000 | 9.10 s/step | ETA 89:48:00 | loss 0.1300 |

---
## 2026-10-27 22:26:35

Progress: OpenWAM 40k 1005/1005, StructWAM 40k 1005/1005

### Full eval: StructWAM 40k vs OpenWAM 40k (+ StructWAM 24k)  (n=1005)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 1005 | 71.2% | 75.2% | 69.1% | 91 / 51 | 0.001 |
| libero_10 | 252 | 67.9% | 68.3% | 57.1% | 23 / 22 | 1.000 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 112 | 96.4% | 93.8% | 90.2% | 3 / 6 | 0.508 |
| Objects Layout | 144 | 83.3% | 86.8% | 84.0% | 8 / 3 | 0.227 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 176 | 57.4% | 58.5% | 51.7% | 13 / 11 | 0.839 |

### Full eval + earlier libero_10 weak-category run: StructWAM 40k vs OpenWAM 40k  (n=1005)

| group | n | OpenWAM 40k | StructWAM 40k | StructWAM 24k | StructWAM 40k only / OpenWAM 40k only | p |
|---|---|---|---|---|---|---|
| ALL | 1005 | 71.2% | 75.2% | 69.1% | 91 / 51 | 0.001 |
| libero_10 | 252 | 67.9% | 68.3% | 57.1% | 23 / 22 | 1.000 |
| libero_goal | 260 | 66.9% | 74.6% | 70.4% | 31 / 11 | 0.003 |
| libero_object | 252 | 75.0% | 77.4% | 73.0% | 15 / 9 | 0.307 |
| libero_spatial | 241 | 75.5% | 80.9% | 75.9% | 22 / 9 | 0.029 |
| Background Textures | 134 | 57.5% | 69.4% | 60.4% | 26 / 10 | 0.011 |
| Camera Viewpoints | 157 | 37.6% | 43.3% | 27.4% | 19 / 10 | 0.136 |
| Language Instructions | 152 | 95.4% | 93.4% | 91.4% | 4 / 7 | 0.549 |
| Light Conditions | 112 | 96.4% | 93.8% | 90.2% | 3 / 6 | 0.508 |
| Objects Layout | 144 | 83.3% | 86.8% | 84.0% | 8 / 3 | 0.227 |
| Robot Initial States | 130 | 81.5% | 92.3% | 90.8% | 18 / 4 | 0.004 |
| Sensor Noise | 176 | 57.4% | 58.5% | 51.7% | 13 / 11 | 0.839 |

### StructWAM 40k vs 24k (full eval tasks)  (n=1005)

| group | n | StructWAM 24k | StructWAM 40k | StructWAM 40k only / StructWAM 24k only | p |
|---|---|---|---|---|---|
| ALL | 1005 | 69.1% | 75.2% | 89 / 27 | 0.000 |
| libero_10 | 252 | 57.1% | 68.3% | 37 / 9 | 0.000 |
| libero_goal | 260 | 70.4% | 74.6% | 18 / 7 | 0.043 |
| libero_object | 252 | 73.0% | 77.4% | 16 / 5 | 0.027 |
| libero_spatial | 241 | 75.9% | 80.9% | 18 / 6 | 0.023 |
| Background Textures | 134 | 60.4% | 69.4% | 19 / 7 | 0.029 |
| Camera Viewpoints | 157 | 27.4% | 43.3% | 30 / 5 | 0.000 |
| Language Instructions | 152 | 91.4% | 93.4% | 6 / 3 | 0.508 |
| Light Conditions | 112 | 90.2% | 93.8% | 6 / 2 | 0.289 |
| Objects Layout | 144 | 84.0% | 86.8% | 6 / 2 | 0.289 |
| Robot Initial States | 130 | 90.8% | 92.3% | 7 / 5 | 0.774 |
| Sensor Noise | 176 | 51.7% | 58.5% | 15 / 3 | 0.008 |

### Training

| arm | step | speed | remaining | loss |
|---|---|---|---|---|
| a2 | 9395/40000 | 8.57 s/step | ETA 72:52:05 | loss 0.1286 |
| b0 | 4107/40000 | 9.05 s/step | ETA 90:14:36 | loss 0.1881 |
| b2 | 4535/40000 | 9.02 s/step | ETA 88:49:05 | loss 0.1214 |

---
