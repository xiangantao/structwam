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
