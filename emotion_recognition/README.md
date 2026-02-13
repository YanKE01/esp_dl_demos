* dataset download: https://tianchi.aliyun.com/dataset/208113
* train with torch 2.2.2:
```powershell
uv run python emotion_recognition/train.py
```
* evaluate checkpoint on RAF-DB valid:
```powershell
uv run python emotion_recognition/evaluate.py
```
* quantize for ESP32-P4:
```powershell
uv run python emotion_recognition/quantize.py
```
