@echo off
setlocal

if not exist .venv (
  python -m venv .venv
)

call .venv\Scripts\activate
python -m pip install -r requirements.txt
python src\generate_preprocessed_data.py
python src\eda.py
python src\load_to_sqlite.py
python src\train_model.py
python src\predict.py

endlocal
