notebook:
	jupyter notebook

mlflow:
	mlflow ui

serve:
	uvicorn server:app --reload
