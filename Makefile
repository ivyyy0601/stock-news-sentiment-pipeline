airflow:
	echo 'export PROJECT_DIR="/Users/ValDLaw/Documents/COLUMBIA/BIG DATA/stock-news-sentiment-pipeline"' >> ~/.zprofile
	echo 'export ALPHAVANTAGE_API_KEY="ZDOU31JUM67TI5K0"' >> ~/.zprofile
	source ~/.zprofile
	airflow standalone

requirements:
	pip install -r src/requirements.txt

dashboard:
	streamlit run ui/app.py