mkdir -p ~/.streamlit/echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml

streamlit run app.py

# mlflow run /app/ --env-manager=local && mlflow ui