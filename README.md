# Calculating correlations between judges


Sample .env file:

```
PROXY_URL='socks5://login:pword@ip:port'
API_TOKEN='sk-or-v1-xxx'
API_LINK='https://openrouter.ai/api/v1'
```

To run:

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
bash generate_plots.sh
```
