### Update app, promote and stop previous version
```bash
gcloud app deploy --quiet --promote --stop-previous-version
```

### Log
```bash
gcloud app logs tail -s default
```

### Set Telegram Webhook
```bash
curl -F "url=https://aica-ai-calendar-assistant.de.r.appspot.com/webhook" https://api.telegram.org/bot{token}/setWebhook
```