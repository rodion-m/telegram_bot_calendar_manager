### Update app, promote and stop previous version
```bash
gcloud app deploy --quiet --promote --stop-previous-version
```

### Log
```bash
gcloud app logs tail -s default
```