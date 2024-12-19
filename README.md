## Running

#### Run locally

Set your keys in Dockerfile.local and then run:

```bash
docker build -f Dockerfile.local -t aica:image . && docker run -d --name aica -p 8443:8443 aica:image
```