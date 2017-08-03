
```bash
docker build -t escherpad/ml:1.0  .
```

```bash
# Dockerfile
RUN apt-get update && apt-get install -y\
    curl \
    vim \
    openjdk-7 jdk
```
