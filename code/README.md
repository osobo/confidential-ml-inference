Build with `make`.

Example command to run:

```
docker run --device /dev/isgx -it cmli-occlum-inference-mnist time 16 32 64
```

The above command will run 16 times (connections) with batch size 32 and with 64 batches per connection.
