.PHONY += all
all: bin/inference-server

bin/inference-server: inference-server.c | bin
	gcc -o "$@" -Wall -Werror -O3 "$<" -lonnxruntime

bin:
	mkdir "$@"

.PHONY += clean
clean:
	rm -rf bin
