#include <onnxruntime_c_api.h>

#include <assert.h>
#include <errno.h>
#include <netdb.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define SERVER_PORT "31648"
#define LISTEN_BACKLOG 1
#define MAX_BUFSIZE (16 * 1024 * 1024)

#define LOG_LEVEL_DEBUG 0
#define LOG_LEVEL_INFO 1
#define LOG_LEVEL_SILENT 10000
#define LOG_LEVEL_WIP 10001

#define LOG_LEVEL LOG_LEVEL_SILENT

#define max(a, b) ((a) >= (b) ? (a) : (b))

#define die(fmt, ...) do { \
        fprintf(stderr, "[%d]: " fmt "\n", __LINE__ __VA_OPT__(,) __VA_ARGS__); \
        exit(1); \
    } while (0)

#define log(level, fmt, ...) do { \
        if (LOG_LEVEL <= level) \
            fprintf(stderr, "SERVER: " fmt "\n" __VA_OPT__(,) __VA_ARGS__); \
        fflush(stderr); \
    } while (0)

#define debug(fmt, ...) log(LOG_LEVEL_DEBUG, fmt __VA_OPT__(,) __VA_ARGS__)
#define info(fmt, ...)  log(LOG_LEVEL_INFO,  fmt __VA_OPT__(,) __VA_ARGS__)
#define wip(fmt, ...)  log(LOG_LEVEL_WIP,  fmt __VA_OPT__(,) __VA_ARGS__)

#define check(cond) do { \
        if (!(cond)) { \
            int _check_errno = errno; \
            const char *_check_err_msg = strerror(_check_errno); \
            die("assert fail with error %d: %s", _check_errno, _check_err_msg); \
        } \
    } while (0)

#define check_msg(cond, ...) do { \
        if (!(cond)) \
            die(__VA_ARGS__); \
    } while (0)

#define check_ort(status_ptr) do { \
        if ((status_ptr) == NULL) \
            break; \
        die("ort status error message: %s", get_ort_api()->GetErrorMessage(status_ptr)); \
    } while (0)

#define ort(func, ...) \
    check_ort(get_ort_api()->func(__VA_ARGS__))

#define ort_release(func, ptr) \
    get_ort_api()->Release##func(ptr)

static bool do_measure_time = true;

static const OrtApi* get_ort_api() {
    return OrtGetApiBase()->GetApi(ORT_API_VERSION);
}

typedef struct node_info {
    const char* name;
    size_t nelems;
    ONNXTensorElementDataType dtype;
    size_t elem_size;
    size_t ndim;
    const int64_t* dims;
} NodeInfo;

typedef struct ctx {
    OrtSession *ort_session;
    NodeInfo model_in;
    NodeInfo model_out;
} Ctx;

typedef struct client_session {
    int sock;

    size_t batch_size;

    uint8_t *buf;
    size_t bufsize;

    uint64_t batch_total_ms;
    uint64_t conn_total_ms;
    uint64_t empty_run_ms;
    uint64_t ort_run_ms;
    uint64_t recv_ms;
    uint64_t send_ms;
} ClientSession;

static inline uint64_t micros() {
    if (do_measure_time) {
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
        return ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
    } else {
        return 0;
    }
}

/// return true if could read all len bytes
static bool read_exact(int fd, void *buf, size_t len) {
    debug("Waiting for exactly %zu bytes", len);
    size_t tot_read = 0;
    while (tot_read < len) {
        ssize_t r = read(fd, buf + tot_read, len - tot_read);
        if (r <= 0)
            return false;
        tot_read += r;
    }
    return true;
}

/// return true if could send all len bytes
static bool write_exact(int fd, const void *buf, size_t len) {
    debug("Sending exactly %zu bytes", len);
    size_t tot_written = 0;
    while (tot_written < len) {
        ssize_t r = write(fd, buf + tot_written, len - tot_written);
        debug("write() -> %zd", r);
        if (r <= 0)
            return false;
        tot_written += r;
    }
    return true;
}

static size_t product(int64_t *factors, size_t nfactor) {
    uint64_t prod = 1;
    for (size_t i = 0; i < nfactor; ++i) {
        int64_t f = factors[i];
        check_msg(f > 0, "Non-positive factor");
        prod *= (uint64_t) f;
    }
    return (size_t) prod;
}

static int server_bind(struct addrinfo *server_addr_out) {
    int r;
    struct addrinfo hints;
    struct addrinfo *res;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE;
    r = getaddrinfo(NULL, SERVER_PORT, &hints, &res);
    check_msg(r == 0, "getaddrinfo error: %s", gai_strerror(r));
    for (struct addrinfo *ai = res; ai != NULL; ai = ai->ai_next) {
        int sock = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (sock < 0)
            continue; // try another addrinfo
        r = bind(sock, ai->ai_addr, ai->ai_addrlen);
        if (r != 0)
            continue; // try another addrinfo
        memcpy(server_addr_out, ai, sizeof(*server_addr_out));
        freeaddrinfo(res);
        return sock;
    }
    die("Could not bind server socket");
}

static void infer_one(
    const Ctx *ctx,
    ClientSession *cs,
    const OrtMemoryInfo *memory_info,
    void *in_data,
    size_t insize,
    void *out_data,
    size_t outsize
) {
    int is_tensor;
    uint64_t ms;

    OrtValue *input_tensor = NULL;
    // This does NOT copy in data
    ort(
        CreateTensorWithDataAsOrtValue,
        memory_info,
        in_data,
        insize,
        ctx->model_in.dims,
        ctx->model_in.ndim,
        ctx->model_in.dtype,
        &input_tensor
    );
    ort(IsTensor, input_tensor, &is_tensor);
    check_msg(is_tensor, "Created input tensor isn't tensor");

    const char* const* input_names = &ctx->model_in.name;
    const char* const* output_names = &ctx->model_out.name;

    ms = micros();
    ms = micros() - ms;
    cs->empty_run_ms += ms;

    // Run input through model
    ms = micros();
    OrtValue* output_tensor = NULL;
    ort(
        Run,
        ctx->ort_session,
        NULL,
        input_names,
        (const OrtValue * const*) &input_tensor,
        1,
        output_names,
        1,
        &output_tensor
    );
    ort(IsTensor, output_tensor, &is_tensor);
    ms = micros() - ms;
    cs->ort_run_ms += ms;
    check_msg(is_tensor, "Model output is not tensor");
    debug("Got an output tensor from ort!");

    // Release input tensor befory copying output tensor to out_data in case they
    // overlap.
    ort_release(Value, input_tensor);

    float *ort_out_data;
    ort(GetTensorMutableData, output_tensor, (void**) &ort_out_data);
    memcpy(out_data, ort_out_data, outsize);

    ort_release(Value, output_tensor);
}

static bool wait_til_data(int sock, uint64_t *ms_out) {
    char buf;
    ssize_t ret = recv(sock, &buf, 1, MSG_PEEK);
    *ms_out = micros();
    return ret == 1;
}

static bool handle_one_batch(const Ctx *ctx, ClientSession *cs) {
    const size_t req_id_size = 4;

    bool still_conn;
    uint8_t *data_buf;
    uint64_t ms1, ms2;

    size_t in_data_size = ctx->model_in.nelems * ctx->model_in.elem_size;
    size_t out_data_size = ctx->model_out.nelems * ctx->model_out.elem_size;

    const size_t bs = cs->batch_size;

    uint64_t batch_beg;
    still_conn = wait_til_data(cs->sock, &batch_beg);
    if (!still_conn)
        return false;

    // Read entire request
    size_t to_read = req_id_size + bs * in_data_size;
    ms1 = micros();
    still_conn = read_exact(cs->sock, cs->buf, to_read);
    ms2 = micros();
    cs->recv_ms += ms2 - ms1;
    if (!still_conn)
        return false;

    // The beginning of buf will stay same for input and output.
    // The rest holds data for input request now and will hold response data later.
    data_buf = cs->buf + req_id_size;

    // Using same buf for in and out.
    // As each input is handled and produces an output, output will overwrite old input.
    // If output size was bigger it would overwrite future input.
    assert(in_data_size >= out_data_size);

    OrtMemoryInfo *memory_info;
    ort(CreateCpuMemoryInfo, OrtArenaAllocator, OrtMemTypeDefault, &memory_info);

    for (size_t i = 0; i < bs; ++i) {
        // TODO: dtype assumed here
        float *in_data = (float*) (data_buf + in_data_size * i);
        float *out_data = (float*) (data_buf + out_data_size * i);
        infer_one(ctx, cs, memory_info, in_data, in_data_size, out_data, out_data_size);
    }

    ort_release(MemoryInfo, memory_info);

    // Send back entire response
    size_t to_send = req_id_size + bs * out_data_size;
    ms1 = micros();
    still_conn = write_exact(cs->sock, cs->buf, to_send);
    ms2 = micros();
    uint64_t batch_end = ms2;
    cs->send_ms += ms2 - ms1;

    cs->batch_total_ms += batch_end - batch_beg;

    return still_conn;
}

static void log_line(ClientSession *cs) {
    // NOTE: This assumes single thread
    static bool printed_header = false;
    if (!printed_header) {
        fprintf(
            stdout,
            "%s,%s,%s,%s,%s,%s\n",
            "batch_total",
            "conn_total",
            "empty_run",
            "ort_run",
            "recv",
            "send"
        );
        printed_header = true;
    }
    fprintf(
        stdout,
        "%f,%f,%f,%f,%f,%f\n",
        cs->batch_total_ms / 1e6,
        cs->conn_total_ms / 1e6,
        cs->empty_run_ms / 1e6,
        cs->ort_run_ms / 1e6,
        cs->recv_ms / 1e6,
        cs->send_ms / 1e6
    );
    fflush(stdout);
}

static void handle_conn(const Ctx *ctx, int sock) {
    bool still_conn;
    uint64_t ms1, ms2;
    ms1 = micros();
    info("Got connection");

    uint32_t batch_size;
    still_conn = read_exact(sock, &batch_size, sizeof(batch_size));
    if (!still_conn) {
        info("Connection closed");
        return;
    }
    batch_size = ntohl(batch_size);

    size_t insize = ctx->model_in.nelems * ctx->model_in.elem_size;
    size_t outsize = ctx->model_out.nelems * ctx->model_out.elem_size;
    size_t bufsize = sizeof(uint32_t) + batch_size * max(insize, outsize);
    info("Allocating %zu byte buffer", bufsize);
    assert(bufsize <= MAX_BUFSIZE);
    void *buf = malloc(bufsize);
    check_msg(buf != NULL, "Could not alloc buffer for connection session");

    ClientSession cs = {
        .sock = sock,
        .batch_size = batch_size,
        .bufsize = bufsize,
        .buf = buf,
        .bufsize = 0,
        .batch_total_ms = 0,
        .conn_total_ms = 0,
        .empty_run_ms = 0,
        .ort_run_ms = 0,
        .recv_ms = 0,
        .send_ms = 0,
    };

    while (handle_one_batch(ctx, &cs));
    free(buf);
    ms2 = micros();
    info("Connection closed");
    close(cs.sock);
    
    cs.conn_total_ms = ms2 - ms1;

    log_line(&cs);
}

static void get_node_info(
    NodeInfo *ni_out,
    OrtSession *session,
    OrtTypeInfo *type_info,
    const char *name
) {
    const OrtTensorTypeAndShapeInfo *tensor_info;
    // TODO: Assuming the single input is tensor for now.
    ort(CastTypeInfoToTensorInfo, type_info, &tensor_info);
    ONNXTensorElementDataType dtype;
    ort(GetTensorElementType, tensor_info, &dtype);
    // TODO: Assuming float for now
    assert(dtype == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    size_t ndim;
    ort(GetDimensionsCount, tensor_info, &ndim);
    int64_t *dims = malloc(sizeof(int64_t) * ndim);
    ort(GetDimensions, tensor_info, dims, ndim);
    
    ni_out->name = name;
    ni_out->nelems = product(dims, ndim);
    ni_out->dtype = dtype;
    ni_out->elem_size = sizeof(float); // TODO: hardcoded for now
    ni_out->ndim = ndim;
    ni_out->dims = dims;
}

static void setup_ctx(Ctx *ctx_out, OrtSession *session, OrtAllocator* alloc) {
    ctx_out->ort_session = session;

    // TODO: Assuming for now that only one input and one output node

    OrtTypeInfo *type_info;
    char *name;

    const size_t input_idx = 0;
    ort(SessionGetInputTypeInfo, session, input_idx, &type_info);
    ort(SessionGetInputName, session, input_idx, alloc, &name);
    get_node_info(&ctx_out->model_in, session, type_info, name);
    ort_release(TypeInfo, type_info);

    const size_t output_idx = 0;
    ort(SessionGetOutputTypeInfo, session, output_idx, &type_info);
    ort(SessionGetOutputName, session, output_idx, alloc, &name);
    get_node_info(&ctx_out->model_out, session, type_info, name);
    ort_release(TypeInfo, type_info);
}

int main(int argc, char **argv) {
    assert(argc == 3);

    const char *measure_time_flag = argv[1];
    const char *model_path = argv[2];

    if (strcmp(measure_time_flag, "time") == 0)
        do_measure_time = true;
    else if (strcmp(measure_time_flag, "notime") == 0)
        do_measure_time = false;
    else
        die("First argument must be time or notime");

    debug("Ort api version: %d", ORT_API_VERSION);

    OrtEnv *env;
    ort(CreateEnv, ORT_LOGGING_LEVEL_WARNING, "inference_server", &env);
    debug("Created env.");

    GraphOptimizationLevel opt_level = ORT_ENABLE_BASIC;

    OrtSessionOptions *session_opts;
    ort(CreateSessionOptions, &session_opts);
    ort(SetSessionGraphOptimizationLevel, session_opts, opt_level);
    debug("Set some options.");

    OrtSession *session;
    ort(CreateSession, env, model_path, session_opts, &session);
    debug("Created session with model.");

    size_t ninputs, noutputs;
    ort(SessionGetInputCount, session, &ninputs);
    ort(SessionGetOutputCount, session, &noutputs);
    assert(ninputs == 1); // TODO
    assert(noutputs == 1); // TODO

    OrtAllocator *alloc;
    ort(GetAllocatorWithDefaultOptions, &alloc);

    Ctx ctx;
    setup_ctx(&ctx, session, alloc);

    struct addrinfo listen_addr;
    int listen_sock = server_bind(&listen_addr);
    debug("Bound listening socket");

    int r = listen(listen_sock, LISTEN_BACKLOG);
    check(r == 0);
    debug("Set to listen");

    struct sockaddr_storage peer_addr;
    socklen_t peer_addr_size = sizeof(peer_addr);
    info("Ready for first connection");
    for (;;) {
        debug("Waiting for connection");
        int peer_sock =
            accept(listen_sock, (struct sockaddr*) &peer_addr, &peer_addr_size);
        check(peer_sock >= 0);
        handle_conn(&ctx, peer_sock);
    }

    alloc->Free(alloc, (char**) ctx.model_in.name);
    alloc->Free(alloc, (char**) ctx.model_out.name);

    free((char*) ctx.model_in.dims);
    free((char*) ctx.model_out.dims);
    ort_release(SessionOptions, session_opts);
    ort_release(Env, env);

    debug("Bye.");

}
