/************************************************************************************
***
***	Copyright 2024 Dell(18588220928g@163.com), All Rights Reserved.
***
***	File Author: Dell, Tue 30 Jan 2024 11:52:34 PM CST
***
************************************************************************************/

/************************************************************************************
Header only file for ggml engine

#define GGML_ENGINE_IMPLEMENTATION
#include <ggml_engine.h>
************************************************************************************/

#ifndef _GGML_ENGINE_H
#define _GGML_ENGINE_H

#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml.h>
#include <nimage/image.h> // IMAGE, TENSOR ...

#include <algorithm>
#include <unordered_map>
#include <vector>

#ifdef GGML_CUDA
#define GGML_USE_CUDA
#define GGML_USE_CUBLAS
#endif

#ifdef GGML_METAL
#define GGML_USE_METAL
#endif

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#ifdef GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include "ggml_nn.h"

#define ENGINE_VERSION "1.0.0"
#define MAX_INPUT_TENSORS 10

#define CheckPoint(fmt, arg...) printf("# CheckPoint: %d(%s): " fmt "\n", (int)__LINE__, __FILE__, ##arg)

// GGML Engine
struct GGMLEngine {
    // General
    int device = 0; // 0 -- CPU, 1 -- CUDA 0, 2 -- CUDA 1, ...
    int cpu_threads = 1;

    // Context
    ggml_context_t* inputs_context = NULL;
    ggml_context_t* weight_context = NULL;

    // Backend
    struct ggml_backend* backend = NULL;
    size_t backend_buffer_size = 0; // ONLY reference backend buffer size, not exactly  ...
    struct ggml_backend_buffer* inputs_backend_buffer = NULL;
    struct ggml_backend_buffer* weight_backend_buffer = NULL;

    // Model weight
    const char* model_name = "";
    const char* weight_prefix = "";

    // Graph
    void* graph_cpu_buffer = NULL;

    // Output tensors
    std::unordered_map<std::string, TENSOR*> output_tensors = {};
};

struct GGMLModel {
    int preload(const char* model_name);
    void remap(const char* oldkey, const char* newkey);
    void dump();
    void clear();

    char* file_name = NULL;
    std::unordered_map<std::string, int> name_maps;
    gguf_context* ctx_gguf_ = NULL;
    ggml_context* ctx_meta_ = NULL;
};

struct GGMLNetwork {
public:
    void dump();
    void set_device(int device) { m_ggml_engine.device = device; }
    bool load(const char* model_path, const char* prefix);
    int load_weight(GGMLModel* model, const char* prefix);

    int start_engine();
    TENSOR* engine_forward(int argc, TENSOR* argv[]);
    TENSOR* get_output_tensor(char* name);
    void stop_engine();

    virtual void create_weight_tensors(ggml_context_t* ctx) = 0;
    virtual void setup_weight_names(const char* prefix) = 0;
    virtual ggml_tensor_t* forward(ggml_context_t* ctx, int argc, ggml_tensor_t* argv[])
    {
        // GGML_UNUSED(ctx);
        GGML_UNUSED(argc);
        auto x = argv[0];
        return ggml_dup(ctx, x); // do not use 'return x;' directly !!!
    }
    virtual size_t get_graph_size()
    {
        return GGML_DEFAULT_GRAPH_SIZE; // 2048
    }

    virtual void create_input_tensors(int argc, TENSOR* argv[], ggml_context_t* ctx, ggml_tensor_t* x[])
    {
        char input_name[64];
        for (int i = 0; i < argc; i++) {
            if (argv[i] == NULL) {
                x[i] = NULL;
                continue;
            }

            snprintf(input_name, sizeof(input_name), "net.input_%d", i);
            x[i] = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, (int64_t)argv[i]->width, (int64_t)argv[i]->height,
                (int64_t)argv[i]->chan, (int64_t)argv[i]->batch);
            ggml_set_name(x[i], input_name);
        }
    }

    virtual void setup_input_values(int argc, TENSOR* argv[], bool is_cpu_backend, ggml_tensor_t* x[])
    {
        if (is_cpu_backend) {
            syslog_debug("Set input values to cpu backend ...");
        } else {
            syslog_debug("Set input values to cuda backend ...");
        }

        for (int i = 0; i < argc; i++) {
            if (argv[i] == NULL)
                continue;

            if (is_cpu_backend) {
                memcpy(x[i]->data, argv[i]->data, ggml_nbytes(x[i]));
            } else {
                ggml_backend_tensor_set(x[i], argv[i]->data, 0, ggml_nbytes(x[i]));
            }
        }
    }

protected:
    GGMLEngine m_ggml_engine = {};
    int m_network_init();
    struct ggml_cgraph* m_build_graph(int argc, ggml_tensor_t* argv[]);
    TENSOR* m_compute(int argc, ggml_tensor_t* argv[]);
    void m_clear_output_tensors();
};

int set_tensor_value(ggml_tensor_t* tensor, TENSOR* nt); // bool to_backend; // nt -- nimage tensor
TENSOR* get_tensor_value(ggml_tensor_t* tensor); // bool from_backend
void ggml_tensor_dump(char* prompt, ggml_tensor_t* tensor);

static int tensor_data_cast(
    ggml_tensor_t* s, void* s_data, ggml_tensor_t* d, void* d_data, std::vector<char> temp_buffer);

#endif // _GGML_ENGINE_H

// ----------------------------------------------------------------------------------
#ifdef GGML_ENGINE_IMPLEMENTATION
#ifdef GGML_METAL
#define GGML_USE_METAL
#include "ggml-metal.h"
#endif

#include "ggml-cuda.h"

#include <thread>
#include <vector>

#define for_each_context_tensor(ctx)                                                                                   \
    for (ggml_tensor_t* t = ggml_get_first_tensor(ctx); t != NULL; t = ggml_get_next_tensor(ctx, t))

static char* _find_model_path(const char* model_name);
static bool _same_data_shape(ggml_tensor_t* tensor, TENSOR* nt);

static struct ggml_backend* _device_backend_init(int device, int* ok_device);
static int _engine_backend_init(GGMLEngine* eng);
static bool _backend_is_cpu(struct ggml_backend* backend);

// --------------------------------------------------------------------------------------
static bool _backend_is_cpu(struct ggml_backend* backend)
{
    if (ggml_backend_is_cpu(backend)) {
        return true;
    }
#ifdef GGML_USE_METAL
    if (ggml_cpu_has_metal() && ggml_backend_is_metal(backend)) {
        return true;
    }
#endif
    return false;
}

int GGMLNetwork::start_engine()
{
    syslog_info("Start Engine (%s) ...", ENGINE_VERSION);

    GGMLEngine* eng = &m_ggml_engine;

    ggml_time_init(); // Nothing for linux but something on windows
    check_point(m_network_init() == RET_OK);
    check_point(_engine_backend_init(eng) == RET_OK);

    syslog_info("Start Engine OK.");

    return RET_OK;
}

void GGMLNetwork::stop_engine()
{
    GGMLEngine* eng = &m_ggml_engine;

    syslog_info("Stop Engine ...");
    // system("nvidia-smi");
    m_clear_output_tensors();

    // Clean backend
    {
        // if (eng->inputs_backend_buffer != NULL) {
        //     ggml_backend_buffer_free(eng->inputs_backend_buffer);
        // }

        if (eng->weight_backend_buffer != NULL) {
            ggml_backend_buffer_free(eng->weight_backend_buffer);
        }

        if (eng->backend != NULL) {
            ggml_backend_free(eng->backend);
        }
    }

    // Clean context
    {
        if (eng->inputs_context != NULL) {
            ggml_free(eng->inputs_context);
        }

        if (eng->weight_context != NULL) {
            ggml_free(eng->weight_context);
        }

        if (eng->graph_cpu_buffer) {
            free(eng->graph_cpu_buffer);
        }

        memset((void*)eng, 0, sizeof(GGMLEngine));
    }
    // system("nvidia-smi");
}

int GGMLNetwork::m_network_init()
{
    int num_tensors;
    GGMLEngine* eng = &m_ggml_engine;

    if (eng->weight_context) // do not repeat init ...
        return RET_OK;

    int64_t start_time = ggml_time_ms();

    // Set default threads
    eng->cpu_threads = 1; // std::thread::hardware_concurrency();
    // Get num of tensors and memoy size via temp context for more presion
    {
        // ggml_tensor_overhead() == 400
        struct ggml_init_params params = {
            /*.mem_size   =*/128 * 1024 * 1024, // 128M
            /*.mem_buffer =*/NULL,
            /*.no_alloc   =*/true, // the tensors no need to be allocated later
        };
        ggml_context_t* temp_context = ggml_init(params);
        check_point(temp_context);
        create_weight_tensors(temp_context);

        num_tensors = 0;
        eng->backend_buffer_size = 0;
        for_each_context_tensor(temp_context)
        {
            num_tensors++;
            eng->backend_buffer_size += ggml_nbytes(t);
        }
        ggml_free(temp_context);
    }

    // Create tensor and their memory on device cpu
    {
        size_t mem_size = ggml_tensor_overhead() * num_tensors;
        struct ggml_init_params params = {
            /*.mem_size   =*/mem_size,
            /*.mem_buffer =*/NULL,
            /*.no_alloc   =*/true, // the tensors will be allocated later
        };

        eng->weight_context = ggml_init(params);
        check_point(eng->weight_context != NULL);

        // eng->weight_context != NULL
        {
            create_weight_tensors(eng->weight_context);
            setup_weight_names(""); // prefix_name
        }
    }

    check_point(eng->weight_context != NULL);

    syslog_info("Network initialising spends %ld ms", ggml_time_ms() - start_time);

    return RET_OK;
}

static int _engine_backend_init(GGMLEngine* eng)
{
    check_point(eng->weight_context != NULL);

    // Create backend and backend buffer according to network ...
    {
        eng->backend = _device_backend_init(eng->device, &eng->device);
        check_point(eng->backend != NULL);

        eng->weight_backend_buffer = ggml_backend_alloc_ctx_tensors(eng->weight_context, eng->backend);
        // check_point(eng->weight_backend_buffer != NULL);
    }

    // Set CPU threads ...
    {
        if (ggml_backend_is_cpu(eng->backend)) {
            ggml_backend_cpu_set_n_threads(eng->backend, eng->cpu_threads);
        }
#ifdef GGML_USE_METAL
        if (ggml_cpu_has_metal() && ggml_backend_is_metal(eng->backend)) {
            ggml_backend_metal_set_n_cb(eng->backend, eng->cpu_threads);
        }
#endif
    }

    return RET_OK;
}

void GGMLNetwork::dump()
{
    check_avoid(m_ggml_engine.weight_context);

    syslog_info("Network information: ");
    syslog_info("  CPU threads: %d", m_ggml_engine.cpu_threads);
    syslog_info("  Backend device NO: %d, name: %s, buffer: %.2f MB", m_ggml_engine.device,
        ggml_backend_name(m_ggml_engine.backend), m_ggml_engine.backend_buffer_size / (1024.0 * 1024.0));
    syslog_info("  Weight model: [%s], prefix: [%s]", m_ggml_engine.model_name, m_ggml_engine.weight_prefix);

    syslog_info("Network tensors:");
    if (m_ggml_engine.inputs_context != NULL)
        for_each_context_tensor(m_ggml_engine.inputs_context) { ggml_tensor_dump((char*)"  ", t); }
    if (m_ggml_engine.weight_context != NULL)
        for_each_context_tensor(m_ggml_engine.weight_context) { ggml_tensor_dump((char*)"  ", t); }
}

bool GGMLNetwork::load(const char* model_name, const char* prefix)
{
    // Only check model exists ...
    char* model_filename = _find_model_path(model_name);
    if (model_filename) {
        free(model_filename);

        m_ggml_engine.model_name = model_name;
        m_ggml_engine.weight_prefix = prefix;

        return true;
    }

    return false;
}

int GGMLNetwork::load_weight(GGMLModel* model, const char* prefix)
{
    GGMLEngine* eng = &m_ggml_engine;
    check_point(eng != NULL);

    size_t prefix_len = strlen(prefix);
    bool cpu_backend = _backend_is_cpu(eng->backend);
    int64_t start_time = ggml_time_ms();

    char real_name[GGML_MAX_NAME];
    size_t read_nbytes;
    std::vector<char> read_buffer, temp_buffer, dest_buffer;

    if (model->file_name == NULL || model->name_maps.empty() || model->ctx_gguf_ == NULL || model->ctx_meta_ == NULL)
        return RET_OK;

    FILE* file = ggml_fopen(model->file_name, "rb");
    if (!file) {
        syslog_error("Open file '%s'", model->file_name);
        return RET_ERROR;
    }

    int n_loaded_tensors = 0;
    size_t start_data_offset = gguf_get_data_offset(model->ctx_gguf_);
    for_each_context_tensor(m_ggml_engine.weight_context)
    {
        snprintf(real_name, sizeof(real_name), "%s%s", prefix, t->name);

        auto it = model->name_maps.find(real_name);
        if (it == model->name_maps.end()) {
            syslog_info("Warnning: '%s' %s[%ld, %ld, %ld, %ld] NOT found in model weight.", real_name,
                ggml_type_name(t->type), t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
            continue;
        }
        int i = (int)it->second;
        char* name = gguf_get_tensor_name(model->ctx_gguf_, i);
        if (strlen(name) >= GGML_MAX_NAME) {
            syslog_error("name = '%s' too long (%ld)", name, strlen(name));
            continue;
        }

        ggml_tensor_t* s = ggml_get_tensor(model->ctx_meta_, name); // s -- come from file
        if (s == NULL) {
            syslog_error("Tensor of name = '%s' not found !", name);
            continue;
        }

        // match shape
        if (s->ne[0] != t->ne[0] || s->ne[1] != t->ne[1] || s->ne[2] != t->ne[2] || s->ne[3] != t->ne[3]) {
            syslog_error("%s shape mismatch: got [%ld, %ld, %ld, %ld], expected [%ld, %ld, %ld, %ld]", real_name,
                s->ne[0], s->ne[1], s->ne[2], s->ne[3], t->ne[0], t->ne[1], t->ne[2], t->ne[3]);
            continue;
        }

        // Read ok ?
        {
            read_nbytes = ggml_nbytes(s);
            read_buffer.reserve(read_nbytes); // fast than read_buffer.resize()
            fseek(file, start_data_offset + gguf_get_tensor_offset(model->ctx_gguf_, i), SEEK_SET);
            size_t n = fread(read_buffer.data(), 1, read_nbytes, file);
            if (n != read_nbytes) {
                syslog_error("Read '%s' data from '%s' ...", s->name, model->file_name);
                continue;
            }
        }

        if (t->type == s->type) { // fast set
            if (cpu_backend) {
                memcpy(t->data, read_buffer.data(), ggml_nbytes(t));
            } else {
                ggml_backend_tensor_set(t, read_buffer.data(), 0, ggml_nbytes(t));
            }
        } else {
            // slow convert: read_buffer (s->type) -->temp_buffer(t->type)<--- dest_buffer == t.data (t->type) ...
            dest_buffer.reserve(ggml_nbytes(t));
            if (tensor_data_cast(s, read_buffer.data(), t, dest_buffer.data(), temp_buffer) != RET_OK) {
                syslog_error("Cast data for '%s'", real_name);
                continue;
            }
            if (cpu_backend) {
                memcpy(t->data, dest_buffer.data(), ggml_nbytes(t));
            } else {
                ggml_backend_tensor_set(t, dest_buffer.data(), 0, ggml_nbytes(t));
            }
        }

        // syslog_debug("Loading %s ... OK", t->name);
        n_loaded_tensors++;
    }

    read_buffer.clear();
    temp_buffer.clear();
    dest_buffer.clear();

    fclose(file);
    syslog_info("Loading %d weights spends %ld ms", n_loaded_tensors, ggml_time_ms() - start_time);

    return RET_OK;
}

// *******************************************************************************
// build graph spends less 100us(0.1ms),
// so we build it every time for dynamic and simple logic
// *******************************************************************************
struct ggml_cgraph* GGMLNetwork::m_build_graph(int argc, ggml_tensor_t* argv[])
{
    CHECK_POINT(argc < MAX_INPUT_TENSORS);
    int64_t start_time = ggml_time_ms();

    size_t buf_size = ggml_tensor_overhead() * this->get_graph_size() + ggml_graph_overhead();
#if 0    
    if (m_ggml_engine.graph_cpu_buffer == NULL) {
        m_ggml_engine.graph_cpu_buffer = malloc(buf_size);
    }
#else
    m_ggml_engine.graph_cpu_buffer = realloc(m_ggml_engine.graph_cpu_buffer, buf_size);
#endif

    CHECK_POINT(m_ggml_engine.graph_cpu_buffer != NULL);
    memset((void*)m_ggml_engine.graph_cpu_buffer, 0, buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/buf_size,
        /*.mem_buffer =*/m_ggml_engine.graph_cpu_buffer,
        /*.no_alloc   =*/true,
    };

    // Create temp context to build the graph
    ggml_context_t* ctx = ggml_init(params0);
    CHECK_POINT(ctx != NULL);

    // struct ggml_cgraph* gf = ggml_new_graph(ctx);
    // struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, 2*GGML_DEFAULT_GRAPH_SIZE, false);
    struct ggml_cgraph* gf = ggml_new_graph_custom(ctx, this->get_graph_size(), false /*grads*/); // !!!!!!!!!
    CHECK_POINT(gf != NULL);
    // ggml_graph_clear(gf);

    ggml_tensor_t* result = this->forward(ctx, argc, argv);
    CHECK_POINT(result != NULL);

    ggml_build_forward_expand(gf, result);

    // Delete temp context
    ggml_free(ctx);
    syslog_info("Building graph spends %ld ms", ggml_time_ms() - start_time);

    return gf;
}

void GGMLNetwork::m_clear_output_tensors()
{
    for (auto& pair : m_ggml_engine.output_tensors) {
        TENSOR* t = (TENSOR*)pair.second;
        if (tensor_valid(t)) {
            tensor_destroy(t);
        }
    }
    m_ggml_engine.output_tensors.clear();
}

TENSOR* GGMLNetwork::m_compute(int argc, ggml_tensor_t* argv[])
{
    TENSOR* output = NULL;
    CHECK_POINT(argc < MAX_INPUT_TENSORS);

    // To support dynamic, build graph from scratch every time ...
    struct ggml_gallocr* compute_gallocr
        = ggml_gallocr_new(ggml_backend_get_default_buffer_type(m_ggml_engine.backend));
    CHECK_POINT(compute_gallocr != NULL);

    // Build graph begin !!!!!!!!!!!!!
    // --------------------------------------------------
    struct ggml_cgraph* gf = m_build_graph(argc, argv);
    CHECK_POINT(gf != NULL);
    CHECK_POINT(ggml_gallocr_alloc_graph(compute_gallocr, gf));
    // --------------------------------------------------
    // Build graph end !!!!!!!!!!!!!

    // Dump compute buffer size ...
    {
        size_t s = ggml_gallocr_get_buffer_size(compute_gallocr, 0);
        syslog_info("Compute backend buffer: %.2f M(%s)", s / 1024.0 / 1024.0,
            ggml_backend_is_cpu(m_ggml_engine.backend) ? "RAM" : "VRAM");
    }

    // Compute ...
    {
        ggml_backend_graph_compute(m_ggml_engine.backend, gf);
        ggml_backend_synchronize(m_ggml_engine.backend);

        // ggml_graph_print(gf);
        // ggml_tensor_t* y = gf->nodes[gf->n_nodes - 1];
        ggml_tensor_t* y = ggml_graph_node(gf, -1);
        CHECK_POINT(y != NULL);

        output = get_tensor_value(y); // true /*from_backend*/);

        // Save output tensors
#if 1
        for (int i = 0; i < ggml_graph_n_leafs(gf); i++) {
            // ggml_tensor_t* leaf = gf->leafs[i];
            ggml_tensor_t* leaf = ggml_graph_leaf(gf, i);
            if (leaf->flags & GGML_TENSOR_FLAG_OUTPUT) {
                // Saving leafs ...
                TENSOR* yt = get_tensor_value(leaf); // true /*from_backend*/);
                m_ggml_engine.output_tensors[std::string(leaf->name)] = yt;
            }
        }
#endif
        for (int i = 0; i < ggml_graph_n_nodes(gf) /*gf->n_nodes*/; i++) {
            // ggml_tensor_t* node = gf->nodes[i];
            ggml_tensor_t* node = ggml_graph_node(gf, i);
            if (node->flags & GGML_TENSOR_FLAG_OUTPUT) {
                // Saving nodes ...
                TENSOR* yt = get_tensor_value(node); // true /*from_backend*/);
                m_ggml_engine.output_tensors[std::string(node->name)] = yt;
            }
        }
    }

    ggml_gallocr_free(compute_gallocr);

    return output;
}

TENSOR* GGMLNetwork::get_output_tensor(char* name)
{
    auto it = m_ggml_engine.output_tensors.find(name);
    if (it == m_ggml_engine.output_tensors.end()) {
        syslog_info("Warnning: '%s' NOT found in output tensors.", name);
        return NULL;
    }
    TENSOR* output_tensor = (TENSOR*)it->second;

    return tensor_copy(output_tensor);
}

TENSOR* GGMLNetwork::engine_forward(int argc, TENSOR* argv[])
{
    ggml_tensor_t* x[MAX_INPUT_TENSORS];

    CHECK_POINT(argc < MAX_INPUT_TENSORS);

    int64_t start_time = ggml_time_ms();
    m_clear_output_tensors();

    // *******************************************************************************
    // Set input data spends less 2ms,
    // so we build it every time for dynamic and simple logic
    // *******************************************************************************
    // Set user input data
    {
        // Re-allocate input backend buffer for dynamic input shape ...
        {
            if (m_ggml_engine.inputs_context != NULL) {
                ggml_free(m_ggml_engine.inputs_context);
            }

            struct ggml_init_params params = {
                /*.mem_size   =*/ggml_tensor_overhead() * MAX_INPUT_TENSORS,
                /*.mem_buffer =*/NULL,
                /*.no_alloc   =*/true, // the tensors will not be allocated later
            };
            m_ggml_engine.inputs_context = ggml_init(params);
            CHECK_POINT(m_ggml_engine.inputs_context);
            // create_input_tensors template:
            // char input_name[64];
            // for (int i = 0; i < argc; i++) {
            //     if (argv[i] == NULL) {
            //         x[i] = NULL;
            //         continue;
            //     }

            //     snprintf(input_name, sizeof(input_name), "net.input_%d", i);
            //     x[i] = ggml_new_tensor_4d(m_ggml_engine.inputs_context, GGML_TYPE_F32,
            //         (int64_t)argv[i]->width, (int64_t)argv[i]->height,
            //         (int64_t)argv[i]->chan, (int64_t)argv[i]->batch);
            //     ggml_set_name(x[i], input_name);
            // }
            create_input_tensors(argc, argv, m_ggml_engine.inputs_context, x);

            // Backend ...
            if (m_ggml_engine.inputs_backend_buffer != NULL) {
                ggml_backend_buffer_free(m_ggml_engine.inputs_backend_buffer);
                m_ggml_engine.inputs_backend_buffer = NULL;
            }

            m_ggml_engine.inputs_backend_buffer
                = ggml_backend_alloc_ctx_tensors(m_ggml_engine.inputs_context, m_ggml_engine.backend);
            CHECK_POINT(m_ggml_engine.inputs_backend_buffer != NULL);
            // syslog_info("Set input spends %ld ms", ggml_time_ms() - start_time);
        }

        bool cpu_backend = _backend_is_cpu(m_ggml_engine.backend);
        // setup_input_values template:
        // for (int i = 0; i < argc; i++) {
        //     if (argv[i] == NULL)
        //         continue;

        //     if (cpu_backend) {
        //         memcpy(x[i]->data, argv[i]->data, ggml_nbytes(x[i]));
        //     } else {
        //         ggml_backend_tensor_set(x[i], argv[i]->data, 0, ggml_nbytes(x[i]));
        //     }
        // }
        setup_input_values(argc, argv, cpu_backend, x);
    }

    // Compute ...
    TENSOR* y = m_compute(argc, x); // y = m_compute(argc, x)

    syslog_info("Engine forward spends %ld ms", ggml_time_ms() - start_time);

    return y;
}

static char* _find_model_path(const char* model_name)
{
    // DO NOT forget to free if return != NULL !!!
    if (access(model_name, F_OK) == 0) {
        syslog_info("Found model '%s'.", model_name);
        return strdup(model_name);
    }

    // Try to find model under modes/
    char filename[512];
    snprintf(filename, sizeof(filename), "models/%s", model_name);
    if (access(filename, F_OK) == 0) {
        syslog_info("Found model '%s'.", filename);
        return strdup(filename);
    }

    syslog_error("Model '%s' NOT Found !!!", model_name);
    return NULL;
}

static struct ggml_backend* _device_backend_init(int device, int* ok_device)
{
    GGML_UNUSED(device);
#ifdef GGML_USE_CUDA
    if (device && ggml_cpu_has_cuda()) {
        struct ggml_backend* backend = ggml_backend_cuda_init(device - 1); // cuda 0 ...
        if (backend) {
            syslog_info("Using CUDA(%d) as Backend.", device - 1);
            *ok_device = device;
            return backend;
        }
    }
#endif

#ifdef GGML_USE_METAL
    if (ggml_cpu_has_metal()) {
        // ggml_metal_log_set_callback(ggml_log_callback_default, nullptr);
        struct ggml_backend* backend = ggml_backend_metal_init();
        if (backend) {
            syslog_info("Using Metal as Backend.");
            *ok_device = 0; // force set to cpu !!!
            return backend;
        }
    }
#endif

    // Fallback to CPU backend
    syslog_info("Using CPU as Backend.");
    *ok_device = 0; // force set to cpu !!!
    return ggml_backend_cpu_init();
}

void ggml_tensor_dump(char* prompt, ggml_tensor_t* tensor)
{
    char output_buffer[1024];

    if (tensor == NULL) {
        printf("%s tensor == NULL\n", prompt);
        return;
    }

    if (tensor->name) {
        snprintf(output_buffer, sizeof(output_buffer), "%6s [%ld, %ld, %ld, %ld], %s", ggml_type_name(tensor->type),
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], tensor->name);
    } else {
        snprintf(output_buffer, sizeof(output_buffer), "%6s [%ld, %ld, %ld, %ld], %s", ggml_type_name(tensor->type),
            tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3], "none-name");
    }

    printf("%s %s\n", prompt, output_buffer);
}

static bool _same_data_shape(ggml_tensor_t* tensor, TENSOR* nt)
{
    // B, C, H, W
    bool ok = (nt->batch == (int)tensor->ne[3] && nt->chan == (int)tensor->ne[2] && nt->height == (int)tensor->ne[1]
        && nt->width == (int)tensor->ne[0]);

    if (!ok) {
        syslog_error("Tensor shape: expect[%d, %d, %d, %d], got [%d, %d, %d, %d]", (int)tensor->ne[3],
            (int)tensor->ne[2], (int)tensor->ne[1], (int)tensor->ne[0], nt->batch, nt->chan, nt->height, nt->width);
        return false;
    }

    return ok;
}

TENSOR* get_tensor_value(ggml_tensor_t* tensor)
{
    CHECK_POINT(tensor);
    void* backend_data = NULL;

    GGML_ASSERT(ggml_is_contiguous(tensor));

    bool from_backend = !(tensor->buffer == NULL || ggml_backend_buffer_is_host(tensor->buffer));

    if (from_backend) {
        size_t nb = (size_t)ggml_nbytes(tensor);
        backend_data = (void*)malloc(nb);
        CHECK_POINT(backend_data != NULL);

        ggml_backend_tensor_get(tensor, backend_data, 0, nb);
    }

    // B, C, H, W
    TENSOR* nt = tensor_create((int)tensor->ne[3], (int)tensor->ne[2], (int)tensor->ne[1], (int)tensor->ne[0]);
    CHECK_TENSOR(nt);
    size_t n = nt->batch * nt->chan * nt->height * nt->width;

    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_I32) {
        if (tensor->type == GGML_TYPE_F32) {
            memcpy(nt->data, from_backend ? backend_data : tensor->data, n * sizeof(float));
        } else {
            int* src_data = from_backend ? (int*)backend_data : (int*)tensor->data;
            for (size_t i = 0; i < n; i++) {
                nt->data[i] = float(src_data[i]);
            }
        }

        // goto free_backend;
        if (from_backend)
            free(backend_data);
        return nt;
    }

    if (tensor->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((ggml_fp16_t*)(from_backend ? backend_data : tensor->data), (float*)nt->data, n);
        // goto free_backend;
        if (from_backend)
            free(backend_data);
        return nt;
    }

    // Dequantize src data to dst
    auto qtype = ggml_internal_get_type_traits(tensor->type);
    CHECK_POINT(qtype.to_float != NULL);
    qtype.to_float(from_backend ? backend_data : tensor->data, (float*)nt->data, n);

    // free_backend:
    if (from_backend)
        free(backend_data);

    return nt;
}

int set_tensor_value(ggml_tensor_t* tensor, TENSOR* nt) // , bool to_backend = false)
{
    check_point(tensor != NULL);
    check_tensor(nt);

    bool to_backend = (tensor->buffer != NULL && !ggml_backend_buffer_is_host(tensor->buffer));

    // B, C, H, W
    check_point(_same_data_shape(tensor, nt));
    size_t nb = ggml_nbytes(tensor);

    // 1) NOT convert data format ...
    if (tensor->type == GGML_TYPE_F32) {
        if (to_backend) {
            ggml_backend_tensor_set(tensor, nt->data, 0, nb);
        } else {
            memcpy(tensor->data, nt->data, nb);
        }
        return RET_OK;
    }

    // 2) Convert nt->data to tensor->data
    void* dst_data = (void*)malloc(nb);
    check_point(dst_data != NULL);
    if (tensor->type == GGML_TYPE_F16) {
        size_t n = tensor->ne[0] * tensor->ne[1] * tensor->ne[2] * tensor->ne[3];
        ggml_fp32_to_fp16_row((float*)nt->data, (ggml_fp16_t*)dst_data, n);
    } else {
        size_t nrows = (size_t)ggml_nrows(tensor);
        size_t n_per_row = ggml_nelements(tensor) / nrows; // tensor->ne[0]
        std::vector<float> matrix(n_per_row, 1.0f); // dummy importance matrix
        ggml_quantize_chunk(tensor->type, (float*)nt->data, dst_data, 0 /*start*/, nrows, n_per_row, matrix.data());
    }
    if (to_backend) {
        ggml_backend_tensor_set(tensor, dst_data, 0, nb);
    } else {
        memcpy(tensor->data, dst_data, nb);
    }
    free(dst_data);

    return RET_OK;
}

int GGMLModel::preload(const char* model_name)
{
    file_name = _find_model_path(model_name);
    check_point(file_name != NULL);

    syslog_info("Preloading weight from '%s' ...", file_name);

    int64_t start_time = ggml_time_ms();

    ctx_gguf_ = gguf_init_from_file(file_name, { true, &ctx_meta_ });
    check_point(ctx_gguf_ != NULL);

    int n_tensors = gguf_get_n_tensors(ctx_gguf_);
    for (int i = 0; i < n_tensors; i++) {
        std::string name = gguf_get_tensor_name(ctx_gguf_, i);
        name_maps[name] = i;
    }

    syslog_info("Preloading weight spends %ld ms", ggml_time_ms() - start_time);

    return RET_OK;
}

void GGMLModel::remap(const char* oldkey, const char* newkey)
{
    int klen = strlen(oldkey);
    if (klen == 0) // nothing to do ...
        return;

    std::unordered_map<std::string, int> new_maps;
    for (auto& p : name_maps) {
        std::string new_name(p.first);
        size_t pos = new_name.find(oldkey);
        if (pos != std::string::npos) {
            new_name.replace(pos, klen, newkey);
        }
        new_maps[new_name] = (int)p.second;
    }

    name_maps = std::move(new_maps);
}

void GGMLModel::dump()
{
    printf("File name: %s\n", file_name);
    for (auto& p : name_maps) { // name_maps -- map <name, i>
        char* name = gguf_get_tensor_name(ctx_gguf_, (int)p.second);
        ggml_tensor_t* d = ggml_get_tensor(ctx_meta_, name);
        if (d == NULL) {
            continue;
        }
        printf("%6s [%6ld, %6ld, %6ld, %6ld], %s\n", ggml_type_name(d->type), d->ne[0], d->ne[1], d->ne[2], d->ne[3],
            p.first.c_str());
    }
}

void GGMLModel::clear()
{
    name_maps.clear();
    if (file_name)
        free(file_name);
    if (ctx_gguf_)
        gguf_free(ctx_gguf_);
    if (ctx_meta_)
        ggml_free(ctx_meta_);
}

static int tensor_data_cast(
    ggml_tensor_t* s, void* s_data, ggml_tensor_t* d, void* d_data, std::vector<char> temp_buffer)
{
    // support s.dims == d.dims ..., here DO NOT using s->data, d->data !!!
    size_t n = s->ne[0] * s->ne[1] * s->ne[2] * s->ne[3];

    // Case 1: dequantize src data to dst
    if (d->type == GGML_TYPE_F32) {
        if (s->type == GGML_TYPE_F32) {
            memcpy(d_data, s_data, n * sizeof(float));
            return RET_OK;
        }
        if (s->type == GGML_TYPE_F16) {
            ggml_fp16_to_fp32_row((ggml_fp16_t*)s_data, (float*)d_data, n);
            return RET_OK;
        }

        // Dequantize src data to dst
        auto qtype = ggml_internal_get_type_traits(s->type);
        check_point(qtype.to_float != NULL);
        qtype.to_float(s_data, (float*)d_data, (int64_t)n);

        return RET_OK;
    }

    // Case 2: quantize float_to dst
    if (s->type == GGML_TYPE_F32) {
        // src->type == GGML_TYPE_F32 done before, so here src->type != GGML_TYPE_32 ...
        if (d->type == GGML_TYPE_F16) {
            ggml_fp32_to_fp16_row((float*)s_data, (ggml_fp16_t*)d_data, n);
            return RET_OK;
        }

        std::vector<float> matrix(d->ne[0] /*n_per_row*/, 1.0f); // dummy importance matrix
        ggml_quantize_chunk(d->type, (float*)s_data, d_data, 0 /*start*/, d->ne[1] * d->ne[2] * d->ne[3] /*nrows*/,
            d->ne[0] /*n_per_row*/, matrix.data());

        return RET_OK;
    }

    // Case 3: src --> float --> dst
    temp_buffer.reserve(n * sizeof(float));

    // Dequantize src to float
    if (s->type == GGML_TYPE_F16) {
        ggml_fp16_to_fp32_row((ggml_fp16_t*)s_data, (float*)temp_buffer.data(), n);
    } else {
        auto qtype = ggml_internal_get_type_traits(s->type);
        check_point(qtype.to_float != NULL);
        qtype.to_float(s_data, (float*)temp_buffer.data(), n);
    }

    // Quantize float to dst
    if (d->type == GGML_TYPE_F16) {
        ggml_fp32_to_fp16_row((float*)temp_buffer.data(), (ggml_fp16_t*)d_data, n);
    } else {
        std::vector<float> matrix(d->ne[0] /*n_per_row*/, 1.0f); // dummy importance matrix
        ggml_quantize_chunk(d->type, (float*)temp_buffer.data(), d_data, 0 /*start*/,
            d->ne[1] * d->ne[2] * d->ne[3] /*nrows*/, d->ne[0] /*n_per_row*/, matrix.data());
    }

    return RET_OK;
}

#endif // GGML_ENGINE_IMPLEMENTATION
