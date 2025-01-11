/************************************************************************************
***
*** Copyright 2024 Dell Du(18588220928@163.com), All Rights Reserved.
***
*** File Author: Dell, Tue 02 Apr 2024 03:49:53 PM CST
***
************************************************************************************/

#include "yolox.h"

#define GGML_ENGINE_IMPLEMENTATION
#include <ggml_engine.h>
#define GGML_NN_IMPLEMENTATION
#include <ggml_nn.h>

#include <sys/stat.h> // for chmod
#include <vector>
#define SCORE_THRESHOLD 0.60

typedef struct {
    float s, x, y, w, h; // s -- score
    int index;
} ScoreBox;
static float overlap(float x1, float w1, float x2, float w2);
static float box_iou(const ScoreBox& a, const ScoreBox& b);
static int decode_detect_result(int h, int w, TENSOR* detect_result);
static void visual_detect_result(IMAGE *image, TENSOR* detect_result);
// -------------------------------------------------------------------------------

static float overlap(float x1, float w1, float x2, float w2)
{
    float l1 = x1 - w1 / 2;
    float l2 = x2 - w2 / 2;
    float left = l1 > l2 ? l1 : l2;
    float r1 = x1 + w1 / 2;
    float r2 = x2 + w2 / 2;
    float right = r1 < r2 ? r1 : r2;
    return right - left;
}

static float box_iou(const ScoreBox& a, const ScoreBox& b)
{
    float w = overlap(a.x, a.w, b.x, b.w);
    float h = overlap(a.y, a.h, b.y, b.h);
    if (w < 0 || h < 0)
        return 0.0;
    float i = w * h;
    float u = a.w * a.h + b.w * b.h - i + 1e-5;

    return i / u;
}

static void nms_sort(std::vector<ScoreBox>& detboxes, float thresh)
{
    size_t total = detboxes.size();
    std::sort(detboxes.begin(), detboxes.begin() + total, [=](const ScoreBox& a, const ScoreBox& b) {
        return a.s > b.s;
    });
    for (size_t i = 0; i < total; ++i) {
        if (detboxes[i].s < 0.01f)
            continue;

        ScoreBox a = detboxes[i];
        for (size_t j = i + 1; j < total; ++j) {
            ScoreBox b = detboxes[j];
            if (box_iou(a, b) > thresh)
                detboxes[j].s = 0.0f;
        }
    }
}

static int decode_detect_result(int h, int w, TENSOR* detect_result)
{
    int n_detected = 0;

    std::vector<ScoreBox> det_boxes;
    for (int i = 0; i < detect_result->height; i++) {
        float* row = tensor_start_row(detect_result, 0, 0, i);

        row[0] = MAX(row[0], 0.0f);
        row[1] = MAX(row[1], 0.0f);
        row[2] = MIN(row[2], (float)w - 1.0);
        row[3] = MIN(row[3], (float)h - 1.0);

        ScoreBox box;
        box.x = row[0];
        box.y = row[1];
        box.w = row[2] - row[0]; // x2 - x1
        box.h = row[3] - row[1]; // y2 - y1
        box.s = row[4]; // obj_score * class_score

        box.index = i;
        if (box.s < SCORE_THRESHOLD || box.w < 0.5 || box.h < 0.5)
            continue;

        det_boxes.push_back(box);
    }

#define NMS_THRESHOLD 0.45
    nms_sort(det_boxes, NMS_THRESHOLD);
#undef NMS_THRESHOLD
    
    // Update detect result score
    for (int i = 0; i < detect_result->height; i++) {
        float* row = tensor_start_row(detect_result, 0, 0, i);
        row[4] = 0.0; // clear all scores
    }
    for (auto it = det_boxes.begin(); it != det_boxes.end(); it++) {
        if (it->s >= SCORE_THRESHOLD) {
            n_detected++;
            float* row = tensor_start_row(detect_result, 0, 0, it->index);
            row[4] = it->s; // update score
        }
    }
    det_boxes.clear();

    return n_detected;
}

static void visual_detect_result(IMAGE *image, TENSOR* detect_result)
{
    RECT rect;
    for (int i = 0; i < detect_result->height; i++) {
        float* row = tensor_start_row(detect_result, 0, 0, i);

        rect.c = (int)row[0];
        rect.r = (int)row[1];
        rect.w = (int)row[2] - (int)row[0];
        rect.h = (int)row[3] - (int)row[1];

        float class_score = row[4];
        int class_id = (int)row[5];

        if (class_score < SCORE_THRESHOLD)
            continue;

        image_drawrect(image, &rect, 0xff0000, 0);
        // image_drawline(image, y1, x1, y2, x2, 0xffffff); // white color
    }
}


int image_detect_predict(YOLOXNetwork* detect_net, char* input_file, char* output_dir)
{
    TENSOR* input_tensor;
    char *p, output_filename[512];

    p = strrchr(input_file, '/');
    if (p != NULL) {
        snprintf(output_filename, sizeof(output_filename), "%s/%s", output_dir, p + 1);
    } else {
        snprintf(output_filename, sizeof(output_filename), "%s/%s", output_dir, input_file);
    }
    printf("Detect %s to %s ...\n", input_file, output_filename);

    input_tensor = tensor_load_image(input_file, 0 /*with_alpha*/);
    check_tensor(input_tensor);

    TENSOR* detect_result = detect_net->forward(input_tensor); // 1x1xnx16
    check_tensor(detect_result);
    int n = decode_detect_result(input_tensor->height, input_tensor->width, detect_result);
    // tensor [output_tensor] size: [1, 8400, 6], min: -51.966671, max: 866.434814, mean: 194.278854

    IMAGE* image = image_from_tensor(input_tensor, 0/*batch*/);
    // tensor_saveas_image(input_tensor, 0 /*batch 0*/, output_filename);
    if (n > 0) {
        visual_detect_result(image, detect_result);
    }
    image_save(image, output_filename);
    chmod(output_filename, 0644);
    image_destroy(image);

    tensor_destroy(detect_result);
    tensor_destroy(input_tensor);

    return RET_OK;
}
