#ifndef __YOLOX__H__
#define __YOLOX__H__
#include "ggml_engine.h"
#include "ggml_nn.h"

#include <vector>

#pragma GCC diagnostic ignored "-Wformat-truncation"


struct BaseConv {
    int in_channels;
    int out_channels;
    int ksize;
    int stride;

    // network hparams
    struct Conv2d conv;
    struct BatchNorm2d bn;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv.in_channels = in_channels;
        conv.out_channels = out_channels;
        conv.kernel_size = { ksize, ksize };
        conv.stride = { stride, stride };
        conv.padding = { (ksize - 1)/2, (ksize - 1)/2 };
        conv.has_bias = false;
        conv.create_weight_tensors(ctx);

        bn.num_features = out_channels;
        bn.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
 
        snprintf(s, sizeof(s), "%s%s", prefix, "conv.");
        conv.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "bn.");
        bn.setup_weight_names(s);
    }


    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        x = conv.forward(ctx, x);
        x = bn.forward(ctx, x);
        x = ggml_silu(ctx, x);

        return x;
    }
};


struct SPPBottleneck {
    int in_channels;
    int out_channels;

    // network params
    struct BaseConv conv1;
    struct BaseConv conv2;
    struct MaxPool2d m[3];

    void create_weight_tensors(struct ggml_context* ctx) {
        conv1.in_channels = in_channels;
        conv1.out_channels = in_channels/2;
        conv1.ksize = 1;
        conv1.stride = 1;
        conv1.create_weight_tensors(ctx);

        // kernel_sizes=(5, 9, 13);
        m[0].kernel_size = 5;
        m[0].stride = 1;
        m[0].padding = 2;
        m[0].create_weight_tensors(ctx);

        m[1].kernel_size = 9;
        m[1].stride = 1;
        m[1].padding = 4;
        m[1].create_weight_tensors(ctx);

        m[2].kernel_size = 13;
        m[2].stride = 1;
        m[2].padding = 6;
        m[2].create_weight_tensors(ctx);

        conv2.in_channels = (in_channels/2) * 4;
        conv2.out_channels = out_channels;
        conv2.ksize = 1;
        conv2.stride = 1;
        conv2.create_weight_tensors(ctx);        
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv1.");
        conv1.setup_weight_names(s);

        for (int i = 0; i < 3; i++) {
            snprintf(s, sizeof(s), "%sm.%d", prefix, i);
            m[i].setup_weight_names(s);
        }

        snprintf(s, sizeof(s), "%s%s", prefix, "conv2.");
        conv2.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // x = self.conv1(x)
        // x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        // x = self.conv2(x)
        // return x
        ggml_tensor_t *y1, *y2, *y3;
        x = conv1.forward(ctx, x);
        y1 = m[0].forward(ctx, x);
        y2 = m[1].forward(ctx, x);
        y3 = m[2].forward(ctx, x);
        x = ggml_cat(ctx, 4, x, y1, y2, y3, 2 /*dim on C*/);
        x = conv2.forward(ctx, x);

        return x;
    }
};

struct Bottleneck {
    int in_channels;
    int out_channels;
    bool shortcut = true;

    // network params
    struct BaseConv conv1;
    struct BaseConv conv2;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv1.in_channels = in_channels;
        conv1.out_channels = out_channels;
        conv1.ksize = 1;
        conv1.stride = 1;
        conv1.create_weight_tensors(ctx);

        conv2.in_channels = out_channels;
        conv2.out_channels = out_channels;
        conv2.ksize = 3;
        conv2.stride = 1;
        conv2.create_weight_tensors(ctx);        
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "conv1.");
        conv1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "conv2.");
        conv2.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // self.use_add = shortcut and in_channels == out_channels
        // y = self.conv2(self.conv1(x))
        // if self.use_add:
        //     y = y + x
        // return y
        ggml_tensor_t *y;
        y = conv2.forward(ctx, conv1.forward(ctx, x));
        if (shortcut && in_channels == out_channels) {
            y = ggml_add(ctx, y, x);
        }

        return y;
    }
};

struct YOLOXHead {
    int num_classes = 80;
    const int in_channels[3] = {256, 512, 1024};

    // network params
    struct BaseConv stems[3];
    struct BaseConv cls_convs[3][2];
    struct BaseConv reg_convs[3][2];

    struct Conv2d cls_preds[3];
    struct Conv2d reg_preds[3];
    struct Conv2d obj_preds[3];


    ggml_tensor_t *grids;
    ggml_tensor_t *strides;


    void create_weight_tensors(struct ggml_context* ctx) {
        for (int i = 0; i < 3; i++) {
            stems[i].in_channels = in_channels[i];
            stems[i].out_channels = 256;
            stems[i].ksize = 1;
            stems[i].stride = 1;
            stems[i].create_weight_tensors(ctx);
        }

        for (int i = 0; i < 3; i++) {
            cls_convs[i][0].in_channels = 256;
            cls_convs[i][0].out_channels = 256;
            cls_convs[i][0].ksize = 3;
            cls_convs[i][0].stride = 1;
            cls_convs[i][0].create_weight_tensors(ctx);

            cls_convs[i][1].in_channels = 256;
            cls_convs[i][1].out_channels = 256;
            cls_convs[i][1].ksize = 3;
            cls_convs[i][1].stride = 1;
            cls_convs[i][1].create_weight_tensors(ctx);
        }

        for (int i = 0; i < 3; i++) {
            reg_convs[i][0].in_channels = 256;
            reg_convs[i][0].out_channels = 256;
            reg_convs[i][0].ksize = 3;
            reg_convs[i][0].stride = 1;
            reg_convs[i][0].create_weight_tensors(ctx);

            reg_convs[i][1].in_channels = 256;
            reg_convs[i][1].out_channels = 256;
            reg_convs[i][1].ksize = 3;
            reg_convs[i][1].stride = 1;
            reg_convs[i][1].create_weight_tensors(ctx);
        }

        for (int i = 0; i < 3; i++) {
            cls_preds[i].in_channels = 256;
            cls_preds[i].out_channels = num_classes;
            cls_preds[i].kernel_size = { 1, 1 };
            cls_preds[i].stride = { 1, 1 };
            cls_preds[i].padding = { 0, 0 };
            cls_preds[i].create_weight_tensors(ctx);        
        }

        for (int i = 0; i < 3; i++) {
            reg_preds[i].in_channels = 256;
            reg_preds[i].out_channels = 4;
            reg_preds[i].kernel_size = { 1, 1 };
            reg_preds[i].stride = { 1, 1 };
            reg_preds[i].padding = { 0, 0 };
            reg_preds[i].create_weight_tensors(ctx);        
        }

        for (int i = 0; i < 3; i++) {
            obj_preds[i].in_channels = 256;
            obj_preds[i].out_channels = 1;
            obj_preds[i].kernel_size = { 1, 1 };
            obj_preds[i].stride = { 1, 1 };
            obj_preds[i].padding = { 0, 0 };
            obj_preds[i].create_weight_tensors(ctx);        
        }

        // # tensor [grids] size: [1, 8400, 2], min: 0.0, max: 79.0, mean: 34.261906
        // # tensor [strides] size: [1, 8400, 1], min: 8.0, max: 32.0, mean: 10.666667
        grids = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 2, 8400, 1);
        strides = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 8400, 1);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        for (int i = 0; i < 3; i++) {
            snprintf(s, sizeof(s), "%sstems.%d.", prefix, i);
            stems[i].setup_weight_names(s);
        }

        for (int i = 0; i < 3; i++) {
            snprintf(s, sizeof(s), "%scls_convs.%d.0.", prefix, i);
            cls_convs[i][0].setup_weight_names(s);

            snprintf(s, sizeof(s), "%scls_convs.%d.1.", prefix, i);
            cls_convs[i][1].setup_weight_names(s);
        }

        for (int i = 0; i < 3; i++) {
            snprintf(s, sizeof(s), "%sreg_convs.%d.0.", prefix, i);
            reg_convs[i][0].setup_weight_names(s);

            snprintf(s, sizeof(s), "%sreg_convs.%d.1.", prefix, i);
            reg_convs[i][1].setup_weight_names(s);
        }

        for (int i = 0; i < 3; i++) {
            snprintf(s, sizeof(s), "%scls_preds.%d.", prefix, i);
            cls_preds[i].setup_weight_names(s);
        }
        for (int i = 0; i < 3; i++) {
            snprintf(s, sizeof(s), "%sreg_preds.%d.", prefix, i);
            reg_preds[i].setup_weight_names(s);
        }
        for (int i = 0; i < 3; i++) {
            snprintf(s, sizeof(s), "%sobj_preds.%d.", prefix, i);
            obj_preds[i].setup_weight_names(s);
        }

        ggml_format_name(grids, "%s%s", prefix, "grids");
        ggml_format_name(strides, "%s%s", prefix, "strides");
    }

    ggml_tensor_t* decode_outputs(struct ggml_context* ctx, ggml_tensor_t* outputs) {
        // # tensor [outputs] size: [1, 8400, 85], min: -2.040908, max: 3.459077, mean: 0.055966

        // # (x1, y1, w2, h2, obj_score, class_scores ...)
        // # ==> (x1, y1, x2, y2, obj_score * class_score, class_id)

        ggml_tensor_t *x1y1 = ggml_slice(ctx, outputs, 0 /*dim*/, 0, 2, 1/*step*/);
        ggml_tensor_t *w2h2 = ggml_slice(ctx, outputs, 0 /*dim*/, 2, 4, 1/*step*/);
        ggml_tensor_t *obj_score = ggml_slice(ctx, outputs, 0 /*dim*/, 4, 5, 1/*step*/);
        ggml_tensor_t *cls_score = ggml_slice(ctx, outputs, 0 /*dim*/, 5, 85, 1/*step*/);

        // # tensor [grids] size: [1, 8400, 2], min: 0.0, max: 79.0, mean: 34.261906
        // # tensor [strides] size: [1, 8400, 1], min: 8.0, max: 32.0, mean: 10.666667

        x1y1 = ggml_add(ctx, x1y1, grids);
        x1y1 = ggml_mul(ctx, x1y1, strides);
        w2h2 = ggml_exp(ctx, w2h2);

        w2h2 = ggml_mul(ctx, w2h2, strides);
        w2h2 = ggml_scale(ctx, w2h2, 0.5f);

        // # Convert to boxes ...
        // x2y2 = x1y1 + w2h2/2.0
        // x1y1 = x1y1 - w2h2/2.0
        ggml_tensor_t *x2y2 = ggml_add(ctx, x1y1, w2h2);
        x1y1 = ggml_sub(ctx, x1y1, w2h2);

        // Find max class ...
        // cls_score    f32 [80, 8400, 1, 1],  (permuted) (cont) (view) (cont)
        cls_score = ggml_max(ctx, cls_score, 0 /*dim*/);
        ggml_tensor_t *cls_id = ggml_slice(ctx, cls_score, 0 /*dim*/, 1, 2, 1/*step*/);
        cls_score = ggml_slice(ctx, cls_score, 0 /*dim*/, 0, 1, 1/*step*/);
        cls_score = ggml_mul(ctx, obj_score, cls_score); // obj_score * cls_score

        outputs = ggml_cat(ctx, 4, x1y1, x2y2, cls_score, cls_id, 0/*dim*/);

        return outputs;
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, std::vector<ggml_tensor_t*> xlist) {
        // # xin is tuple: len = 3
        // #     tensor [item] size: [1, 256, 8, 8], min: 0.0, max: 0.0, mean: 0.0
        // #     tensor [item] size: [1, 512, 4, 4], min: 0.0, max: 0.0, mean: 0.0
        // #     tensor [item] size: [1, 1024, 2, 2], min: 0.0, max: 0.0, mean: 0.0

        int B, C, H, W;
        ggml_tensor_t *x, *cls_feat, *cls_output, *reg_feat, *reg_output, *obj_output, *output;
        ggml_tensor_t *outputs[3];

        for (int i = 0; i < 3; i++) {
            x = stems[i].forward(ctx, xlist[i]);
            cls_feat = cls_convs[i][0].forward(ctx, x);
            cls_feat = cls_convs[i][1].forward(ctx, cls_feat);

            cls_output = cls_preds[i].forward(ctx, cls_feat);
            cls_output = ggml_sigmoid(ctx, cls_output);

            reg_feat = reg_convs[i][0].forward(ctx, x);
            reg_feat = reg_convs[i][1].forward(ctx, reg_feat);
            reg_output = reg_preds[i].forward(ctx, reg_feat);

            obj_output = obj_preds[i].forward(ctx, reg_feat);
            // obj_output    f32 [80, 80, 1, 1], 

            obj_output = ggml_sigmoid(ctx, obj_output);

            output = ggml_cat(ctx, 3, reg_output, obj_output, cls_output, 2/*dim*/);
            // output    f32 [80, 80, 85, 1], 

            W = (int)output->ne[0];
            H = (int)output->ne[1];
            C = (int)output->ne[2];
            B = (int)output->ne[3];
            outputs[i] = ggml_reshape_3d(ctx, output, W*H, C, B);
        }

        output = ggml_cat(ctx, 3, outputs[0], outputs[1], outputs[2], 0/*dim*/);
        output = ggml_cont(ctx, ggml_permute(ctx, output, 1, 0, 2, 3)); // [8400, 85, 1, 1] --> [85, 8400, 1, 1]
        // # tensor [outputs] size: [1, 8400, 85], min: -2.040908, max: 3.459077, mean: 0.055966

        output = decode_outputs(ctx, output);

    	return output;
    }
};


struct CSPLayer {
    int in_channels;
    int out_channels;
    int nlayers = 1;
    bool shortcut=true;

    // network params
    struct BaseConv conv1;
    struct BaseConv conv2;
    struct BaseConv conv3;
    struct Bottleneck m[9];

    void create_weight_tensors(struct ggml_context* ctx) {
        int hidden_channels = out_channels/2;
        conv1.in_channels = in_channels;
        conv1.out_channels = hidden_channels;
        conv1.ksize = 1;
        conv1.stride = 1;
        conv1.create_weight_tensors(ctx);

        conv2.in_channels = in_channels;
        conv2.out_channels = hidden_channels;
        conv2.ksize = 1;
        conv2.stride = 1;
        conv2.create_weight_tensors(ctx);

        conv3.in_channels = 2*hidden_channels;
        conv3.out_channels = out_channels;
        conv3.ksize = 1;
        conv3.stride = 1;
        conv3.create_weight_tensors(ctx);

        for (int i = 0; i < nlayers; i++) {
            m[i].in_channels = hidden_channels;
            m[i].out_channels = hidden_channels;
            m[i].shortcut = shortcut;
            m[i].create_weight_tensors(ctx);
        }
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "conv1.");
        conv1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "conv2.");
        conv2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "conv3.");
        conv3.setup_weight_names(s);
        for (int i = 0; i < nlayers; i++) {
            snprintf(s, sizeof(s), "%sm.%d.", prefix, i);
            m[i].setup_weight_names(s);
        }
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        ggml_tensor_t *x1, *x2;
        x1 = conv1.forward(ctx, x);
        x2 = conv2.forward(ctx, x);
        for (int i = 0; i < nlayers; i++) {
            x1 = m[i].forward(ctx, x1);
        }
        x = ggml_concat(ctx, x1, x2, 2/*dim*/);
        x = conv3.forward(ctx, x);

    	return x;
    }
};


struct Focus {
    int in_channels = 3;
    int out_channels;

    // network params
    struct BaseConv conv;

    void create_weight_tensors(struct ggml_context* ctx) {
        conv.in_channels = 4 * in_channels;
        conv.out_channels = out_channels;
        conv.ksize = 3;
        conv.stride = 1;
        conv.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "conv.");
        conv.setup_weight_names(s);
    }

    ggml_tensor_t* forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // # tensor [x1] size: [1, 3, 640, 640], min: 0.0, max: 255.0, mean: 128.074554
        ggml_tensor_t *patch_top, *patch_top_left, *patch_top_right;
        ggml_tensor_t *patch_bot, *patch_bot_left, *patch_bot_right;

        patch_top = ggml_slice(ctx, x, 1 /*dim on H*/, 0, 640, 2/*step*/);
        patch_bot = ggml_slice(ctx, x, 1 /*dim on H*/, 1, 640, 2/*step*/);

        patch_top_left = ggml_slice(ctx, patch_top, 0 /*dim on W*/, 0, 640, 2 /*step*/);
        patch_top_right = ggml_slice(ctx, patch_top, 0 /*dim on W*/, 1, 640, 2 /*step*/);
        patch_bot_left = ggml_slice(ctx, patch_bot, 0 /*dim on W*/, 0, 640, 2 /*step*/);
        patch_bot_right = ggml_slice(ctx, patch_bot, 0 /*dim on W*/, 1, 640, 2 /*step*/);

        x = ggml_cat(ctx, 4, patch_top_left, patch_bot_left, patch_top_right, patch_bot_right, 2 /*dim*/);
        // # tensor [x2] size: [1, 12, 320, 320], min: 0.0, max: 255.0, mean: 128.074554
        return conv.forward(ctx, x);
    }
};

struct CSPDarknet {
    const int base_channels = 64;
    const int base_depth = 3;

    // network params
    struct Focus stem;
    struct BaseConv dark2_0;
    struct CSPLayer dark2_1;
    struct BaseConv dark3_0;
    struct CSPLayer dark3_1;
    struct BaseConv dark4_0;
    struct CSPLayer dark4_1;
    struct BaseConv dark5_0;
    struct SPPBottleneck dark5_1;
    struct CSPLayer dark5_2;

    void create_weight_tensors(struct ggml_context* ctx) {
        // # stem
        // self.stem = Focus(3, base_channels)
        stem.in_channels = 3;
        stem.out_channels = base_channels;
        stem.create_weight_tensors(ctx);

        // # dark2
        // self.dark2 = nn.Sequential(
        //     BaseConv(base_channels, base_channels * 2, 3, 2),
        //     CSPLayer(base_channels * 2, base_channels * 2, n=base_depth, shortcut=True)
        // )
        dark2_0.in_channels = base_channels;
        dark2_0.out_channels = base_channels * 2;
        dark2_0.ksize = 3;
        dark2_0.stride = 2;
        dark2_0.create_weight_tensors(ctx);

        dark2_1.in_channels = base_channels * 2;
        dark2_1.out_channels = base_channels * 2;
        dark2_1.nlayers = base_depth;
        dark2_1.shortcut = true;
        dark2_1.create_weight_tensors(ctx);

        // # dark3
        // self.dark3 = nn.Sequential(
        //     BaseConv(base_channels * 2, base_channels * 4, 3, 2),
        //     CSPLayer(base_channels * 4, base_channels * 4, n=base_depth * 3, shortcut=True)
        // )
        dark3_0.in_channels = base_channels * 2;
        dark3_0.out_channels = base_channels * 4;
        dark3_0.ksize = 3;
        dark3_0.stride = 2;
        dark3_0.create_weight_tensors(ctx);

        dark3_1.in_channels = base_channels * 4;
        dark3_1.out_channels = base_channels * 4;
        dark3_1.nlayers = base_depth * 3;
        dark3_1.shortcut = true;
        dark3_1.create_weight_tensors(ctx);

        // # dark4
        // self.dark4 = nn.Sequential(
        //     BaseConv(base_channels * 4, base_channels * 8, 3, 2),
        //     CSPLayer(base_channels * 8, base_channels * 8, n=base_depth * 3, shortcut=True)
        // )
        dark4_0.in_channels = base_channels * 4;
        dark4_0.out_channels = base_channels * 8;
        dark4_0.ksize = 3;
        dark4_0.stride = 2;
        dark4_0.create_weight_tensors(ctx);

        dark4_1.in_channels = base_channels * 8;
        dark4_1.out_channels = base_channels * 8;
        dark4_1.nlayers = base_depth * 3;
        dark4_1.shortcut = true;
        dark4_1.create_weight_tensors(ctx);

        // # dark5
        // self.dark5 = nn.Sequential(
        //     BaseConv(base_channels * 8, base_channels * 16, 3, 2),
        //     SPPBottleneck(base_channels * 16, base_channels * 16),
        //     CSPLayer(base_channels * 16, base_channels * 16, n=base_depth, shortcut=False),
        // )
        dark5_0.in_channels = base_channels * 8;
        dark5_0.out_channels = base_channels * 16;
        dark5_0.ksize = 3;
        dark5_0.stride = 2;
        dark5_0.create_weight_tensors(ctx);

        dark5_1.in_channels = base_channels * 16;
        dark5_1.out_channels = base_channels * 16;
        dark5_1.create_weight_tensors(ctx);

        dark5_2.in_channels = base_channels * 16;
        dark5_2.out_channels = base_channels * 16;
        dark5_2.nlayers = base_depth;
        dark5_2.shortcut = false;
        dark5_2.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];

        snprintf(s, sizeof(s), "%s%s", prefix, "stem.");
        stem.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "dark2.0.");
        dark2_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "dark2.1.");
        dark2_1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "dark3.0.");
        dark3_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "dark3.1.");
        dark3_1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "dark4.0.");
        dark4_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "dark4.1.");
        dark4_1.setup_weight_names(s);

        snprintf(s, sizeof(s), "%s%s", prefix, "dark5.0.");
        dark5_0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "dark5.1.");
        dark5_1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "dark5.2.");
        dark5_2.setup_weight_names(s);
    }

    std::vector<ggml_tensor_t *> forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        // # outputs = {}
        // x = self.stem(x)
        // # outputs["stem"] = x
        // # d1 = x
        // x = self.dark2(x)
        // # outputs["dark2"] = x
        // # d2 = x
        // x = self.dark3(x)
        // d3 = x
        // # outputs["dark3"] = x
        // x = self.dark4(x)
        // # outputs["dark4"] = x
        // d4 = x
        // x = self.dark5(x)
        // # outputs["dark5"] = x
        // d5 = x
        // # return {k: v for k, v in outputs.items() if k in self.out_features}
        // return d3, d4, d5
        std::vector<ggml_tensor_t *> xlist;

        x = stem.forward(ctx, x);
        x = dark2_0.forward(ctx, x);
        x = dark2_1.forward(ctx, x);

        x = dark3_0.forward(ctx, x);
        x = dark3_1.forward(ctx, x);
        xlist.push_back(x); // d3

        x = dark4_0.forward(ctx, x);
        x = dark4_1.forward(ctx, x);
        xlist.push_back(x); // d4

        x = dark5_0.forward(ctx, x);
        x = dark5_1.forward(ctx, x);
        x = dark5_2.forward(ctx, x);
        xlist.push_back(x); // d5

    	return xlist;
    }
};

struct YOLOPAFPN {
    const int in_channels[3] = { 256, 512, 1024 };

    // network params
    struct CSPDarknet backbone;
    struct BaseConv lateral_conv0;
    struct CSPLayer C3_p4;
    struct BaseConv reduce_conv1;
    struct CSPLayer C3_p3;
    struct BaseConv bu_conv2;
    struct CSPLayer C3_n3;
    struct BaseConv bu_conv1;
    struct CSPLayer C3_n4;

    void create_weight_tensors(struct ggml_context* ctx) {
        // self.backbone = CSPDarknet()
        // backbone.base_channels = 64;
        // backbone.base_depth = 3;
        backbone.create_weight_tensors(ctx);

        // self.lateral_conv0 = BaseConv(in_channels[2], in_channels[1], 1, 1)
        lateral_conv0.in_channels = in_channels[2];
        lateral_conv0.out_channels = in_channels[1];
        lateral_conv0.ksize = 1;
        lateral_conv0.stride = 1;
        lateral_conv0.create_weight_tensors(ctx);

        // self.C3_p4 = CSPLayer(2 * in_channels[1], in_channels[1], 3, shortcut=False)  # cat
        C3_p4.in_channels = 2 * in_channels[1];
        C3_p4.out_channels = in_channels[1];
        C3_p4.nlayers = 3;
        C3_p4.shortcut = false;
        C3_p4.create_weight_tensors(ctx);

        // self.reduce_conv1 = BaseConv(in_channels[1], in_channels[0], 1, 1)
        reduce_conv1.in_channels = in_channels[1];
        reduce_conv1.out_channels = in_channels[0];
        reduce_conv1.ksize = 1;
        reduce_conv1.stride = 1;
        reduce_conv1.create_weight_tensors(ctx);

        // self.C3_p3 = CSPLayer(2 * in_channels[0], in_channels[0], 3, shortcut=False)
        C3_p3.in_channels = 2 * in_channels[0];
        C3_p3.out_channels = in_channels[0];
        C3_p3.nlayers = 3;
        C3_p3.shortcut = false;
        C3_p3.create_weight_tensors(ctx);

        // # bottom-up conv
        // self.bu_conv2 = BaseConv(in_channels[0], in_channels[0], 3, 2)
        // self.C3_n3 = CSPLayer(2 * in_channels[0], in_channels[1], 3, shortcut=False)
        bu_conv2.in_channels = in_channels[0];
        bu_conv2.out_channels = in_channels[0];
        bu_conv2.ksize = 3;
        bu_conv2.stride = 2;
        bu_conv2.create_weight_tensors(ctx);

        C3_n3.in_channels = 2 * in_channels[0];
        C3_n3.out_channels = in_channels[1];
        C3_n3.nlayers = 3;
        C3_n3.shortcut = false;
        C3_n3.create_weight_tensors(ctx);

        // # bottom-up conv
        // self.bu_conv1 = BaseConv(in_channels[1], in_channels[1], 3, 2)
        // self.C3_n4 = CSPLayer(2 * in_channels[1], in_channels[2], 3, shortcut=False)
        bu_conv1.in_channels = in_channels[1];
        bu_conv1.out_channels = in_channels[1];
        bu_conv1.ksize = 3;
        bu_conv1.stride = 2;
        bu_conv1.create_weight_tensors(ctx);

        C3_n4.in_channels = 2*in_channels[1];
        C3_n4.out_channels = in_channels[2];
        C3_n4.nlayers = 3;
        C3_n4.shortcut = false;
        C3_n4.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "backbone.");
        backbone.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "lateral_conv0.");
        lateral_conv0.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "C3_p4.");
        C3_p4.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "reduce_conv1.");
        reduce_conv1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "C3_p3.");
        C3_p3.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "bu_conv2.");
        bu_conv2.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "C3_n3.");
        C3_n3.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "bu_conv1.");
        bu_conv1.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "C3_n4.");
        C3_n4.setup_weight_names(s);
    }

    std::vector<ggml_tensor_t *> forward(struct ggml_context* ctx, ggml_tensor_t* x) {
        int W, H;
        std::vector<ggml_tensor_t *>xlist;
        std::vector<ggml_tensor_t *>ylist;
        ggml_tensor_t *dark3, *dark4, *dark5;

        xlist = backbone.forward(ctx, x);
        dark3 = xlist[0]; dark4 = xlist[1]; dark5 = xlist[2];

        ggml_tensor_t *dark5_fpn_out = lateral_conv0.forward(ctx, dark5);
        W = dark5_fpn_out->ne[0];
        H = dark5_fpn_out->ne[1];
        ggml_tensor_t *f_out0 = ggml_interpolate(ctx, dark5_fpn_out, 0 /*W*/, 2*W);
        f_out0 = ggml_interpolate(ctx, f_out0, 1 /*H*/, 2*H);
        f_out0 = ggml_concat(ctx, f_out0, dark4, 2/*dim*/);
        f_out0 = C3_p4.forward(ctx, f_out0);

        ggml_tensor_t *fpn_out1 = reduce_conv1.forward(ctx, f_out0);
        W = fpn_out1->ne[0];
        H = fpn_out1->ne[1];
        ggml_tensor_t *f_out1 = ggml_interpolate(ctx, fpn_out1, 0 /*W*/, 2*W);
        f_out1 = ggml_interpolate(ctx, f_out1, 1 /*H*/, 2*H);
        f_out1 = ggml_concat(ctx, f_out1, dark3, 2 /*dim*/);
        ggml_tensor_t *pan_out2 = C3_p3.forward(ctx, f_out1);

        ggml_tensor_t *p_out1 = bu_conv2.forward(ctx, pan_out2);
        p_out1 = ggml_concat(ctx, p_out1, fpn_out1, 2 /*dim*/);
        ggml_tensor_t *pan_out1 = C3_n3.forward(ctx, p_out1);

        ggml_tensor_t *p_out0 = bu_conv1.forward(ctx, pan_out1);
        p_out0 = ggml_concat(ctx, p_out0, dark5_fpn_out, 2/*dim*/);
        ggml_tensor_t *pan_out0 = C3_n4.forward(ctx, p_out0);

        ylist.push_back(pan_out2);
        ylist.push_back(pan_out1);
        ylist.push_back(pan_out0);

    	return ylist;
    }
};

struct YOLOX : GGMLNetwork {
    int MAX_H = 640;
    int MAX_W = 640;
    int MAX_TIMES = 1;

    // network params
    struct YOLOPAFPN backbone;
    struct YOLOXHead head;

    size_t get_graph_size()
    {
        return GGML_DEFAULT_GRAPH_SIZE * 2; // 2048
    }

    void create_weight_tensors(struct ggml_context* ctx) {
        backbone.create_weight_tensors(ctx);

        head.num_classes = 80;
        head.create_weight_tensors(ctx);
    }

    void setup_weight_names(const char *prefix) {
        char s[GGML_MAX_NAME];
        snprintf(s, sizeof(s), "%s%s", prefix, "backbone.");
        backbone.setup_weight_names(s);
        snprintf(s, sizeof(s), "%s%s", prefix, "head.");
        head.setup_weight_names(s);
    }


    ggml_tensor_t* forward(struct ggml_context* ctx, int argc, ggml_tensor_t* argv[]) {
        GGML_ASSERT(argc == 1);
        ggml_tensor_t *x = argv[0];

        int W = (int)x->ne[0];
        int H = (int)x->ne[1];
        x = ggml_interpolate(ctx, x, 0 /*on W*/, MAX_W);
        x = ggml_interpolate(ctx, x, 1 /*on H*/, MAX_H);
        x = ggml_scale(ctx, x, 255.0f);

        CheckPoint();

        std::vector<ggml_tensor_t *>fpn_outs = backbone.forward(ctx, x);
        CheckPoint();

        ggml_tensor_t *detect_result = head.forward(ctx, fpn_outs);
        // # tensor [detect_result] size: [1, 8400, 6], min: -51.966671, max: 866.434814, mean: 194.278854
        // # (x1, y1, x2, y2, obj_score * class_score, class_id)

        float sh = (float)H/(float)MAX_H;
        float sw = (float)W/(float)MAX_W;

        ggml_tensor_t *x1 = ggml_slice(ctx, detect_result, 0 /*dim*/, 0, 1, 1/*step*/);
        ggml_tensor_t *y1 = ggml_slice(ctx, detect_result, 0 /*dim*/, 1, 2, 1/*step*/);
        ggml_tensor_t *x2 = ggml_slice(ctx, detect_result, 0 /*dim*/, 2, 3, 1/*step*/);
        ggml_tensor_t *y2 = ggml_slice(ctx, detect_result, 0 /*dim*/, 3, 4, 1/*step*/);
        ggml_tensor_t *os = ggml_slice(ctx, detect_result, 0 /*dim*/, 4, 6, 1/*step*/); // others

        x1 = ggml_scale(ctx, x1, sw);
        y1 = ggml_scale(ctx, y1, sh);
        x2 = ggml_scale(ctx, x2, sw);
        y2 = ggml_scale(ctx, y2, sh);

        x = ggml_cat(ctx, 5, x1, y1, x2, y2, os, 0/*dim*/);

        return x;
    }
};


struct YOLOXNetwork {
    YOLOX net;
    GGMLModel model;

    int init(int device)
    {
        // -----------------------------------------------------------------------------------------
        net.set_device(device);
        net.start_engine();
        // net.dump();

        check_point(model.preload("models/image_detect_f16.gguf") == RET_OK);
        load();

        return RET_OK;
    }

    int load()
    {
        return net.load_weight(&model, "");
    }

    TENSOR* forward(TENSOR* image_tensor)
    {
        TENSOR* argv[1];
        argv[0] = image_tensor;

        TENSOR* detect_result = net.engine_forward(ARRAY_SIZE(argv), argv);

        return detect_result;
    }


    void exit()
    {
        model.clear();

        net.stop_engine();
    }
};

#endif // __YOLOX__H__
