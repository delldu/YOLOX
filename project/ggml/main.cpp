/************************************************************************************
***
*** Copyright 2024 Dell Du(18588220928@163.com), All Rights Reserved.
***
*** File Author: Dell, Tue 02 Apr 2024 03:49:53 PM CST
***
************************************************************************************/

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <ggml_engine.h>
#include <nimage/tensor.h>

#include "yolox.h"

#define DEFAULT_DEVICE 1
#define DEFAULT_OUTPUT "output"

int image_detect_predict(YOLOXNetwork* detect_net, char* image_file, char* output_dir);

static void image_detect_help(char* cmd)
{
    printf("Usage: %s [option] image_files\n", cmd);
    printf("    -h, --help                   Display this help, version %s.\n", ENGINE_VERSION);
    printf("    -d, --device <no>            Set device (0 -- cpu, 1 -- cuda0, 2 -- cuda1, ..., default: %d)\n",
        DEFAULT_DEVICE);
    printf("    -o, --output                 output dir, default: %s.\n", DEFAULT_OUTPUT);

    exit(1);
}

int main(int argc, char** argv)
{
    int optc;
    int option_index = 0;
    int device_no = DEFAULT_DEVICE;
    char* output_dir = (char*)DEFAULT_OUTPUT;

    struct option long_opts[] = {
        { "help", 0, 0, 'h' },
        { "device", 1, 0, 'd' },
        { "output", 1, 0, 'o' }, { 0, 0, 0, 0 }
    };

    if (argc <= 1)
        image_detect_help(argv[0]);

    while ((optc = getopt_long(argc, argv, "h d: o:", long_opts, &option_index)) != EOF) {
        switch (optc) {
        case 'd':
            device_no = atoi(optarg);
            break;
        case 'o':
            output_dir = optarg;
            break;
        case 'h': // help
        default:
            image_detect_help(argv[0]);
            break;
        }
    }

    // client
    if (optind == argc) // no input image, nothing to do ...
        return 0;

    YOLOXNetwork detect_net;

    // int network
    {
        detect_net.init(device_no);
    }

    // if (optind < argc - 1) {
    //     printf("image_files are required.\n");
    //     image_detect_help(argv[0]);
    // }

    for (int i = optind; i < argc; i++) {
        image_detect_predict(&detect_net, argv[i], output_dir);
    }

    // free network ...
    {
        detect_net.exit();
    }

    return 0;
}