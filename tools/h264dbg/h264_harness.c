// H.264 debug harness — decodes a raw Annex-B .h264 through the locally
// built (and instrumented) FFmpeg tree, demuxed packet by packet.
// Build/run recipes in tools/h264dbg/README.md.
#include <stdio.h>
#include <stdlib.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>

int main(int argc, char **argv) {
    AVFormatContext *ic = NULL;
    if (avformat_open_input(&ic, argv[1], NULL, NULL) < 0) {
        fprintf(stderr, "open fail\n");
        return 1;
    }
    const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    AVCodecContext *ctx = avcodec_alloc_context3(codec);
    ctx->thread_count = 1;
    avcodec_parameters_to_context(ctx, ic->streams[0]->codecpar);
    if (avcodec_open2(ctx, codec, NULL) < 0) {
        fprintf(stderr, "codec open fail\n");
        return 1;
    }
    AVPacket *pkt = av_packet_alloc();
    AVFrame *frame = av_frame_alloc();
    int frames = 0;
    while (av_read_frame(ic, pkt) >= 0) {
        avcodec_send_packet(ctx, pkt);
        av_packet_unref(pkt);
        while (avcodec_receive_frame(ctx, frame) >= 0) {
            frames++;
            av_frame_unref(frame);
        }
    }
    avcodec_send_packet(ctx, NULL);
    while (avcodec_receive_frame(ctx, frame) >= 0) {
        frames++;
        av_frame_unref(frame);
    }
    fprintf(stderr, "== frames=%d\n", frames);
    return 0;
}
