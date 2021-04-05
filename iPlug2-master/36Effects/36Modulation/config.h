#define PLUG_NAME "36Modulation"
#define PLUG_MFR "36Effect"
#define PLUG_VERSION_HEX 0x00010000
#define PLUG_VERSION_STR "2.0.0"
#define PLUG_UNIQUE_ID '36Mo'
#define PLUG_MFR_ID '36Ef'
#define PLUG_URL_STR "https://iplug2.github.io"
#define PLUG_EMAIL_STR "spam@me.com"
#define PLUG_COPYRIGHT_STR "Copyright 2019 36 Inc"
#define PLUG_CLASS_NAME IPlugSideChain

#define BUNDLE_NAME "36Effect"
#define BUNDLE_MFR "36"
#define BUNDLE_DOMAIN "com"

#define SHARED_RESOURCES_SUBPATH "36Effect"

#define PLUG_CHANNEL_IO "\
1-1 \
1.1-1 \
1.2-1 \
1.2-2 \
2.1-1 \
2.1-2 \
2-2 \
2.2-2"

#define PLUG_LATENCY 0
#define PLUG_TYPE 0
#define PLUG_DOES_MIDI_IN 0
#define PLUG_DOES_MIDI_OUT 0
#define PLUG_DOES_MPE 0
#define PLUG_DOES_STATE_CHUNKS 0
#define PLUG_HAS_UI 1
#define PLUG_WIDTH 400
#define PLUG_HEIGHT 100
#define PLUG_FPS 60
#define PLUG_SHARED_RESOURCES 0

#define AUV2_ENTRY IPlugSideChain_Entry
#define AUV2_ENTRY_STR "IPlugSideChain_Entry"
#define AUV2_FACTORY IPlugSideChain_Factory
#define AUV2_VIEW_CLASS IPlugSideChain_View
#define AUV2_VIEW_CLASS_STR "IPlugSideChain_View"

#define AAX_TYPE_IDS 'ISC1', 'ISC2'
#define AAX_TYPE_IDS_AUDIOSUITE 'SCA1', 'SCA2'
#define AAX_PLUG_MFR_STR "Acme"
#define AAX_PLUG_NAME_STR "IPlugSideChain\nIPSC"
#define AAX_PLUG_CATEGORY_STR "Effect"
#define AAX_DOES_AUDIOSUITE 1

#define VST3_SUBCATEGORY "Fx"

#define APP_NUM_CHANNELS 2
#define APP_N_VECTOR_WAIT 0
#define APP_MULT 1
#define APP_COPY_AUV3 0
#define PLUG_HOST_RESIZE 0
#define APP_SIGNAL_VECTOR_SIZE 64

#define ROBOTO_FN "Roboto-Regular.ttf"