extern "C"{
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/ringbuf.h"

#include "driver/adc.h"
#include "esp_timer.h"
#include "esp_err.h"
#include "driver/usb_serial_jtag.h"
#include "esp_vfs_usb_serial_jtag.h"
#include "esp_vfs_dev.h"
}
#include "../sensors/sensors.hpp"
#include "esp_log.h"

// ==== ADC & GPIO mapping ====
#define ECG1_CHANNEL   ADC1_CHANNEL_1   // GPIO2
#define ECG2_CHANNEL   ADC1_CHANNEL_3   // GPIO4
#define EMG1_CHANNEL   ADC1_CHANNEL_0   // GPIO1
#define EMG2_CHANNEL   ADC1_CHANNEL_2   // GPIO3

// ==== Sampling rates ====
#define EXG_FS         2000
#define IMU_FS         1000
#define PPG_FS         200

static const int64_t Ts_EXG_us = 1000000LL / EXG_FS;  // 500us

// ==== Global latest samples ====
uint16_t g_latest_ecg1 = 0;
uint16_t g_latest_ecg2 = 0;
uint16_t g_latest_emg1 = 0;
uint16_t g_latest_emg2 = 0;

// ==== Frame definitions ====

typedef struct __attribute__((packed)) {
    uint8_t  header1;   // 0xAA
    uint8_t  header2;   // 0x55
    uint32_t seq;
    uint16_t ecg1;
    uint16_t ecg2;
    uint16_t emg1;
    uint16_t emg2;
} ExgFrame;

typedef struct __attribute__((packed)) {
    uint8_t  header1;   // 0xAA
    uint8_t  header2;   // 0x61
    uint32_t seq;
    int16_t  ax, ay, az;
    int16_t  gx, gy, gz;
} ImuFrame;

typedef struct __attribute__((packed)) {
    uint8_t  header1;   // 0xAA
    uint8_t  header2;   // 0x62
    uint32_t seq;
    uint16_t hr;
    uint16_t spo2;
    uint8_t  ppg_conf;
    uint8_t  ppg_status;
    uint32_t ppg_ir;
    uint32_t ppg_red;
} PpgFrame;

_Static_assert(sizeof(ExgFrame) == 14, "ExgFrame must be 14 bytes");
_Static_assert(sizeof(ImuFrame) == 18, "ImuFrame must be 18 bytes");
_Static_assert(sizeof(PpgFrame) == 20, "PpgFrame must be 20 bytes");

// ==== Ring Buffer for TX ====
#define TX_RINGBUF_SIZE  (128 * 1024)
static RingbufHandle_t g_tx_ringbuf = NULL;

// ==== ADC init ====
static void adc_init_exg(void)
{
    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(ECG1_CHANNEL, ADC_ATTEN_DB_12);
    adc1_config_channel_atten(ECG2_CHANNEL, ADC_ATTEN_DB_12);
    adc1_config_channel_atten(EMG1_CHANNEL, ADC_ATTEN_DB_12);
    adc1_config_channel_atten(EMG2_CHANNEL, ADC_ATTEN_DB_12);
}

static inline uint16_t adc_read_raw(adc1_channel_t ch)
{
    int v = adc1_get_raw(ch);
    if (v < 0)    v = 0;
    if (v > 4095) v = 4095;
    return (uint16_t)v;
}

// ==== USB console init ====
static void init_usb_binary_console(void)
{
    usb_serial_jtag_driver_config_t cfg = {
        .tx_buffer_size = 16384, 
        .rx_buffer_size = 1024,
    };
    ESP_ERROR_CHECK(usb_serial_jtag_driver_install(&cfg));

    esp_vfs_dev_usb_serial_jtag_set_tx_line_endings(ESP_LINE_ENDINGS_LF);
    esp_vfs_dev_usb_serial_jtag_set_rx_line_endings(ESP_LINE_ENDINGS_LF);
    esp_vfs_usb_serial_jtag_use_driver();
}

// ==== Send frame to ring buffer (ISR safe) ====
static inline void IRAM_ATTR ringbuf_send(const void* data, size_t len)
{
    xRingbufferSendFromISR(g_tx_ringbuf, data, len, NULL);
}

// ==== 2kHz Timer Callback ====
static void IRAM_ATTR exg_timer_cb(void *arg)
{
    static uint32_t seq = 0;
    static uint32_t imu_div = 0;
    static uint32_t ppg_div = 0;

    uint32_t cur_seq = seq++;

    // ---- ExG Frame @ 2kHz ----
    ExgFrame f = {
        .header1 = 0xAA,
        .header2 = 0x55,
        .seq     = cur_seq,
        .ecg1    = g_latest_ecg1,
        .ecg2    = g_latest_ecg2,
        .emg1    = g_latest_emg1,
        .emg2    = g_latest_emg2,
    };
    ringbuf_send(&f, sizeof(ExgFrame));

    // ---- IMU Frame @ 1kHz ----
    imu_div++;
    if (imu_div >= 2) {
        imu_div = 0;
        ImuFrame im = {
            .header1 = 0xAA,
            .header2 = 0x61,
            .seq     = cur_seq,
            .ax = g_latest_imu.ax,
            .ay = g_latest_imu.ay,
            .az = g_latest_imu.az,
            .gx = g_latest_imu.gx,
            .gy = g_latest_imu.gy,
            .gz = g_latest_imu.gz,
        };
        ringbuf_send(&im, sizeof(ImuFrame));
    }

    // ---- PPG Frame @ 200Hz ----
// ---- PPG Frame @ 200Hz ----
ppg_div++;
if (ppg_div >= 10) {
    ppg_div = 0;
    PpgFrame pf = {
        .header1    = 0xAA,
        .header2    = 0x62,
        .seq        = cur_seq,
        .hr         = g_latest_ppg.heart_rate,
        .spo2       = g_latest_ppg.oxygen,
        .ppg_conf   = g_latest_ppg.confidence,
        .ppg_status = g_latest_ppg.status,
        .ppg_ir     = g_latest_ppg.ir_led,
        .ppg_red    = g_latest_ppg.red_led,
    };
    ringbuf_send(&pf, sizeof(PpgFrame));
}
}

// ==== TX Task: read from ring buffer and output ====
static void tx_task(void *arg)
{
    uint8_t batch_buf[2048];  // batch buffer
    size_t batch_len = 0;

    while (1) {
        size_t item_size;
        void* item = xRingbufferReceive(g_tx_ringbuf, &item_size, pdMS_TO_TICKS(10));
        
        if (item != NULL) {
            // add data to batch buffer
            if (batch_len + item_size <= sizeof(batch_buf)) {
                memcpy(batch_buf + batch_len, item, item_size);
                batch_len += item_size;
            }
            vRingbufferReturnItem(g_tx_ringbuf, item);

            // send if batch buffer is full
            if (batch_len >= 256) {
                usb_serial_jtag_write_bytes(batch_buf, batch_len, pdMS_TO_TICKS(10));
                batch_len = 0;
            }
        } else {
            // timeout, flush remaining data
            if (batch_len > 0) {
                usb_serial_jtag_write_bytes(batch_buf, batch_len, pdMS_TO_TICKS(10));
                batch_len = 0;
            }
        }
    }
}

// ==== ADC Sampling Task @ 2kHz ====
static void exg_sample_task(void *arg)
{
    const TickType_t period = pdMS_TO_TICKS(1);
    TickType_t last_wake = xTaskGetTickCount();

    while (1) {
        for (int i = 0; i < 2; i++) {
            g_latest_emg1 = adc_read_raw(EMG1_CHANNEL);
            g_latest_emg2 = adc_read_raw(EMG2_CHANNEL);
            g_latest_ecg1 = adc_read_raw(ECG1_CHANNEL);
            g_latest_ecg2 = adc_read_raw(ECG2_CHANNEL);

            if (i == 0) {
                esp_rom_delay_us(500);
            }
        }
        vTaskDelayUntil(&last_wake, period);
    }
}

// ==== Main ====
extern "C" void app_main(void)
{
    esp_log_level_set("*", ESP_LOG_NONE);

    // Create Ring Buffer
    g_tx_ringbuf = xRingbufferCreate(TX_RINGBUF_SIZE, RINGBUF_TYPE_BYTEBUF);
    assert(g_tx_ringbuf != NULL);

    adc_init_exg();
    init_usb_binary_console();

    ESP_ERROR_CHECK(sensors::init());

    // TX task @ high priority
    xTaskCreatePinnedToCore(tx_task, "tx_task", 4096, NULL, 7, NULL, 1);

    xTaskCreatePinnedToCore(exg_sample_task, "exg_sample", 4096, NULL, 6, NULL, 1);

    esp_timer_handle_t exg_timer;
    const esp_timer_create_args_t exg_args = {
        .callback = &exg_timer_cb,
        .arg      = NULL,
        .name     = "exg_timer",
    };
    ESP_ERROR_CHECK(esp_timer_create(&exg_args, &exg_timer));
    ESP_ERROR_CHECK(esp_timer_start_periodic(exg_timer, Ts_EXG_us));
}