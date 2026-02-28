#include "sensors.hpp"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/i2c.h"
#include "esp_log.h"

static const char* TAG = "SENSORS";
using namespace maxim;

ImuRawData   g_latest_imu;
BioData      g_latest_ppg;

// ================= I2C configuration =================
#define I2C_Master_NUM   I2C_NUM_0
#define I2C_EXG_SDA_GPIO 11
#define I2C_EXG_SCL_GPIO 12
#define I2C_EXG_FREQ_HZ  400000

#define I2C_MASTER_TX_BUF_DISABLE   0
#define I2C_MASTER_RX_BUF_DISABLE   0

// ================= Devices I2C Address =================
#define MPU6050_ADDR   0x68

// ================== MPU6050 registers ==================
#define MPU6050_REG_PWR_MGMT_1    0x6B
#define MPU6050_REG_WHO_AM_I      0x75
#define MPU6050_REG_ACCEL_XOUT_H  0x3B
#define MPU6050_REG_SMPLRT_DIV    0x19
#define MPU6050_REG_CONFIG        0x1A
#define MPU6050_REG_GYRO_CONFIG   0x1B
#define MPU6050_REG_ACCEL_CONFIG  0x1C

// ================== PPG / MAX32664 ==================
static Max32664Hub g_bioHub(I2C_MAX32664_RESET_PIN, I2C_MAX32664_MFIO_PIN);

// ============ I2C helper ===========

static esp_err_t i2c_write_reg(uint8_t dev_addr, uint8_t reg_addr,
                               const uint8_t* data, size_t len)
{
    uint8_t buf[1 + len];
    buf[0] = reg_addr;
    if (len > 0 && data != nullptr) {
        for (size_t i = 0; i < len; ++i) {
            buf[1 + i] = data[i];
        }
    }
    return i2c_master_write_to_device(I2C_Master_NUM, dev_addr,
                                      buf, 1 + len,
                                      pdMS_TO_TICKS(10));
}

static esp_err_t i2c_read_reg(uint8_t dev_addr, uint8_t reg_addr,
                              uint8_t* data, size_t len)
{
    return i2c_master_write_read_device(I2C_Master_NUM, dev_addr,
                                        &reg_addr, 1,
                                        data, len,
                                        pdMS_TO_TICKS(10));
}

// ================= MPU6050 internal =================

static esp_err_t mpu6050_init_internal(void)
{
    esp_err_t err;
    uint8_t whoami = 0;

    err = i2c_read_reg(MPU6050_ADDR, MPU6050_REG_WHO_AM_I, &whoami, 1);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "MPU6050 WHO_AM_I read failed: %s", esp_err_to_name(err));
        return err;
    }
    ESP_LOGI(TAG, "MPU6050 WHO_AM_I = 0x%02X", whoami);

    uint8_t val;

    val = 0x01;
    err = i2c_write_reg(MPU6050_ADDR, MPU6050_REG_PWR_MGMT_1, &val, 1);
    if (err != ESP_OK) return err;

    val = 0x01;
    err = i2c_write_reg(MPU6050_ADDR, MPU6050_REG_CONFIG, &val, 1);
    if (err != ESP_OK) return err;

    val = 0x10;
    err = i2c_write_reg(MPU6050_ADDR, MPU6050_REG_GYRO_CONFIG, &val, 1);
    if (err != ESP_OK) return err;

    val = 0x08;
    err = i2c_write_reg(MPU6050_ADDR, MPU6050_REG_ACCEL_CONFIG, &val, 1);
    if (err != ESP_OK) return err;

    val = 0;
    err = i2c_write_reg(MPU6050_ADDR, MPU6050_REG_SMPLRT_DIV, &val, 1);
    if (err != ESP_OK) return err;

    ESP_LOGI(TAG, "MPU6050 init done @ 1kHz");
    return ESP_OK;
}

static esp_err_t mpu6050_read_internal(ImuRawData* out)
{
    uint8_t buf[14];
    esp_err_t err = i2c_read_reg(MPU6050_ADDR,
                                 MPU6050_REG_ACCEL_XOUT_H,
                                 buf, sizeof(buf));
    if (err != ESP_OK) return err;

    out->ax       = (int16_t)((buf[0] << 8) | buf[1]);
    out->ay       = (int16_t)((buf[2] << 8) | buf[3]);
    out->az       = (int16_t)((buf[4] << 8) | buf[5]);
    out->temp_raw = (int16_t)((buf[6] << 8) | buf[7]);
    out->gx       = (int16_t)((buf[8] << 8) | buf[9]);
    out->gy       = (int16_t)((buf[10] << 8) | buf[11]);
    out->gz       = (int16_t)((buf[12] << 8) | buf[13]);

    return ESP_OK;
}

// ================= PPG / MAX32664 internal =================

static esp_err_t ppg_init_internal(void)
{
    esp_log_level_set(TAG, ESP_LOG_INFO);
    esp_log_level_set("max32664", ESP_LOG_INFO);

    ESP_LOGI(TAG, "PPG: i2c_bus_init");
    esp_err_t res = g_bioHub.i2c_bus_init(I2C_MASTER_SDA, I2C_MASTER_SCL);
    if (res != ESP_OK) {
        ESP_LOGE(TAG, "BIO_HUB i2c_bus_init failed: %s", esp_err_to_name(res));
        return res;
    }

    ESP_LOGI(TAG, "PPG: begin()");
    uint8_t error = g_bioHub.begin();
    if (error != 0) {
        ESP_LOGE(TAG, "BIO_HUB.begin() failed, error=%u", error);
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "PPG sensor started");

    error = g_bioHub.config_sensor_bpm(MODE_ONE);
    if (error != 0) {
        ESP_LOGE(TAG, "config_sensor() failed, error=%u", error);
        return ESP_FAIL;
    }

    error = g_bioHub.set_sample_rate(200);
    if (error != 0) {
        ESP_LOGE(TAG, "set_sample_rate failed");
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "PPG sample rate = %u Hz", g_bioHub.read_sample_rate());

    error = g_bioHub.set_pulse_width(411);
    if (error != 0) {
        ESP_LOGE(TAG, "set_pulse_width failed");
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "PPG pulse width = %u us", g_bioHub.read_pulse_width());

    ESP_LOGI(TAG, "PPG configured");
    vTaskDelay(pdMS_TO_TICKS(1000));

    return ESP_OK;
}

// ================= IMU 1kHz Task =================

static void imu_sample_task(void *arg)
{
    const TickType_t period = pdMS_TO_TICKS(1);
    TickType_t last_wake = xTaskGetTickCount();
    ImuRawData imu_tmp{};

    while (1) {
        if (mpu6050_read_internal(&imu_tmp) == ESP_OK) {
            g_latest_imu = imu_tmp;
        }
        vTaskDelayUntil(&last_wake, period);
    }
}

// ================= PPG Task =================

static void ppg_sample_task(void *arg)
{
    while (1) {
        uint8_t samples = g_bioHub.num_samples_out_fifo();
        
        while (samples > 0) {
            BioData body = g_bioHub.read_sensor_bpm();
            g_latest_ppg = body;
            samples--;
        }
        
        vTaskDelay(pdMS_TO_TICKS(1));
    }
}

// ================= API =================

esp_err_t sensors::init()
{
    ESP_LOGI(TAG, "sensors::init enter");

    ESP_ERROR_CHECK(ppg_init_internal());
    ESP_ERROR_CHECK(mpu6050_init_internal());

    xTaskCreatePinnedToCore(imu_sample_task, "imu_sample", 2048, NULL, 6, NULL, 0);
    xTaskCreatePinnedToCore(ppg_sample_task, "ppg_sample", 4096, NULL, 7, NULL, 0);

    ESP_LOGI(TAG, "sensors::init done");
    return ESP_OK;
}