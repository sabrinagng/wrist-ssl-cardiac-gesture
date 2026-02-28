#pragma once

#include <stdint.h>
#include "esp_err.h"
#include "max32664.hpp"

namespace maxim {
    class BioData;
}
using BioData = maxim::BioData;

struct ImuRawData {
    int16_t ax, ay, az;
    int16_t gx, gy, gz;
    int16_t temp_raw;
};

extern ImuRawData   g_latest_imu;
extern BioData      g_latest_ppg;

namespace sensors {

// Initialize I2C + PPG + IMU
esp_err_t init();
}