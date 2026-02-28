# Firmware — Multimodal Biosignal Acquisition

ESP32-S3 firmware for real-time acquisition of ECG, EMG, IMU, and PPG signals, streaming all data over USB as a binary frame protocol.

Built with [ESP-IDF](https://docs.espressif.com/projects/esp-idf/en/latest/) v5.5.1 on [Arduino Nano ESP32 (ABX00083)](https://docs.arduino.cc/hardware/nano-esp32/).

## Hardware

| Sensor | Signal | Sampling Rate | Interface |
|--------|--------|---------------|-----------|
| [BioAmp EXG Pill](https://github.com/upsidedownlabs/BioAmp-EXG-Pill) × 2 | ECG (chest + wrist) | 2000 Hz | ADC (12-bit) |
| [BioAmp EXG Pill](https://github.com/upsidedownlabs/BioAmp-EXG-Pill) × 2 | sEMG (flexor + extensor) | 2000 Hz | ADC (12-bit) |
| MPU6050 | IMU (accel + gyro, 6-axis) | 1000 Hz | I2C |
| MAX32664 | PPG (IR + Red + HR + SpO2) | 200 Hz | I2C |

### GPIO / ADC Mapping

| Signal | ADC Channel | GPIO |
|--------|-------------|------|
| ECG Chest | ADC1_CH1 | GPIO2 |
| ECG Wrist | ADC1_CH3 | GPIO4 |
| EMG 1 | ADC1_CH0 | GPIO1 |
| EMG 2 | ADC1_CH2 | GPIO3 |

I2C bus for MPU6050: SDA=GPIO11, SCL=GPIO12.

## Architecture

The firmware uses a dual-core design on the ESP32-S3 to avoid task conflicts:

```
Core 0                          Core 1
┌─────────────────┐             ┌──────────────────┐
│ imu_sample_task │ 1kHz        │ exg_sample_task  │ 2kHz ADC reads
│ ppg_sample_task │ FIFO read   │ tx_task          │ USB output
└────────┬────────┘             └────────┬─────────┘
         │                               │
         ▼                               ▼
   g_latest_imu              2kHz Timer Callback (ISR)
   g_latest_ppg              ├─ ExG frame every tick
                             ├─ IMU frame every 2 ticks
                             └─ PPG frame every 10 ticks
                                        │
                                        ▼
                                   Ring Buffer
                                        │
                                        ▼
                                 tx_task → USB
```

**Data flow:**
1. `exg_sample_task` reads 4 ADC channels at 2kHz into global variables
2. `imu_sample_task` reads MPU6050 at 1kHz, `ppg_sample_task` reads MAX32664 FIFO
3. A 2kHz hardware timer callback packs the latest samples into frames and pushes them to a ring buffer
4. `tx_task` drains the ring buffer in batches over USB Serial JTAG

## Binary Frame Protocol

All frames share a common header: `0xAA` + type byte. Transmitted over USB Serial JTAG at 921600 baud. This fixed-header design allows the stimulus program to parse and demultiplex the interleaved streams in real time by scanning for `0xAA` and dispatching on the type byte.

### ExG Frame (14 bytes) — Type `0x55` @ 2000 Hz

| Offset | Size | Field |
|--------|------|-------|
| 0 | 1 | Header `0xAA` |
| 1 | 1 | Type `0x55` |
| 2 | 4 | Sequence number (uint32) |
| 6 | 2 | ECG Chest (uint16, 12-bit ADC) |
| 8 | 2 | ECG Wrist (uint16) |
| 10 | 2 | EMG 1 (uint16) |
| 12 | 2 | EMG 2 (uint16) |

### IMU Frame (18 bytes) — Type `0x61` @ 1000 Hz

| Offset | Size | Field |
|--------|------|-------|
| 0 | 1 | Header `0xAA` |
| 1 | 1 | Type `0x61` |
| 2 | 4 | Sequence number (uint32) |
| 6 | 2 | Accel X (int16) |
| 8 | 2 | Accel Y (int16) |
| 10 | 2 | Accel Z (int16) |
| 12 | 2 | Gyro X (int16) |
| 14 | 2 | Gyro Y (int16) |
| 16 | 2 | Gyro Z (int16) |

### PPG Frame (20 bytes) — Type `0x62` @ 200 Hz

| Offset | Size | Field |
|--------|------|-------|
| 0 | 1 | Header `0xAA` |
| 1 | 1 | Type `0x62` |
| 2 | 4 | Sequence number (uint32) |
| 6 | 2 | Heart Rate (uint16) |
| 8 | 2 | SpO2 (uint16) |
| 10 | 1 | Confidence (uint8) |
| 11 | 1 | Status (uint8) |
| 12 | 4 | IR LED (uint32) |
| 16 | 4 | Red LED (uint32) |

All multi-byte fields are **little-endian** (native ESP32).

## Data Rate

| Stream | Frame Size | Rate | Bandwidth |
|--------|-----------|------|-----------|
| ExG | 14 B | 2000 Hz | 28.0 KB/s |
| IMU | 18 B | 1000 Hz | 18.0 KB/s |
| PPG | 20 B | 200 Hz | 4.0 KB/s |
| **Total** | | | **~50 KB/s** |

## Build & Flash

```bash
cd firmware
idf.py set-target esp32s3
idf.py build
idf.py flash
```

## Project Structure

```
firmware/
├── main/
│   └── main.cpp            # Entry point, ADC, timer, TX task
├── sensors/
│   ├── sensors.hpp          # Sensor API (init, global data)
│   └── sensors.cpp          # MPU6050 + MAX32664 drivers, I2C, sampling tasks
├── CMakeLists.txt
├── dependencies.lock        # MAX32664 driver auto-downloaded via idf.py build
└── sdkconfig
```