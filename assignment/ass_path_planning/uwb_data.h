#include<bits/stdc++.h>
typedef struct _serials {
        uint16_t len;
        std::shared_ptr<uint8_t[]> data_ptr;
    } serials;

class uwb_data{
    public:
        uwb_data(uint16_t self_id, uint16_t target_id, uint8_t CRC8);

        uwb_data(uint8_t *data, uint16_t size = 19) {
            memcpy(&header, data, 1);
            memcpy(&self_id, data + 1, 2);
            memcpy(&target_id, data + 3, 2);
            memcpy(&CRC8, data + 5, 1);
            uint16_t len = (uint16_t) (*(uint16_t *) (data + 6));
            this->data = std::vector<uint8_t>(len);
            memcpy(this->data.data(), data + 8, len);
            memcpy(&CRC16, data + 8 + len, 2);
            memcpy(&ender, data + len + 10, 1);
        }

        serials serialize();

        void set_data(std::vector<uint8_t> data);

        void set_data(const float x, const float y);

        std::vector<uint8_t> get_data() {
            return this->data;
        }

    private:
        uint8_t header;
        uint16_t self_id;
        uint16_t target_id;
        uint8_t CRC8;
        std::vector<uint8_t> data;
        uint16_t CRC16;
        uint8_t ender;
    };

uwb_data::uwb_data(uint16_t self_id, uint16_t target_id, uint8_t CRC8):
                        header(0x5A), self_id(self_id),target_id(target_id), CRC8(CRC8), ender(0x7F){
    this->data = std::vector<uint8_t>(8);
    this->CRC16 = 0x0F;
}

serials uwb_data::serialize() {
    uint16_t data_len = this->data.size();
    auto data = std::shared_ptr<uint8_t[]>(new uint8_t[11 + data_len]);
    uint32_t current_size = 0;

    auto pos = data.get();
    // pos[0] = '2';
    // pos[1] = '3';
    // pos[2] = '3';
    // pos[3] = '\n';
    current_size = (sizeof(uint16_t) + sizeof(uint8_t)) << 1;
    memcpy(pos, &(this->header), current_size);
    pos += current_size;

    current_size = sizeof(uint16_t);
    memcpy(pos, &data_len, current_size);
    pos += current_size;

    current_size = data_len * sizeof(uint8_t);
    memcpy(pos, this->data.data(), current_size);
    pos += current_size;

    current_size = sizeof(uint16_t);
    memcpy(pos, &(this->CRC16), current_size);
    pos += current_size;

    current_size = sizeof(uint8_t);
    memcpy(pos, &(this->ender), current_size);
    pos += current_size;

    return serials{
        .len = static_cast<uint16_t>(data_len + 11),
        .data_ptr = data
    };
}

void uwb_data::set_data(std::vector<uint8_t> data) {
    this->data = std::move(data);
}

void uwb_data::set_data(const float x, const float y) {
    data = std::vector<uint8_t>(8);
    std::memcpy(&data[0], &x, sizeof(float));
    std::memcpy(&data[4], &y, sizeof(float));
}