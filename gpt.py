import easyocr
import math


def translate_log(value, src_min=-5672, src_max=5672, dst_min=-32768, dst_max=32768, offset=0, scale=10):
    if value:
        multiplier = 1
        if (value < 0):
            multiplier = -1
            value = -value
        x = math.log(value+4, 0.25) * multiplier
        print(value, x)
        normalized_value = ((x + 5672) * 65536)/11344 - 32768
        return int(normalized_value)
    else:
        return 0


# print(translate_log(-480))
# print(translate_log(-233))
# print(translate_log(-133))
# print(translate_log(0))
# print(translate_log(133))
# print(translate_log(233))
# print(translate_log(480))

# print(math.log(480, 4)*1000)