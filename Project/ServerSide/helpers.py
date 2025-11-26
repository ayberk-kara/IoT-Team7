import constants

def determine_floor(altitude):
    for lower_bound, upper_bound, floor_label in constants.FLOOR_RANGES:
        if lower_bound <= altitude < upper_bound:
            return floor_label
    return "Unknown"

def get_element(data, key):
    for element in data:
        if element.get("name") == key:
            return element
    return None