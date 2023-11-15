def convert_center_and_percentage_to_pixel_coordinates(center_percentage_coordinates, width_percentage,
                                                       height_percentage, image_width, image_height):
    x_percentage, y_percentage = center_percentage_coordinates

    x_center_pixel = x_percentage * image_width
    y_center_pixel = y_percentage * image_height
    width_pixel = width_percentage * image_width
    height_pixel = height_percentage * image_height

    x_pixel = x_center_pixel - (width_pixel / 2)
    y_pixel = y_center_pixel - (height_pixel / 2)

    return x_pixel, y_pixel, width_pixel, height_pixel


print(convert_center_and_percentage_to_pixel_coordinates((0.425, 0.5852), 0.1333, 0.2111, 1440, 1440))
