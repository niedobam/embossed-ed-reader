import numpy as np
import cv2
import os


template_folder_path = "./assets/templates/"
template_table = {
    "template_0.bmp": 0,
    "template_1.bmp": 1,
    "template_2.bmp": 2,
    "template_3.bmp": 4,
    "template_4.bmp": 1,
    "template_5.bmp": 0,
    "template_6.bmp": 2,
    "template_7.bmp": 6,
}


def load_pictures(folder_path):
    files = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            files.append(file_path)
    return sorted(files)


def show_image(image, frame_title):
    cv2.imshow(frame_title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    image_folder = "./assets/images/"
    image_buffer = load_pictures(image_folder)

    for image in image_buffer:
        image_original = cv2.imread(image)
        image_grayscale = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

        identifiers = []
        all_rectangles = []

        for template_path, value in template_table.items():

            template = cv2.imread(f"{template_folder_path}{template_path}",
                                  cv2.IMREAD_GRAYSCALE)

            template_width, template_height = template.shape[::-1]

            template_matching_method = cv2.TM_CCOEFF_NORMED
            result = cv2.matchTemplate(image_grayscale,
                                       template,
                                       template_matching_method)

            threshold = 0.778

            locations = np.where(result >= threshold)

            rectangles = []
            for point in zip(*locations[::-1]):
                for _ in range(2):
                    rectangles.append([int(point[0]),
                                       int(point[1]),
                                       template_width,
                                       template_height])

            grouped_rectangles, _ = cv2.groupRectangles(rectangles, 1, 0.5)
            all_rectangles.append(grouped_rectangles)
            for (x, y, w, h) in grouped_rectangles:
                identifiers.append((x, value))
                cv2.rectangle(image_original,
                              (x, y),
                              (x + w, y + h),
                              (0, 255, 0),
                              2)

        sorted_identifiers = sorted(identifiers, key=lambda x: x[0])

        unsorted_expiration_date = []
        for i in sorted_identifiers:
            unsorted_expiration_date.append(i[1])

        day = f"{unsorted_expiration_date[0]}{unsorted_expiration_date[1]}"
        month = f"{unsorted_expiration_date[2]}{unsorted_expiration_date[3]}"
        year = ''.join(str(num) for num in unsorted_expiration_date[4:8])

        sorted_expiration_date = f"{day} {month} {year}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 2
        color = (0, 0, 255)
        thickness = 3
        position = (30, 80)
        cv2.putText(image_original, sorted_expiration_date, position,
                    font, font_scale, color, thickness)

        show_image(image_original, "Result")


if __name__ == "__main__":
    main()
