import pytesseract
import pdf2image
import PIL

pdf2image.convert_from_path("Data/01.pdf")[0].save("01.jpg", "JPEG")

text = str(pytesseract.image_to_string(PIL.Image.open("01.jpg"))).split("\n")
text_without_spaces = []
for line in text:
    if not (line.isspace() or line == ""):
        text_without_spaces.append(line)

processed_text = " ".join(text_without_spaces[6:-3])

print(processed_text)