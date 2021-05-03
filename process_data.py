import pytesseract
import pdf2image
import PIL

def pdf(file):
    pdf2image.convert_from_path(file)[0].save("process.jpg", "JPEG")

    text = str(pytesseract.image_to_string(PIL.Image.open("01.jpg"))).split("\n")
    text_without_spaces = []
    for line in text:
        if not (line.isspace() or line == ""):
            text_without_spaces.append(line)

    return " ".join(text_without_spaces[6:-3])

print(pdf("Data/sample.pdf"))