from os import path
import requests

if __name__ == "__main__":
    root = path.abspath(path.dirname(__file__))
    root_url = "http://www.r0k.us/graphics/kodak/kodak/"

    # leave out portrait images
    for idx in (1, 2, 3, 5, 6, 7, 8, 11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24):
        image = f"kodim{idx:02d}.png"

        file = path.join(root, image)
        url = root_url + image

        print(f"Downloading content of {url} to {file}")

        with open(file, "wb") as fh:
            fh.write(requests.get(url).content)
