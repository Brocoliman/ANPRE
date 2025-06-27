import os
import numpy as np
import fitz
import matplotlib.pyplot as plt


class PDFSet:
    """
    Container for one set of pdf files (either set of full cases, or set of sheets)
    """
    
    def __init__(self, data, load_format="none", print_events=True):
        self.data = data
        self.load_format = load_format
        self.print_events = print_events
        self.iter = iter(self.data)

    @classmethod
    def fromFolder(cls, src: str, load_format="none", print_events=True):
        """Finds valid PDF files from a source folder and stores list of objects and filenames

        Args:
            src (str): Source folder path
            load_format (str): Whether to load PDF object data, and if as PyMuPDF object (fitz) or numpy ndarray
                - none: do not load PDF object data
                - fitz: load into fitz pdf
                - numpyN: load into array with N dpi
        """
        data = []
        output = cls(data, load_format, print_events)

        output.src = src
        output.filenames_full = os.listdir(src)

        output.filenames = []
        output.filenames_removed = []
        for fn in output.filenames_full:
            if fn.endswith(".pdf"):
                output.filenames.append(fn)
            else:
                output.filenames_removed.append(fn)


        if load_format == "none":
            pass
        if load_format.startswith("fitz"):
            for i, filename in enumerate(output.filenames):
                path = os.path.join(src, filename)
                doc = fitz.open(path)
                data.append(doc)
                if output.print_events:
                    print(f"[TASK] Loading {i}: {filename} into fitz done.")
        if load_format.startswith("numpy"):
            for i, filename in enumerate(output.filenames):
                path = os.path.join(src, filename)
                doc = fitz.open(path)
                doc_pages = []
                for page in doc:
                    pix = page.get_pixmap(dpi=int(load_format[5:]))
                    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                        pix.height, pix.width, pix.n
                    )  # HWC
                    doc_pages.append(img)
                data.append(doc_pages)
                if print_events:
                    print(f"[TASK] Loading {i}: {filename} into numpy done.")
        
        return output
    
    def __len__(self):
        return len(self.filenames)

    def __iter__(self):
        return self.iter

    def get_colormaps(self, visualize=False) -> np.ndarray:
        """Find average color per page in list of case files

        Args:
            data (CaseSet): Source data containing list of list of numpy images
            visualize (bool, optional): Tool to visualize the colorma per doc and page. Defaults to False.
        """

        # Record average page colors (int, RGB)
        colormaps = np.ones((len(self), 300, 3), dtype=np.uint8)

        for idx, doc in enumerate(self):
            for i, img in enumerate(doc):
                avg = np.mean(img, axis=(0, 1))
                colormaps[idx, i] = avg
                if i >= 350:
                    break

            print(
                f"[EVENT] Colormapping {idx} done. {'Note: report exceeded 300 pages' if len(doc)>300 else ''}"
            )

        if visualize:
            plt.figure(figsize=(15, 2))
            plt.imshow(colormaps, interpolation="nearest")
            plt.title("Page average colors")
            plt.show()

        return colormaps
